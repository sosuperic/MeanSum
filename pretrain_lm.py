# pretrain_lm.py

"""
Pretrain language model
"""

import os
import time

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch import optim

from data_loaders.summ_dataset_factory import SummDatasetFactory
from models.custom_parallel import DataParallelModel, DataParallelCriterion
from models.mlstm import StackedLSTM, mLSTM, StackedLSTMEncoder
from models.nn_utils import move_to_cuda, setup_gpus, OptWrapper, calc_grad_norm, save_model, Batch
from project_settings import SAVED_MODELS_DIR, HParams, EDOC_ID, PAD_ID
from utils import save_run_data, create_argparse_and_update_hp, update_moving_avg


#######################################
#
# Utils
#
#######################################

def create_lm_data_iter(data, lm_seq_len):
    """

    Args:
        data: [batch_size, -1] tensor (e.g. result of batchify())
        lm_seq_len: int
        model_type: str (mlstm or transformer)

    Returns:
        iterator that returns Batch's

        batch.src = [batch_size, seq_len+1] tensor
        the mlstm model transposes this and does src.t()[t] for every time step, predicting src.t()[t+1]
    """
    nbatches = (data.size(1) - 2) // lm_seq_len + 1  # up to and including end of sequences
    for i in range(nbatches):
        start = i * lm_seq_len
        length = min(lm_seq_len, data.size(1) - start)  # + 1 for target
        batch = data.narrow(1, start, length).long()

        yield Batch(batch, trg=batch, pad=PAD_ID)


def copy_state(state):
    if isinstance(state, tuple):
        # return (Variable(state[0].data), Variable(state[1].data))
        # Detach from graph (otherwise computation graph goes across batches, and
        # backward() will be called twice). Need to set grad to true after cloning
        # because clone() uses same requires_grad (False after detach)
        return (state[0].detach().clone().requires_grad_(),
                state[1].detach().clone().requires_grad_())
    else:
        return state.detach().clone().requires_grad_()


#######################################
#
# Train
#
#######################################

class LanguageModel(object):
    def __init__(self, hp, opt, save_dir):
        self.hp = hp
        self.opt = opt
        self.save_dir = save_dir

    def run_epoch(self, data_iter, nbatches, epoch, split, optimizer=None, tb_writer=None):
        """

        Args:
            data_iter: Pytorch DataLoader
            nbatches: int (number of batches in data_iter)
            epoch: int
            split: str ('train', 'val')
            optimizer: Wrapped optim (e.g. OptWrapper, NoamOpt)
            tb_writer: Tensorboard SummaryWriter

        Returns:
            1D tensor containing average loss across all items in data_iter
        """

        loss_avg = 0
        n_fwds = 0
        for s_idx, (texts, ratings, metadata) in enumerate(data_iter):
            start = time.time()

            # Add special tokens to texts
            x, lengths, labels = self.dataset.prepare_batch(texts, ratings,
                                                            doc_append_id=EDOC_ID)
            iter = create_lm_data_iter(x, self.hp.lm_seq_len)
            for b_idx, batch_obj in enumerate(iter):
                if optimizer:
                    optimizer.optimizer.zero_grad()

                #
                # Forward pass
                #
                if self.hp.model_type == 'mlstm':
                    # Note: iter creates a sequence of length hp.lm_seq_len + 1, and batch_obj.trg is all about the
                    # last token, while batch_obj.trg_y is all but the first token. They're named as such because
                    # the Batch class was originally designed for the Encoder-Decoder version of the Transformer, and
                    # the trg variables correspond to inputs to the Decoder.
                    batch = move_to_cuda(batch_obj.trg)  # it's trg because doesn't include last token
                    batch_trg = move_to_cuda(batch_obj.trg_y)
                    batch_size, seq_len = batch.size()

                    if b_idx == 0:
                        h_init, c_init = self.model.module.rnn.state0(batch_size) if self.ngpus > 1 \
                            else self.model.rnn.state0(batch_size)
                        h_init = move_to_cuda(h_init)
                        c_init = move_to_cuda(c_init)

                    # Forward steps for lstm
                    result = self.model(batch, h_init, c_init)
                    hiddens, cells, outputs = zip(*result) if self.ngpus > 1 else result

                    # Calculate loss
                    loss = 0
                    batch_trg = batch_trg.transpose(0, 1).contiguous()  # [seq_len, batch]
                    if self.ngpus > 1:
                        for t in range(len(outputs[0])):
                            # length ngpus list of outputs at that time step
                            loss += self.loss_fn([outputs[i][t] for i in range(len(outputs))], batch_trg[t])
                    else:
                        for t in range(len(outputs)):
                            loss += self.loss_fn(outputs[t], batch_trg[t])
                    loss_value = loss.item() / self.hp.lm_seq_len

                    # We only do bptt until lm_seq_len. Copy the hidden states so that we can continue the sequence
                    if self.ngpus > 1:
                        h_init = torch.cat([copy_state(hiddens[i][-1]) for i in range(self.ngpus)], dim=0)
                        c_init = torch.cat([copy_state(cells[i][-1]) for i in range(self.ngpus)], dim=0)
                    else:
                        h_init = copy_state(hiddens[-1])
                        c_init = copy_state(cells[-1])

                elif self.hp.model_type == 'transformer':
                    # This is the decoder only version now
                    logits = self.model(move_to_cuda(batch_obj.trg), move_to_cuda(batch_obj.trg_mask))
                    # logits: [batch, seq_len, vocab]
                    loss = self.loss_fn(logits, move_to_cuda(batch_obj.trg_y))
                    loss /= move_to_cuda(batch_obj.ntokens.float())  # normalize by number of non-pad tokens
                    loss_value = loss.item()
                    if self.ngpus > 1:
                        # With the custom DataParallel, there is no gather() and the loss is calculated per
                        # minibatch split on each GPU (see DataParallelCriterion's forward() -- the return
                        # value is divided by the number of GPUs). We simply undo that operation here.
                        # Also, note that the KLDivLoss in LabelSmoothing is already normalized by both
                        # batch and seq_len, as we use size_average=False to prevent any normalization followed
                        # by a manual normalization using the batch.ntokens. This oddity is because
                        # KLDivLoss does not support ignore_index=PAD_ID as CrossEntropyLoss does.
                        loss_value *= len(self.opt.gpus.split(','))

                #
                # Backward pass
                #
                gn = -1.0  # dummy for val (norm can't be < 0 anyway)
                if optimizer:
                    loss.backward()
                    gn = calc_grad_norm(self.model)  # not actually using this, just for printing
                    optimizer.step()
                loss_avg = update_moving_avg(loss_avg, loss_value, n_fwds + 1)
                n_fwds += 1

            # Print
            print_str = 'Epoch={}, batch={}/{}, split={}, time={:.4f} --- ' \
                        'loss={:.4f}, loss_avg_so_far={:.4f}, grad_norm={:.4f}'
            if s_idx % self.opt.print_every_nbatches == 0:
                print(print_str.format(
                    epoch, s_idx, nbatches, split, time.time() - start,
                    loss_value, loss_avg, gn
                ))
                if tb_writer:
                    # Step for tensorboard: global steps in terms of number of reviews
                    # This accounts for runs with different batch sizes
                    step = (epoch * nbatches * self.hp.batch_size) + (s_idx * self.hp.batch_size)
                    tb_writer.add_scalar('stats/loss', loss_value, step)

            # Save periodically so we don't have to wait for epoch to finish
            save_every = nbatches // 10
            if save_every != 0 and s_idx % save_every == 0:
                save_model(self.save_dir, self.model, self.optimizer, epoch, self.opt, 'intermediate')

        print('Epoch={}, split={}, --- '
              'loss_avg={:.4f}'.format(epoch, split, loss_avg))

        return loss_avg

    def train(self):
        """
        Main train loop
        """
        #
        # Get data, setup
        #

        self.dataset = SummDatasetFactory.get(self.opt.dataset)
        subwordenc = self.dataset.subwordenc
        train_iter = self.dataset.get_data_loader(split='train', n_docs=self.hp.n_docs, sample_reviews=True,
                                                  batch_size=self.hp.batch_size, shuffle=True)
        train_nbatches = train_iter.__len__()
        val_iter = self.dataset.get_data_loader(split='val', n_docs=self.hp.n_docs, sample_reviews=False,
                                                batch_size=self.hp.batch_size, shuffle=False)
        val_nbatches = val_iter.__len__()

        tb_path = os.path.join(self.save_dir, 'tensorboard/')
        print('Tensorboard events will be logged to: {}'.format(tb_path))
        os.mkdir(tb_path)
        os.mkdir(tb_path + 'train/')
        os.mkdir(tb_path + 'val/')
        self.tb_tr_writer = SummaryWriter(tb_path + 'train/')
        self.tb_val_writer = SummaryWriter(tb_path + 'val/')

        #
        # Get model and loss
        #
        if len(self.opt.load_model) > 0:
            raise NotImplementedError('Need to save run to same directory, handle changes in hp, etc.')
            # checkpoint = torch.load(opt.load_model)
            # self.model = checkpoint['model']
        else:
            if self.hp.model_type == 'mlstm':
                embed = nn.Embedding(subwordenc.vocab_size, self.hp.emb_size)
                lstm = StackedLSTM(mLSTM,
                                   self.hp.lstm_layers, self.hp.emb_size, self.hp.hidden_size,
                                   subwordenc.vocab_size,
                                   self.hp.lstm_dropout,
                                   layer_norm=self.hp.lstm_ln)
                self.model = StackedLSTMEncoder(embed, lstm)
                self.loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID)
            elif self.hp.model_type == 'transformer':
                self.model = make_model(subwordenc.vocab_size, subwordenc.vocab_size, N=self.hp.tsfr_blocks,
                                        d_model=self.hp.hidden_size, d_ff=self.hp.tsfr_ff_size,
                                        dropout=self.hp.tsfr_dropout, tie_embs=self.hp.tsfr_tie_embs,
                                        decoder_only=True)
                self.loss_fn = LabelSmoothing(size=subwordenc.vocab_size, smoothing=self.hp.tsfr_label_smooth)
        if torch.cuda.is_available():
            self.model.cuda()
        self.ngpus = 1
        if len(self.opt.gpus) > 1:
            self.ngpus = len(self.opt.gpus.split(','))
            self.model = DataParallelModel(self.model)
            self.loss_fn = DataParallelCriterion(self.loss_fn)

        n_params = sum([p.nelement() for p in self.model.parameters()])
        print('Number of parameters: {}'.format(n_params))

        #
        # Get optimizer
        #
        if self.hp.optim == 'normal':
            self.optimizer = OptWrapper(self.model, self.hp.lm_clip,
                                        optim.Adam(self.model.parameters(), lr=self.hp.lm_lr))
        elif self.hp.optim == 'noam':
            d_model = self.model.module.tgt_embed[0].d_model if self.ngpus > 1 else \
                self.model.tgt_embed[0].d_model
            self.optimizer = NoamOpt(d_model, 2, self.hp.noam_warmup,
                                     torch.optim.Adam(self.model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

        #
        # Train epochs
        #
        for e in range(hp.max_nepochs):
            try:
                self.model.train()
                loss_avg = self.run_epoch(train_iter, train_nbatches, e, 'train',
                                          optimizer=self.optimizer, tb_writer=self.tb_tr_writer)
                self.tb_tr_writer.add_scalar('overall_stats/loss_avg', loss_avg, e)

            except KeyboardInterrupt:
                print('Exiting from training early')

            self.model.eval()
            loss_avg = self.run_epoch(val_iter, val_nbatches, e, 'val', optimizer=None)
            self.tb_val_writer.add_scalar('overall_stats/loss_avg', loss_avg, e)
            save_model(self.save_dir, self.model, self.optimizer, e, self.opt, loss_avg)


if __name__ == '__main__':
    # Get hyperparams
    hp = HParams()
    hp, run_name, parser = create_argparse_and_update_hp(hp)

    # Add training language model args
    parser.add_argument('--dataset', default='yelp',
                        help='yelp,amazon')
    parser.add_argument('--save_model_fn', default='lm',
                        help="Model filename to save")
    parser.add_argument('--save_model_basedir', default=os.path.join(SAVED_MODELS_DIR, 'lm', '{}', '{}'),
                        help="Base directory to save different runs' checkpoints to")
    parser.add_argument('--load_model', default='',
                        help="Model filename to load")
    parser.add_argument('--print_every_nbatches', default=10,
                        help="Print stats every n batches")
    parser.add_argument('--gpus', default='0',
                        help="CUDA visible devices, e.g. 2,3")
    opt = parser.parse_args()

    # Create directory to store results and save run info
    save_dir = os.path.join(opt.save_model_basedir.format(hp.model_type, opt.dataset), run_name)
    save_run_data(save_dir, hp=hp)

    setup_gpus(opt.gpus, hp.seed)

    # Run
    lm = LanguageModel(hp, opt, save_dir)
    lm.train()
