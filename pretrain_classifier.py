# pretrain_classifier.py

"""
Train a text classifier (e.g. predict Yelp ratings)

Example usage:
python pretrain_classifier.py --model_type=cnn --clf_lr=0.0005 --cnn_n_feat_maps=256 --batch_size=128 --gpus=0
"""

from collections import OrderedDict, defaultdict
import os
import pdb
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch import optim

from data_loaders.summ_dataset_factory import SummDatasetFactory
from data_loaders.yelp_dataset import YelpDataset
from models.nn_utils import setup_gpus, OptWrapper, calc_grad_norm, save_model, calc_clf_acc, convert_to_onehot, \
    calc_per_rating_acc
from models.text_cnn import BasicTextCNN
from project_settings import SAVED_MODELS_DIR, HParams, DatasetConfig
from utils import save_run_data, create_argparse_and_update_hp, update_moving_avg, load_file


#######################################
#
# Train
#
#######################################

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, emb_size,
                 cnn_filter_sizes, cnn_n_feat_maps, cnn_dropout,
                 cnn_output_size, n_labels,
                 onehot_inputs=False, mse=False):

        super(TextClassifier, self).__init__()
        self.vocab_size = vocab_size
        self.onehot_inputs = onehot_inputs
        self.mse = mse  # treating classification as regression problem

        layers = [
            ('embed', nn.Embedding(vocab_size, emb_size)),
            ('cnn', BasicTextCNN(cnn_filter_sizes, cnn_n_feat_maps, emb_size, cnn_dropout))
        ]
        if mse:
            layers.append(('fc_out', nn.Linear(cnn_output_size, 1)))
        else:
            layers.append(('fc_out', nn.Linear(cnn_output_size, n_labels)))
        self.model = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        if self.onehot_inputs:
            if x.dim() == 2:
                x = convert_to_onehot(x, self.vocab_size)  # [batch, seq_len] -> [batch, seq_len, vocab]
            inp_emb = torch.matmul(x.float(), self.model.embed.weight)  # [batch, seq_len, emb_size]
            cnn_emb = self.model.cnn(inp_emb)
            logits = self.model.fc_out(cnn_emb)
        else:
            logits = self.model(x)

        return logits


class TextClassifierTrainer(object):
    def __init__(self, hp, opt, save_dir):
        self.hp = hp
        self.opt = opt
        self.save_dir = save_dir

    def run_epoch(self, data_iter, nbatches, epoch, split, optimizer=None, tb_writer=None, save_intermediate=True):
        """

        Args:
            data_iter: iterable providing minibatches
            nbatches: int (number of batches in data_iter)
            epoch: int
            split: str ('train', 'val')
            optimizer: Wrapped optim (e.g. OptWrapper)
            tb_writer: Tensorboard SummaryWriter
            save_intermediate: boolean (save intermediate checkpoints)

        Returns:
            1D tensor containing average loss across all items in data_iter
        """

        loss_avg = 0
        acc_avg = 0
        rating_diff_avg = 0

        per_rating_counts = defaultdict(int)
        per_rating_acc = defaultdict(int)

        for s, batch in enumerate(data_iter):
            start = time.time()
            if optimizer:
                optimizer.optimizer.zero_grad()

            texts, ratings, metadata = batch
            batch_size = len(texts)
            x, lengths, labels = self.dataset.prepare_batch(texts, ratings)

            #
            # Forward pass
            #
            logits = self.model(x)
            if self.hp.clf_mse:
                logits = logits.squeeze(1)  # [batch, 1] -> [batch]
                loss = self.loss_fn(logits, labels.float())
            else:
                loss = self.loss_fn(logits, labels)
            loss_value = loss.item()
            acc = calc_clf_acc(logits, labels).item()

            #
            # Backward pass
            #
            gn = -1.0  # dummy for val (norm can't be < 0 anyway)
            if optimizer:
                loss.backward()
                gn = calc_grad_norm(self.model)  # not actually using this, just for printing
                optimizer.step()

            #
            # Print etc.
            #
            loss_avg = update_moving_avg(loss_avg, loss_value, s + 1)
            acc_avg = update_moving_avg(acc_avg, acc, s + 1)
            print_str = 'Epoch={}, batch={}/{}, split={}, time={:.4f} --- ' \
                        'loss={:.4f}, loss_avg_so_far={:.4f}, acc={:.4f}, acc_avg_so_far={:.4f}, grad_norm={:.4f}'

            if self.hp.clf_mse:
                rating_diff = (labels - logits.round().long()).float().mean()
                rating_diff_avg = update_moving_avg(rating_diff_avg, rating_diff, s + 1)
                print_str += ', rating_diff={:.4f}, rating_diff_avg_so_far={:.4f}'.format(rating_diff, rating_diff_avg)

                true_ratings = labels + 1
                pred_ratings = logits.round() + 1
                probs = torch.ones(batch_size)  # dummy
                per_rating_counts, per_rating_acc = calc_per_rating_acc(pred_ratings, true_ratings,
                                                                        per_rating_counts, per_rating_acc)
            else:
                true_ratings = labels + 1
                probs, max_idxs = torch.max(F.softmax(logits, dim=1), dim=1)
                pred_ratings = max_idxs + 1
                per_rating_counts, per_rating_acc = calc_per_rating_acc(pred_ratings, true_ratings,
                                                                        per_rating_counts, per_rating_acc)

            if s % self.opt.print_every_nbatches == 0:
                print(print_str.format(
                    epoch, s, nbatches, split, time.time() - start,
                    loss_value, loss_avg, acc, acc_avg, gn
                ))
                print('Review: {}'.format(texts[0]))
                print('True rating: {}'.format(true_ratings[0]))
                print('Predicted rating: {}'.format(pred_ratings[0]))
                print('Predicted rating probability: {:.4f}'.format(probs[0]))
                print('Per rating accuracy: {}'.format(dict(per_rating_acc)))

                if tb_writer:
                    # Global steps in terms of number of items
                    # This accounts for runs with different batch sizes
                    step = (epoch * nbatches * self.hp.batch_size) + (s * self.hp.batch_size)
                    tb_writer.add_scalar('loss/batch_loss', loss_value, step)
                    tb_writer.add_scalar('loss/avg_loss', loss_avg, step)
                    tb_writer.add_scalar('acc/batch_acc', acc, step)
                    tb_writer.add_scalar('acc/avg_acc', acc_avg, step)
                    if self.hp.clf_mse:
                        tb_writer.add_scalar('rating_diff/batch_diff', rating_diff, step)
                        tb_writer.add_scalar('rating_diff/avg_diff', rating_diff_avg, step)

                    tb_writer.add_text('predictions/review', texts[0], step)
                    tb_writer.add_text('predictions/true_pred_prob',
                                       'True={}, Pred={}, Prob={:.4f}'.format(
                                           true_ratings[0], pred_ratings[0], probs[0]),
                                       step)
                    for r, acc in per_rating_acc.items():
                        tb_writer.add_scalar('acc/curavg_per_rating_acc_{}'.format(r), acc, step)


            # Save periodically so we don't have to wait for epoch to finish
            if save_intermediate:
                save_every = nbatches // 10
                if save_every != 0 and s % save_every == 0:
                    model_to_save = self.model.module if len(self.opt.gpus) > 1 else self.model
                    save_model(self.save_dir, model_to_save, self.optimizer, epoch, self.opt, 'intermediate')

        print_str = 'Epoch={}, split={}, --- ' \
              'loss_avg={:.4f}, acc_avg={:.4f}, per_rating_acc={}'.format(
            epoch, split, loss_avg, acc_avg, dict(per_rating_acc))
        if self.hp.clf_mse:
            print_str += ', rating_diff_avg={:.4f}'.format(rating_diff_avg)
        print(print_str)

        return loss_avg, acc_avg, rating_diff_avg, per_rating_acc

    def train(self):
        """
        Main train loop
        """
        #
        # Get data, setup
        #

        # NOTE: Use n_docs=1 so we can classify one review
        self.dataset = SummDatasetFactory.get(self.opt.dataset)
        train_iter = self.dataset.get_data_loader(split='train', sample_reviews=True, n_docs=1,
                                                  batch_size=self.hp.batch_size, shuffle=True)
        val_iter = self.dataset.get_data_loader(split='val', sample_reviews=False, n_docs=1,
                                                batch_size=self.hp.batch_size, shuffle=False)

        self.tb_tr_writer = None
        self.tb_val_writer = None
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
        if len(self.opt.load_train_model) > 0:
            raise NotImplementedError('Need to save run to same directory, handle changes in hp, etc.')
            # checkpoint = torch.load(opt.load_model)
            # self.model = checkpoint['model']
        else:
            if self.hp.model_type == 'cnn':
                cnn_output_size = self.hp.cnn_n_feat_maps * len(self.hp.cnn_filter_sizes)
                self.model = TextClassifier(self.dataset.subwordenc.vocab_size, self.hp.emb_size,
                                            self.hp.cnn_filter_sizes, self.hp.cnn_n_feat_maps, self.hp.cnn_dropout,
                                            cnn_output_size, self.dataset.n_ratings_labels,
                                            onehot_inputs=self.hp.clf_onehot, mse=self.hp.clf_mse)

        if self.hp.clf_mse:
            self.loss_fn = nn.MSELoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            self.model.cuda()
        if len(self.opt.gpus) > 1:
            self.model = nn.DataParallel(self.model)

        n_params = sum([p.nelement() for p in self.model.parameters()])
        print('Number of parameters: {}'.format(n_params))

        #
        # Get optimizer
        #
        self.optimizer = OptWrapper(
            self.model,
            self.hp.clf_clip,
            optim.Adam(self.model.parameters(), lr=self.hp.clf_lr))

        #
        # Train epochs
        #
        for e in range(hp.max_nepochs):
            try:
                self.model.train()
                loss_avg, acc_avg, rating_diff_avg, per_rating_acc = self.run_epoch(
                    train_iter, train_iter.__len__(), e, 'train',
                    optimizer=self.optimizer, tb_writer=self.tb_tr_writer)
                self.tb_tr_writer.add_scalar('overall/loss', loss_avg, e)
                self.tb_tr_writer.add_scalar('overall/acc', acc_avg, e)
                self.tb_tr_writer.add_scalar('overall/rating_diff', rating_diff_avg, e)
                for r, acc in per_rating_acc.items():
                    self.tb_tr_writer.add_scalar('overall/per_rating_acc_{}_stars'.format(r), acc, e)
            except KeyboardInterrupt:
                print('Exiting from training early')

            self.model.eval()
            loss_avg, acc_avg, rating_diff_avg, per_rating_acc = self.run_epoch(
                val_iter, val_iter.__len__(), e, 'val', optimizer=None)
            self.tb_val_writer.add_scalar('overall/loss', loss_avg, e)
            self.tb_val_writer.add_scalar('overall/acc', acc_avg, e)
            self.tb_val_writer.add_scalar('overall/rating_diff', rating_diff_avg, e)
            for r, acc in per_rating_acc.items():
                self.tb_val_writer.add_scalar('overall/per_rating_acc_{}'.format(r), acc, e)
            fn_str = 'l{:.4f}_a{:.4f}_d{:.4f}'.format(loss_avg, acc_avg, rating_diff_avg)
            model_to_save = self.model.module if len(self.opt.gpus) > 1 else self.model
            save_model(self.save_dir, model_to_save, self.optimizer, e, self.opt, fn_str)

    def test(self):
        """
        Run trained model on test set
        """
        #
        # Setup data, logging
        #
        self.dataset = SummDatasetFactory.get(self.opt.dataset)
        test_iter = self.dataset.get_data_loader(split='test', sample_reviews=False, n_docs=1,
                                                 batch_size=self.hp.batch_size, shuffle=False)

        tb_path = os.path.join(self.save_dir, 'tensorboard/test/')
        if not os.path.exists(tb_path):
            os.mkdir(tb_path)
        self.tb_test_writer = SummaryWriter(tb_path)

        #
        # Get model and loss
        #
        self.model = torch.load(opt.load_test_model)['model']
        if self.hp.clf_mse:
            self.loss_fn = nn.MSELoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            self.model.cuda()
        if len(self.opt.gpus) > 1:
            self.model = nn.DataParallel(self.model)

        n_params = sum([p.nelement() for p in self.model.parameters()])
        print('Number of parameters: {}'.format(n_params))

        #
        # Test
        #
        self.model.eval()
        with torch.no_grad():
            loss_avg, acc_avg, rating_diff_avg, per_rating_acc = self.run_epoch(
                test_iter, test_iter.__len__(), 0, 'test',
                tb_writer=self.tb_test_writer, save_intermediate=False)
        self.tb_test_writer.add_scalar('overall/loss', loss_avg, 0)
        self.tb_test_writer.add_scalar('overall/acc', acc_avg, 0)
        self.tb_test_writer.add_scalar('overall/rating_diff', rating_diff_avg, 0)
        for r, acc in per_rating_acc.items():
            self.tb_test_writer.add_scalar('overall/per_rating_acc_{}_stars'.format(r), acc, 0)


if __name__ == '__main__':
    # Get hyperparams
    hp = HParams()
    hp, run_name, parser = create_argparse_and_update_hp(hp)

    # Add training language model args
    parser.add_argument('--dataset', default='yelp',
                        help='yelp,amazon')

    parser.add_argument('--save_model_basedir', default=os.path.join(SAVED_MODELS_DIR, 'clf', '{}', '{}'),
                        help="Base directory to save different runs' checkpoints to")
    parser.add_argument('--save_model_fn', default='clf',
                        help="Model filename to save")
    parser.add_argument('--load_train_model', default='',
                        help="Path to model to finetune (not implemented)")

    parser.add_argument('--print_every_nbatches', default=50,
                        help="Print stats every n batches")
    parser.add_argument('--gpus', default='0',
                        help="CUDA visible devices, e.g. 2,3")

    parser.add_argument('--mode', default='train',
                        help="train or test")
    parser.add_argument('--load_test_model', default=None,
                        help="Path to model to test")
    opt = parser.parse_args()

    setup_gpus(opt.gpus, hp.seed)

    if opt.mode == 'train':
        # Create directory to store results and save run info
        save_dir = os.path.join(opt.save_model_basedir.format(hp.model_type, opt.dataset), run_name)
        save_run_data(save_dir, hp=hp)

        # Run
        clf = TextClassifierTrainer(hp, opt, save_dir)
        clf.train()

    elif opt.mode == 'test':
        # Get directory model was saved in. Will be used to save tensorboard test results to
        if opt.load_test_model is None:
            opt.load_test_model = DatasetConfig(opt.dataset).clf_path
        save_dir = os.path.dirname(opt.load_test_model)

        # Run
        clf = TextClassifierTrainer(hp, opt, save_dir)
        clf.test()
