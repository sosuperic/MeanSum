# nn_utils.py

import math
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from project_settings import PAD_ID
from utils import load_file, update_moving_avg


################################################################################################
#
# Setup
#
################################################################################################


def move_to_cuda(x):
    """Move tensor to cuda"""
    if torch.cuda.is_available():
        if type(x) == tuple:
            x = tuple([t.cuda() for t in x])
        else:
            x = x.cuda()
    return x


def setup_gpus(gpu_ids, seed):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


################################################################################################
#
# Running experiments
#
################################################################################################
def save_model(save_dir, model, optimizer, epoch, opt, extra_fn):
    """
    Args:
        save_dir: str (path to directory)
        model: nn.Module
        optimizer: wrapped Optimizer instance
        epoch: int
        opt: argparse (options)
        extra_fn: append to filename, e.g. <loss_avg>
    """
    checkpoint = {
        'model': model,
        'optimizer': optimizer,
        'epoch': epoch,
        'opt': opt
    }
    save_fn = '{}_e{}_{:.2f}.pt' if type(extra_fn) == float else '{}_e{}_{}.pt'
    save_fn = save_fn.format(opt.save_model_fn, epoch, extra_fn)
    save_path = os.path.join(save_dir, save_fn)
    print('Saving to: {}'.format(save_path))
    torch.save(checkpoint, save_path)


def save_models(save_dir, models, optimizers, epoch, opt, extra_fn):
    """
    Save multiple models and optimizers. Currently used by summarization model, which consists of
    a language model, a discriminator, and a classifier.
    Args:
        save_dir: str (path to directory)
        models: dict
            key: str (name)
            value: some object
        optimizers: dict
            key: str (name)
            value: wrapped Optimizer instance
        epoch: int
        opt: argparse (options)
        extra_fn: append to filename, e.g. <loss_avg>
    """
    checkpoint = {
        'epoch': epoch,
        'opt': opt
    }
    for name, model in models.items():
        checkpoint[name] = model
    for name, optimizer in optimizers.items():
        checkpoint[name] = optimizer
    save_fn = '{}_e{}_{:.2f}.pt' if type(extra_fn) == float else '{}_e{}_{}.pt'
    save_fn = save_fn.format(opt.save_model_fn, epoch, extra_fn)
    save_path = os.path.join(save_dir, save_fn)
    print('Saving to: {}'.format(save_path))
    torch.save(checkpoint, save_path)


def update_hp_from_loaded_model(load_path, hp=None, exclude=None, include_match=None):
    """
    Args:
        load_path: str
    Returns:
    """
    old_hp = load_file(os.path.join(os.path.dirname(load_path), 'hp.json'))
    for name, value in old_hp.items():
        # hp[name] = value
        # TODO: fix this
        if (name not in exclude) and (name in include):
            setattr(hp, name, value)
    return hp


################################################################################################
#
# Decoding utils
#
################################################################################################
def logits_to_prob(logits, method,
                   tau=1.0, eps=1e-10, gumbel_hard=False):
    """
    Args:
        logits: [batch_size, vocab_size]
        method: 'gumbel', 'softmax'
        gumbel_hard: boolean
        topk: int (used for beam search)
    Returns: [batch_size, vocab_size]
    """
    if tau == 0.0:
        raise ValueError(
            'Temperature should not be 0. If you want greedy decoding, pass "greedy" to prob_to_vocab_id()')
    if method == 'gumbel':
        prob = F.gumbel_softmax(logits, tau=tau, eps=eps, hard=gumbel_hard)
    elif method == 'softmax':
        prob = F.softmax(logits / tau, dim=1)
    return prob


def prob_to_vocab_id(prob, method, k=1):
    """
    Produce vocab id given probability distribution over vocab
    Args:
        prob: [batch_size, vocab_size]
        method: str ('greedy', 'sample')
        k: int (used for beam search)
    Returns:
        prob: [batch_size * k, vocab_size]
            Rows are repeated:
                [[0.3, 0.2, 0.5],
                 [0.1, 0.7, 0.2]]
            Becomes (with k=2):
                [[0.3, 0.2, 0.5],
                 [0.3, 0.2, 0.5],
                 [0.1, 0.7, 0.2]
                 [0.1, 0.7, 0.2]]
        ids: [batch_size * k] LongTensor
    """
    if method == 'greedy':
        _, ids = torch.topk(prob, k, dim=1)
    elif method == 'sample':
        ids = torch.multinomial(prob, k)
    batch_size = prob.size(0)
    prob = prob.repeat(1, k).view(batch_size * k, -1)
    ids = ids.view(-1)
    return prob, ids


################################################################################################
#
# Gradient utils
#
################################################################################################
def calc_grad_norm(model):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        if p.requires_grad and (p.grad is not None):
            modulenorm = p.grad.norm()
            totalnorm += modulenorm ** 2
    return math.sqrt(totalnorm)


def clip_gradient(model, clip):
    """Clip the gradient."""
    for p in model.parameters():
        if p.requires_grad and (p.grad is not None):
            p.grad = p.grad.clamp(-clip, clip)


def freeze(module):
    for param in module.parameters():
        param.requires_grad = False


################################################################################################
#
# Loss functions, accuracies
#
################################################################################################
class LabelSmoothing(nn.Module):
    """"
    We implement label smoothing using the KL div loss.
    Instead of using a one-hot target distribution, we create a distribution
    that has confidence of the correct word and the rest of the smoothing
    mass distributed throughout the vocabulary.
    https://arxiv.org/pdf/1512.00567.pdf
    Basically combination of one-hot and uniform
    """

    def __init__(self, size, padding_idx=PAD_ID, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        # False so that we can normalize by the number of non-pad tokens after
        # (nn.CrossEntropyLoss() has a ignore_index=0 for this, but not KLDivLoss)
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size  # number of labels, e.g. vocab size
        self.true_dist = None

    def forward(self, x, target):
        """
        Args:
            x: [batch_size, seq_len, self.size (num_labels)]
            target: [batch_size. seq_len]
        """
        x = x.contiguous().view(-1, x.size(-1))
        target = target.contiguous().view(-1)
        x = F.log_softmax(x, dim=-1)
        assert x.size(1) == self.size
        # true_dist = x.clone()
        true_dist = x.detach().clone()
        true_dist.fill_(self.smoothing / (self.size - 2))  # smoothing mass distributed through vocab
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)  # correct word has confidence
        true_dist[:, self.padding_idx] = 0  # replace all indices that have pad with 0? but why's it just one value
        mask = torch.nonzero(target == self.padding_idx)
        # if no values in target == padding_idx, returns tensor([]) which has dim() == 1
        if mask.dim() > 1:
            # if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        loss = self.criterion(x, true_dist)
        return loss


def calc_clf_acc(logits, labels):
    """
    Args:
        logits:
            [batch_size, n_labels] FloatTensor or
            [batch_size] FloatTensor (i.e. when classifier is being trained with MSE)
        labels: [batch_size] LongTensor
    Returns: float
    """
    batch_size = logits.size(0)
    if logits.dim() == 1:
        indices = logits.round().long()
    else:
        _, indices = torch.max(logits, dim=1)
    acc = torch.eq(indices, labels).sum().float() / batch_size
    return acc

def calc_per_rating_acc(pred_ratings, true_ratings, per_rating_counts, per_rating_acc):
    """
    Calculate the accuracy of each star rating

    Args:
        pred_ratings: 1D Tensor (e.g. batch_size)
        true_ratings: 1D Tensor (e.g. batch_size)
        per_rating_counts: dict: rating to int
        per_rating_acc: dict: rating to float

    Returns:
        Updated per_rating_counts and per_rating_acc
    """
    for b_idx in range(true_ratings.size(0)):
        true_rating, pred_rating = true_ratings[b_idx].item(), pred_ratings[b_idx].item()
        per_rating_counts[true_rating] += 1
        avg_so_far = per_rating_acc[true_rating]
        item_acc = true_rating == pred_rating
        rating_count = per_rating_counts[true_rating]
        per_rating_acc[true_rating] = update_moving_avg(avg_so_far, item_acc, rating_count)

    return per_rating_counts, per_rating_acc


def classify_summ_batch(clf_model, summary_batch, ratings_batch, dataset,
                        per_rating_counts=None, per_rating_acc=None):
    """
    Used to evaluate models on test set (run_evaluations.py, test() in train_sum.py)

    Args:
        clf_model: pretrained TextClassifier
        summary_batch: list of strs (length batch_size)
        ratings_batch: list of 1D tensors (from data_iter, length batch_size)
        dataset: YelpDataset instance
            (Really prepare_batch() should be a static method... or not part of YelpDataset. This should've been
            refactored at some point.)
        per_rating_counts: dict: rating to int
        per_rating_acc: dict: rating to acc

    Returns:
        acc: 0D float Tensor
        per_rating_acc: dict: int to acc
        pred_ratings: [batch_size] Tensor
        pred_probs: [batch_size] Tensor
    """
    docs_ids, _, labels = dataset.prepare_batch(summary_batch, ratings_batch)

    if (per_rating_counts is None) and (per_rating_acc) is None:
        per_rating_counts = defaultdict(int)
        per_rating_acc = defaultdict(int)

    try:
        logits = clf_model(docs_ids)
        acc = calc_clf_acc(logits, labels)
        pred_probs, pred_ratings = torch.max(F.softmax(logits, dim=1), dim=1)
        pred_ratings = pred_ratings + 1  # [batch]
        true_ratings = labels + 1
        per_rating_counts, per_rating_acc = calc_per_rating_acc(pred_ratings, true_ratings,
                                                                per_rating_counts, per_rating_acc)
    except Exception as e:  # summary is too short for CNN-based classifier, which has 3,4,5 width kernels
        acc, pred_ratings, pred_probs = None, None, None

    return acc, per_rating_counts, per_rating_acc, pred_ratings, pred_probs


def calc_lm_nll(lm, input):
    """
    Calculate the negative log likelihood of the input according to a language model lm

    Args:
        lm: StackedLSTMEncoder (trained language model)
        input:
            [batch_size, seq_len] (e.g. text prepared using prepare_batch())

    Returns: 0D float tensor
    """
    batch_size = input.size(0)

    # Encode with fixed lm
    h_init, c_init = lm.rnn.state0(batch_size)
    h_init, c_init = move_to_cuda(h_init), move_to_cuda(c_init)
    _, _, outputs = lm(input, h_init, c_init)

    # Calculate the number of non-pad tokens so that we can calculate the NLL using
    # only the tokens up to and including the EDOC token
    n_nonpads = (input != PAD_ID).sum(dim=1).float()  # [batch]
    zero_logprobs = move_to_cuda(torch.zeros(batch_size))
    logprobs = move_to_cuda(torch.zeros(batch_size))

# So this n_nonpads issue means we go longer than than expected.
    # But we calculate max_logprobs as F.softmax(output of feeding in zeros into encoder))
    # ( which may be 0's? I'm not sure). And with vocab size that large... if anything it's worse than it hsould be?

    # Keeping outputs as a list (as opposed to concatenating into a tensor) to reduce memory
    for t, output in enumerate(outputs):  # list of [batch, vocab]
        max_logprobs = torch.log(torch.max(F.softmax(output, dim=1), dim=1)[0])
        max_logprobs = torch.where(t < n_nonpads, max_logprobs, zero_logprobs)
        logprobs += max_logprobs
    logprobs /= n_nonpads
    mean_nll = -logprobs.mean()

    return mean_nll

################################################################################################
#
# Optimizer wrappers
#
################################################################################################
class NoamOpt:
    """
    Optim wrapper for learning rate schedule. Note: __init__ takes an optimizer. However,
    the learning rate used to initialize that optimizer is not used.
    Section 5.3, Equation 3
    """

    def __init__(self, model_size, factor=2, warmup=4000, optimizer=None):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        """Update parameters and rate"""
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        """Implement `lrate` above"""
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))


class OptWrapper(object):
    """
    Wrapper around optimizer. Implements learning rate schedule. Passed to run_epoch()
    """

    def __init__(self, model, clip, optimizer,
                 epoch_nsteps=None, epoch_decay=None, decay_method='times'):
        """
        Args:
            model: nn.Module (mLSTM)
            clip: float (value at which to clip)
            optimizer: optim instasnce
            epoch_nsteps: number of steps in an epoch
                - can be used to update learning rate after epoch is done
            epoch_decay: float (amount to decay by)
            decay_method: str ('times', 'minus')
        """
        self.model = model
        self.clip = clip
        self.optimizer = optimizer
        self.epoch_nsteps = epoch_nsteps
        self.epoch_decay = epoch_decay
        self.decay_method = decay_method
        self._nstep = 0

    def step(self):
        self._nstep += 1
        clip_gradient(self.model, self.clip)
        self.optimizer.step()
        # Note: torch.optim now actually contains learning rate schedulers
        # However, they do not do gradient clipping
        # update learning rate at end of epoch potentially
        if (self.epoch_nsteps) and (self._nstep == self.epoch_nsteps):
            print('Decaying learning rate by {} and {}'.format(self.epoch_decay, self.decay_method))
            for p in self.optimizer.param_groups:
                if self.decay_method == 'times':
                    p['lr'] *= self.epoch_decay
                elif self.decay_method == 'minus':
                    p['lr'] -= self.epoch_decay


class StepAnnealer(object):
    """
    Generic wrapper around a value. Lowers the value every n steps.
    This can be used, for example, for annealing the temperature over time.
    """

    def __init__(self, init_val,
                 interval_size=None, intervals=None, intervals_vals=None,
                 alpha=None, method=None,
                 min_val=None):
        """
        Args:
            init_val: float (starting value)
            One of these two should be set:
            interval_size: int (update the value every interval_size steps)
            intervals: list of ints (update the value at these steps)
            intervals_vals: list of floats (if intervals is given, decay to these values)
            alpha: float (amount to decay by)
            method: str
                'times': multiply the value by alpha when decaying
                'minus': subtract alpha from the current value when decaying
            min_val: float (lowest the value can become)
        """
        self.init_val = init_val
        self.interval_size = interval_size
        self.intervals = intervals
        self.intervals_vals = intervals_vals
        self.cur_interval = 0
        self.alpha = alpha
        self.method = method
        self.min_val = min_val
        self.cur_step = 0
        self.val = init_val

    def step(self):
        self.cur_step += 1
        if self.interval_size:
            if self.cur_step % self.interval_size == 0:
                self.update_val()
        elif self.intervals:
            if self.cur_interval <= len(self.intervals) - 1:
                if self.cur_step == self.intervals[self.cur_interval]:
                    self.update_val()
            self.cur_interval += 1

    def update_val(self):
        if self.intervals_vals:
            self.cur_val = self.intervals_vals[self.cur_interval]
        else:
            if self.method == 'times':
                new_val = self.val * self.alpha
            elif self.method == 'minus':
                new_val = self.val - self.alpha
            self.val = max(new_val, self.min_val)


################################################################################################
#
# Data (mostly for Transformer, but Batch could theoretically be used for anything
#
################################################################################################
def convert_to_onehot(x, size):
    """
    Convert a dense vector of ints into a one hot representation
    Args:
        x: [batch_size, seq_len] LongTensor
        size: int (number of labels, e.g. vocab_size)
    Returns:
        [batch_size, seq_len, size] LongTensor
    """
    x = x.unsqueeze(-1)  # [batch, seq_len, 1]
    x_onehot = torch.LongTensor(x.size(0), x.size(1), size).zero_()  # [batch, seq_len, size]
    x_onehot = move_to_cuda(x_onehot)
    x_onehot.scatter_(2, x, 1)  # 2nd dimension, indices, fill with 1's
    return x_onehot


def convert_onehot_to_dense(onehot):
    """
    Args:
        onehot: [batch_size, seq_len, vocab_size]
    Returns: [batch_size, seq_len]
    """
    return torch.argmax(onehot, dim=2)


#
# Mask, Transformer
#
def subsequent_mask(size):
    """
    Mask out subsequent positions, e.g. on target
    Args:
        size: int (e.g. seq_len)
    Returns:
        [1, size, size] ByteTensor
        Example:
            [[[1, 0, 0],
              [1, 1, 0],
              [1, 1, 1]]]
    """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')  # upper triangle are 1's
    subsequent_mask = torch.from_numpy(subsequent_mask) == 0
    return subsequent_mask


class Batch:
    """
    Object for holding a batch of data with mask during training.
    """

    def __init__(self, src, trg=None, pad=PAD_ID):
        """
        Args:
            src: [batch_size, src_seq_len]
            trg: [batch_size, trg_seq_len]
            pad: int (id of pad token)
        """
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)  # [batch, 1, src_seq_len]
        if trg is not None:
            # "This masking, ***combined with the fact that the output embeddings are offset by one position***,
            # ensures that the predictions for position i can depend only only on the known outputs at
            # positions less than i"
            #
            # I.e. The targets are inputs into the decoder. If we didn't use the subsequent mask,
            # we would be able to use future tokens to predict the i-th token.
            # If we didn't offset by one position, then the i-th token would be used to predict the
            # i-th token.
            #
            # In order to offset by one position, we set trg (the inputs to the decoder) up until
            # but not including the last token (:-1), as the last token would not have a next token
            # to predict. We also set trg_y (the targets used to calculate the loss with the output
            # probabilities of the decoder) to 1:, i.e. the next tokens.
            self.trg = trg[:, :-1]  # [batch, trg_seq_len - 1]
            self.trg_y = trg[:, 1:]  # [batch, trg_seq_len - 1]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        """
        Create a mask to hide padding and future words.
        Args:
            tgt: [batch_size, seq_len] LongTensor
            pad: int (id of pad token)
        """
        # mask padding
        tgt_mask = (tgt != pad).unsqueeze(-2)  # [batch_size, 1, seq_len]
        # mask future words
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask)  # [batch_size, seq_len, seq_len]
        return tgt_mask
