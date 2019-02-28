# generate_from_lm.py

"""
Load a trained language model and generate text

Example usage:
PYTHONPATH=. python generate_from_lm.py \
--init="Although the food" --tau=0.5 \
--sample_method=gumbel --g_eps=1e-5
--load_model=<path/model.pt>
"""
import pdb

import torch
import torch.nn as nn

from models.custom_parallel import DataParallelModel
from models.mlstm import StackedLSTMEncoderDecoder
from models.nn_utils import move_to_cuda, setup_gpus, logits_to_prob, prob_to_vocab_id
from project_settings import HParams, PAD_ID, DatasetConfig
from utils import load_file, create_argparse_and_update_hp

#######################################
#
# Setup
#
#######################################

hp = HParams()
hp, run_name, parser = create_argparse_and_update_hp(hp)

parser.add_argument('--dataset', default='yelp',
                    help='yelp,amazon; will determine which subwordenc to use')
parser.add_argument('--init', default='The meaning of life is ',
                    help="Initial text ")
parser.add_argument('--load_model', default=None,
                    help="Path to model to load")
parser.add_argument('--seq_len', type=int, default=50,
                    help="Maximum sequence length")
parser.add_argument('--softmax_method', type=str, default='softmax',
                    help="softmax or gumbel")
parser.add_argument('--sample_method', type=str, default='sample',
                    help="sample or greedy")
parser.add_argument('--gumbel_hard', type=bool, default=False,
                    help="whether to produce one-hot from Gumbel softmax")

parser.add_argument('--beam_size', type=int, default=1,
                    help="Width for beam search")
parser.add_argument('--len_norm_factor', type=float, default=0.0,
                    help="Normalization factor")
parser.add_argument('--len_norm_const', type=float, default=5.0,
                    help="Normalization constant")

parser.add_argument('--gpus', default='0',
                    help="CUDA visible devices, e.g. 2,3")

opt = parser.parse_args()

setup_gpus(opt.gpus, hp.seed)

ds_conf = DatasetConfig(opt.dataset)
if opt.load_model is None:
    opt.load_model = ds_conf.lm_path

#######################################
#
# Run
#
#######################################


def batchify(data, batch_size):
    """
    Args:
        data: 1D Tensor
        batch_size: int
    Returns:
        data: reshaped Tensor of size (batch_size, -1)
        Example where data is non-negative integers and batch_size = 4
        [[0  1  2  3  4  5  6 ]
         [7  8  9  10 11 12 13]
         [14 15 16 17 18 19 20]
         [21 22 23 24 25 26 27]]
    Note: not currently using this anymore. Was used when reading in data from text fileW
    """
    nbatch = data.size(0) // batch_size
    data = data.narrow(0, 0, nbatch * batch_size)  # same as slice
    data = data.view(batch_size, -1).contiguous()
    return data


#
# Prepare initial input text
#
subwordenc = load_file(ds_conf.subwordenc_path)
init_texts = [init for init in opt.init.split('|')]
init_tokens = [subwordenc.encode(init) for init in init_texts]
init_lens = [len(init) for init in init_tokens]
max_len = max(init_lens)
init_tokens_padded = [tokens + [PAD_ID for _ in range(max_len - len(tokens))] for tokens in init_tokens]
init_tensor = [batchify(torch.LongTensor(init), 1) for init in init_tokens_padded]
init_tensor = torch.cat(init_tensor, dim=0)  # [batch, lens
init_tensor = move_to_cuda(init_tensor)
batch_size = init_tensor.size(0)

#
# Load and set up model
#
checkpoint = torch.load(opt.load_model)
model = checkpoint['model']
if isinstance(model, nn.DataParallel):
    model = model.module

ngpus = 1 if len(opt.gpus) == 1 else len(opt.gpus.split(','))

#
# Generate
# #
if 'mlstm' in opt.load_model:
    # Set up encoder decoder
    embed, rnn = model.embed, model.rnn
    enc_dec = StackedLSTMEncoderDecoder(embed, rnn)
    if torch.cuda.is_available():
        enc_dec.cuda()
    enc_dec = DataParallelModel(enc_dec) if ngpus > 1 else enc_dec
    enc_dec.eval()

    # Generate
    result = enc_dec(init_tensor,
                     dec_kwargs={'seq_len': opt.seq_len,
                                 'softmax_method': opt.softmax_method,
                                 'sample_method': opt.sample_method,
                                 'tau': hp.tau,
                                 'gumbel_hard': opt.gumbel_hard,
                                 'k': opt.beam_size,
                                 'subwordenc': subwordenc})
    probs, ids, texts, extra = zip(*result) if ngpus > 1 else result
    if ngpus > 1:  # flatten: each gpu returns lists of texts
        texts = [batch_text for gpu_texts in texts for batch_text in gpu_texts]

    for i in range(batch_size):
        print(init_texts[i] + texts[i])
        print('-' * 100)
