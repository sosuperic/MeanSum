# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Program to build a SubwordTextEncoder.

Example usage:

PYTHONPATH=. python data_loaders/build_subword_encoder.py \
--dataset=yelp \
--output_dir=./ \
--output_fn=subwordenc

PYTHONPATH=. python data_loaders/build_subword_encoder.py \
--corpus_filepattern=datasets/yelp_dataset/processed/reviews_texts_train.txt \
--output_dir=./ \
--output_fn=tmp_enc
"""

import os
import pdb

from data_loaders import text_encoder, tokenizer
from data_loaders.summ_dataset_factory import SummDatasetFactory
from project_settings import RESERVED_TOKENS, HParams
from utils import save_file, create_argparse_and_update_hp


def main(opt):
    if opt.dataset:
        dataset = SummDatasetFactory.get(opt.dataset)
        dl = dataset.get_data_loader(split='train', n_docs=1, sample_reviews=False,
                                     batch_size=1, num_workers=0, shuffle=False)
        print('Writing reviews to file')
        with open('/tmp/{}_data.txt'.format(opt.dataset), 'w') as f:
            for texts, ratings, metadata in dl:
                f.write('{}\n'.format(texts[0]))
        print('Creating token counts')
        token_counts = tokenizer.corpus_token_counts(
            '/tmp/{}_data.txt'.format(opt.dataset),
            opt.corpus_max_lines,
            split_on_newlines=True)
    elif opt.corpus_filepattern:
        token_counts = tokenizer.corpus_token_counts(
            opt.corpus_filepattern,
            opt.corpus_max_lines,
            split_on_newlines=True)
    else:
        raise ValueError('Must provide --dataset or provide --corpus_filepattern')

    print('Building to target size')
    encoder = text_encoder.SubwordTextEncoder.build_to_target_size(
        opt.target_size, token_counts, 0, 1e9,
        reserved_tokens=RESERVED_TOKENS)

    print('Saving tokenizer')
    vocab_fp = os.path.join(opt.output_dir, opt.output_fn + '.txt')  # stores vocab coutns
    encoder.store_to_file(vocab_fp)
    enc_fp = os.path.join(opt.output_dir, opt.output_fn + '.pkl')
    save_file(encoder, enc_fp, verbose=True)

    pdb.set_trace()


if __name__ == '__main__':
    hp = HParams()
    hp, run_name, parser = create_argparse_and_update_hp(hp)
    parser.add_argument('--output_dir', default='/tmp/')
    parser.add_argument('--output_fn', default='subwordenc')

    parser.add_argument('--corpus_filepattern', default=None, help='Corpus of one or more text files')
    parser.add_argument('--corpus_max_lines', default=float('inf'), help='How many lines of coprus to read')
    parser.add_argument('--dataset', default=None, help='yelp,amazon')

    parser.add_argument('--target_size', default=32000, help='Target size of vocab')
    parser.add_argument('--num_iterations', default=4, help='Number of iterations')
    opt = parser.parse_args()

    main(opt)
