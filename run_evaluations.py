# run_evaluations.py

"""
Run various evaluations

Usage:
1. python run_evaluations.py --summ_baselines=ledes-1,ledes-2,\
worst_review,best_review --tau=2.0 --gpus=0 --n_docs=8 --batch_size=16
2. python run_evaluations.py --summ_baselines=lm_autoenc,extractive --tau=2.0 \
--gpus=1 --n_docs=8 --batch_size=4
3. python run_evaluations.py --clf_baseline --tau=2.0 \
--gpus=2 --n_docs=8 --batch_size=4


"""
import copy
import os
import pdb
import re
from collections import defaultdict

import torch
import torch.nn as nn
from pprint import pprint as pp
import sys

import nltk
import numpy as np
import torch.nn.functional as F

from data_loaders.summ_dataset import SummDataset
from data_loaders.summ_dataset_factory import SummDatasetFactory
from models.custom_parallel import DataParallelModel
from models.mlstm import StackedLSTMDecoder
from models.nn_utils import classify_summ_batch, setup_gpus, calc_lm_nll
from models.summarization import SummarizationModel
from pretrain_classifier import TextClassifier
from train_sum import Summarizer

from evaluation.eval_utils import EvalMetrics

# sys.path.append('external/text_summarizer')
# from external.text_summarizer.centroid_w2v import CentroidW2VSummarizer
from project_settings import HParams, WORD2VEC_PATH, OUTPUTS_EVAL_DIR, DatasetConfig
from utils import save_file, create_argparse_and_update_hp, update_moving_avg, load_file


class Evaluations(object):
    def __init__(self, hp, opt):
        self.hp = hp
        self.opt = opt

        self.dataset = SummDatasetFactory.get(opt.dataset)

    def get_test_set_data_iter(self, batch_size=1):
        dl = self.dataset.get_data_loader(split='test', sample_reviews=False, n_docs=self.hp.n_docs,
                                          category=self.opt.az_cat,
                                          batch_size=batch_size, shuffle=False)
        return dl

    def run_summarization_baseline(self, method):
        """
        Args:
            method: str ('extractive', 'ledes-<n>', 'best_review', 'lm_autoenc')

        Saves outputs to: outputs/eval/<dataset>/<n_docs>/<method>
        """
        batch_size = self.hp.batch_size if method == 'lm_autoenc' else 1
        dl = self.get_test_set_data_iter(batch_size=batch_size)

        if torch.cuda.is_available():
            clf_model = torch.load(self.opt.load_clf)['model']
        else:
            raise Exception('You should run on a cuda machine to load and use the classifcation model')

        print('\n', '=' * 50)
        print('Running {} baseline'.format(method))
        if method == 'extractive':
            evaluator, summaries, acc, per_rating_acc = self.extractive_baseline(dl, clf_model)
        elif 'ledes' in method:  # e.g. ledes-2
            n = int(method.split('-')[1])
            evaluator, summaries, acc, per_rating_acc = self.ledes_baseline(dl, n, clf_model)
        elif method == 'best_review':
            evaluator, summaries, acc, per_rating_acc = self.best_or_worst_review_baseline(dl, 'best', clf_model)
        elif method == 'worst_review':
            evaluator, summaries, acc, per_rating_acc = self.best_or_worst_review_baseline(dl, 'worst', clf_model)
        elif method == 'lm_autoenc':
            evaluator, summaries, acc, per_rating_acc = self.lm_autoenc_baseline(dl, clf_model)

        # Calculate NLL of summaries using fixed, pretrained LM
        pretrained_lm = torch.load(self.opt.load_lm)['model']  # StackedLSTMEncoder
        pretrained_lm = pretrained_lm.module if isinstance(pretrained_lm, nn.DataParallel) else pretrained_lm
        avg_nll = 0.0
        loop_idx = 0
        for i in range(0, len(summaries), batch_size):
            batch_summs = summaries[i: i+ batch_size]
            batch_texts = [d['summary'] for d in batch_summs]
            dummy_ratings = [torch.LongTensor([0]) for _ in range(len(batch_texts))]
            try:
                batch_x, _, _ = self.dataset.prepare_batch(batch_texts, dummy_ratings)
                nll = calc_lm_nll(pretrained_lm, batch_x)
                if not np.isnan(nll.detach().cpu().numpy()):
                    avg_nll = update_moving_avg(avg_nll, nll.item(), loop_idx + 1)
                    loop_idx += 1
                else:
                    # lm_autoenc baseline has a rare edge case where a nan is produced
                    continue
            except Exception as e:
            # worst_review in the Amazon dataset has a rare edge case
            # where the worst review is an empty string.
            # No reviews should be empty, but it appears to just be one or two reviews
                print(e)
                continue

        # Save summaries, stats, rouge scores, etc.
        dataset_dir = self.opt.dataset if self.opt.az_cat is None else 'amazon_{}'.format(self.opt.az_cat)
        out_dir = os.path.join(OUTPUTS_EVAL_DIR, dataset_dir, 'n_docs_{}'.format(self.hp.n_docs), method)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        summs_out_fp = os.path.join(out_dir, 'summaries.json')
        save_file(summaries, summs_out_fp)
        out_fp = os.path.join(out_dir, 'stats.json')
        save_file({'acc': acc, 'per_rating_acc': per_rating_acc, 'nll': avg_nll}, out_fp)

        print('-' * 50)
        print('Rating accuracy: ', acc)
        print('NLL: ', avg_nll)
        print('Per rating accuracy: ', dict(per_rating_acc))
        for stat, rouge_dict in evaluator.get_avg_stats_dicts().items():
            print('-' * 50)
            print(stat.upper())
            print(evaluator.to_str(rouge_dict))

            out_fp = os.path.join(out_dir, 'avg_{}-rouges.json'.format(stat))
            save_file(rouge_dict, out_fp)
            out_fp = os.path.join(out_dir, 'avg_{}-rouges.csv'.format(stat))
            evaluator.to_csv(rouge_dict, out_fp)

        out_fp = os.path.join(out_dir, '{}-rouges.pdf')
        evaluator.plot_rouge_distributions(show=self.opt.show_figs, out_fp=out_fp)

    def extractive_baseline(self, data_iter, clf_model=None):
        """
        Run an extractive method
        """
        evaluator = EvalMetrics(remove_stopwords=self.hp.remove_stopwords,
                                use_stemmer=self.hp.use_stemmer,
                                store_all=True)
        summarizer = CentroidW2VSummarizer(WORD2VEC_PATH, length_limit=2,
                                           topic_threshold=0.3, sim_threshold=0.95,
                                           reordering=True, subtract_centroid=False, keep_first=False,
                                           bow_param=0, length_param=0, position_param=0,
                                           debug=False)

        summaries = []
        accuracy = 0.0
        per_rating_counts = defaultdict(int)
        per_rating_acc = defaultdict(int)
        for i, (texts, ratings, metadata) in enumerate(data_iter):
            for j, text in enumerate(texts):
                # texts is a list of of length batch_size
                # each item in texts is a str, i.e. n_docs documents concatenated together
                src_docs = SummDataset.split_docs(text)
                # limit is number of words
                # concatenate documents without the token
                summary = summarizer.summarize(SummDataset.concat_docs(src_docs, edok_token=False),
                                               limit=self.dataset.conf.extractive_max_len)
                evaluator.batch_update_avg_rouge([summary], [src_docs])
                acc, per_rating_counts, per_rating_acc, pred_ratings, pred_probs = \
                    classify_summ_batch(clf_model, [summary], [ratings[j]], self.dataset,
                                        per_rating_counts, per_rating_acc)

                if acc is None:
                    print('Summary was too short to classify')
                    pred_rating, pred_prob = None, None
                else:
                    pred_rating, pred_prob = pred_ratings[j].item(), pred_probs[j].item()
                    accuracy = update_moving_avg(accuracy, acc, i * len(texts) + j + 1)

                dic = {'docs': text, 'summary': summary, 'rating': ratings[j].item(),
                       'pred_rating': pred_rating, 'pred_prob': pred_prob}
                for k, values in metadata.items():
                    dic[k] = values[j]
                summaries.append(dic)

        return evaluator, summaries, accuracy.item(), per_rating_acc

    def ledes_baseline(self, data_iter, n=1, clf_model=None):
        """
        Add up until the first n sentences from each review, or until the maximum review length is exceeded
        """
        evaluator = EvalMetrics(remove_stopwords=self.hp.remove_stopwords,
                                use_stemmer=self.hp.use_stemmer,
                                store_all=True)
        summaries = []
        accuracy = 0.0
        per_rating_counts = defaultdict(int)
        per_rating_acc = defaultdict(int)
        for i, (texts, ratings, metadata) in enumerate(data_iter):
            # texts is a list of of length batch_size
            # each item in texts is a str, i.e. n_docs documents concatenated together
            for j, text in enumerate(texts):
                src_docs = SummDataset.split_docs(text)

                summary = []
                doc_sents = [nltk.sent_tokenize(doc) for doc in src_docs]
                summary_len = 0
                doc_idx, sent_idx = 0, 0

                # Keep adding sentences as long as summary isn't over maximum length and
                # there are still sentences to add
                while (summary_len < self.dataset.conf.review_max_len) and (sent_idx < n):
                    # Current document has this many sentences
                    if sent_idx < len(doc_sents[doc_idx]):
                        sent = doc_sents[doc_idx][sent_idx]
                        sent_tok_len = len(nltk.word_tokenize(sent))

                        # Adding sentence won't exceed maximum length
                        if summary_len + sent_tok_len <= self.dataset.conf.review_max_len:
                            summary.append(sent)
                            summary_len += sent_tok_len

                    # Move on to next document
                    doc_idx = (doc_idx + 1) % len(src_docs)
                    if doc_idx == 0:  # back to the first doc, all first sentences have been added
                        sent_idx += 1

                summary = ' '.join(summary)
                evaluator.batch_update_avg_rouge([summary], [src_docs])
                acc, per_rating_counts, per_rating_acc, pred_ratings, pred_probs = \
                    classify_summ_batch(clf_model, [summary], [ratings[j]], self.dataset,
                                        per_rating_counts, per_rating_acc)

                if acc is None:
                    print('Summary was too short to classify')
                    pred_rating, pred_prob = None, None
                else:
                    pred_rating, pred_prob = pred_ratings[j].item(), pred_probs[j].item()
                    accuracy = update_moving_avg(accuracy, acc, i * len(texts) + j + 1)

                dic = {'docs': text, 'summary': summary, 'rating': ratings[j].item(),
                       'pred_rating': pred_rating, 'pred_prob': pred_prob}
                for k, values in metadata.items():
                    dic[k] = values[j]
                summaries.append(dic)

        return evaluator, summaries, accuracy.item(), per_rating_acc

    def best_or_worst_review_baseline(self, data_iter, method='best', clf_model=None):
        """
        When summarizing n_docs reviews, calculate the average ROUGE1-F for each review as if it was the summary.
        Choose the document with the best / worst score.

        Note: it'd be far more efficient to calculate best and worst at the same time as all the rouges
        are already calculated...
        """
        evaluator = EvalMetrics(remove_stopwords=self.hp.remove_stopwords,
                                use_stemmer=self.hp.use_stemmer,
                                store_all=True)
        summaries = []
        accuracy = 0.0
        per_rating_counts = defaultdict(int)
        per_rating_acc = defaultdict(int)
        for i, (texts, ratings, metadata) in enumerate(data_iter):
            # texts is a list of of length batch_size
            # each item in texts is a str, i.e. n_docs documents concatenated together
            for j, text in enumerate(texts):
                bw_evaluator = None
                bw_rouge1_f = 0.0 if method == 'best' else 1.0
                bw_doc = None

                # Set each document as the summary and find the best one
                src_docs = SummDataset.split_docs(text)
                for doc in src_docs:
                    cur_evaluator = EvalMetrics(remove_stopwords=self.hp.remove_stopwords,
                                                use_stemmer=self.hp.use_stemmer,
                                                store_all=True)
                    avg_rouges, _, _, _ = cur_evaluator.batch_update_avg_rouge([doc], [src_docs])
                    is_better_worse = (method == 'best' and (avg_rouges['rouge1']['f'] >= bw_rouge1_f)) or \
                                      (method == 'worst' and (avg_rouges['rouge1']['f'] <= bw_rouge1_f))
                    if is_better_worse:
                        bw_evaluator = cur_evaluator
                        bw_rouge1_f = avg_rouges['rouge1']['f']
                        bw_doc = doc

                evaluator.update_with_evaluator(bw_evaluator)

                try:
                    acc, per_rating_counts, per_rating_acc, pred_ratings, pred_probs = \
                        classify_summ_batch(clf_model, [bw_doc], [ratings[j]], self.dataset,
                                            per_rating_counts, per_rating_acc)
                except Exception as e:
                # worst_review in the Amazon dataset has a rare edge case
                # where the worst review is an empty string.
                # No reviews should be empty, but it appears to just be one or two reviews
                    pass

                if acc is None:
                    print('Summary was too short to classify')
                    pred_rating, pred_prob = None, None
                else:
                    pred_rating, pred_prob = pred_ratings[j].item(), pred_probs[j].item()
                    accuracy = update_moving_avg(accuracy, acc, i * len(texts) + j + 1)

                dic = {'docs': text, 'summary': bw_doc, 'rating': ratings[j].item(),
                       'pred_rating': pred_rating, 'pred_prob': pred_prob}
                for k, values in metadata.items():
                    dic[k] = values[j]
                summaries.append(dic)

        return evaluator, summaries, accuracy.item(), per_rating_acc

    def lm_autoenc_baseline(self, data_iter, clf_model=None):
        """
        Use the pretrained language model to initialize an encoder-decoder model. This is basically the
        unsupervised abstractive summarization model without training.
        """

        # Load encoder decoder by initializing with languag emodel
        docs_enc = torch.load(self.opt.load_lm)['model']  # StackedLSTMEncoder
        docs_enc = docs_enc.module if isinstance(docs_enc, nn.DataParallel) else docs_enc
        summ_dec = StackedLSTMDecoder(copy.deepcopy(docs_enc.embed), copy.deepcopy(docs_enc.rnn))

        # Create Summarizer so that we can use run_epoch()
        # Copy hp and opt as we're modifying some params. This way there won't be any unexpected errors
        # if it's used by another method
        hp = copy.deepcopy(self.hp)
        hp.sum_cycle = False
        hp.autoenc_docs = False
        hp.sum_clf = False
        opt = copy.deepcopy(self.opt)
        opt.print_every_nbatches = float('inf')

        summarizer = Summarizer(hp, opt, '/tmp/')
        summarizer.tb_val_sub_writer = None
        summarizer.tau = self.hp.tau
        summarizer.ngpus = 1 if len(self.opt.gpus) == 1 else len(self.opt.gpus.split(','))
        summarizer.sum_model = torch.load(self.opt.load_lm)
        summarizer.dataset = self.dataset

        summarizer.fixed_lm =  torch.load(self.opt.load_lm)['model']  # StackedLSTMEncoder
        summarizer.fixed_lm = summarizer.fixed_lm.module if isinstance(summarizer.fixed_lm, nn.DataParallel) \
            else summarizer.fixed_lm

        # Create SummarizationModel
        docs_autodec, combine_encs_h_net, combine_encs_c_net = None, None, None
        summ_enc, docs_dec, discrim_model, clf_model_arg, fixed_lm = None, None, None, None, None
        summarizer.sum_model = SummarizationModel(docs_enc, docs_autodec,
                                                  combine_encs_h_net, combine_encs_c_net, summ_dec,
                                                  summ_enc, docs_dec, discrim_model, clf_model_arg, fixed_lm,
                                                  hp, self.dataset)
        if torch.cuda.is_available():
            summarizer.sum_model.cuda()
        if summarizer.ngpus > 1:
            summarizer.sum_model = DataParallelModel(summarizer.sum_model)
        summarizer.sum_model.eval()
        with torch.no_grad():
            stats_avgs, evaluator, summaries = summarizer.run_epoch(
                data_iter, data_iter.__len__(), 0, 'test',
                store_all_rouges=True, store_all_summaries=True,
                save_intermediate=False, run_val_subset=False)

        #
        # Pass summaries through classifier
        #
        # Note: I know that since the SummarizationModel already calculates the classification accuracy
        # if sum_clf=True. Hence, technically, I could refactor it to add everything that I'd like to compute
        # in the forward pass and add to stats(). However, I think it's cleaner /easier to just do everything
        # I want here, especially if I add more things like per rating counts and accuracy. Plus,
        # it's just one pass through the test set -- which I'll run infrequently to evaluate a trained model.
        # I think that it takes more time is fine.
        #
        results = []
        accuracy = 0.0
        per_rating_counts = defaultdict(int)
        per_rating_acc = defaultdict(int)
        for i, (texts, ratings_batch, metadata) in enumerate(data_iter):
            summaries_batch = summaries[i * self.hp.batch_size: i * self.hp.batch_size + len(texts)]
            acc, per_rating_counts, per_rating_acc, pred_ratings, pred_probs = \
                classify_summ_batch(clf_model, summaries_batch, ratings_batch, self.dataset,
                                    per_rating_counts, per_rating_acc)

            if acc is None:
                print('Summary was too short to classify')
                pred_ratings = [None for _ in range(len(summaries_batch))]
                pred_probs = [None for _ in range(len(summaries_batch))]
            else:
                accuracy = update_moving_avg(accuracy, acc, i + 1)

            for j in range(len(summaries_batch)):
                dic = {'docs': texts[j],
                       'summary': summaries_batch[j],
                       'rating': ratings_batch[j].item(),
                       'pred_rating': pred_ratings[j].item(),
                       'pred_prob': pred_probs[j].item()}
                for k, values in metadata.items():
                    dic[k] = values[j]
                results.append(dic)

        return evaluator, results, accuracy.item(), per_rating_acc

    def run_clf_baseline(self):
        """
        Calculate the classification accuracy when the input is all the reviews concatenated together. This provdies
        a sort of ceiling on how well each of the summarization methods can do, as the classification model
        is not perfect either.
        """
        print('\n', '=' * 50)
        print('Running classifier baseline')

        # Load classifier
        clf_model = torch.load(self.opt.load_clf)['model']
        clf_model = clf_model.module if isinstance(clf_model, nn.DataParallel) else clf_model
        if torch.cuda.is_available():
            clf_model.cuda()
        if len(self.opt.gpus) > 1:
            clf_model = nn.DataParallel(clf_model)

        summaries = []
        accuracy = 0.0
        per_rating_counts = defaultdict(int)
        per_rating_acc = defaultdict(int)
        dl = self.get_test_set_data_iter(self.hp.batch_size)
        for i, (texts, ratings_batch, metadata) in enumerate(dl):
            summaries_batch = []
            for j, text in enumerate(texts):
                # texts is a list of of length batch_size
                # each item in texts is a str, i.e. n_docs documents concatenated together
                # concatenate documents without the token
                src_docs = SummDataset.split_docs(text)
                summary = SummDataset.concat_docs(src_docs, edok_token=False)
                summaries_batch.append(summary)

            acc, per_rating_counts, per_rating_acc, pred_ratings, pred_probs = \
                classify_summ_batch(clf_model, summaries_batch, ratings_batch, self.dataset,
                                    per_rating_counts, per_rating_acc)
            accuracy = update_moving_avg(accuracy, acc, i + 1)

            for j in range(len(summaries_batch)):
                dic = {'docs': summaries_batch[j],
                       'rating': ratings_batch[j].item(),
                       'pred_rating': pred_ratings[j].item(),
                       'pred_prob': pred_probs[j].item()}
                for k, values in metadata.items():
                    dic[k] = values[j]
                summaries.append(dic)


        # Calculate NLL of summaries using fixed, pretrained LM
        pretrained_lm = torch.load(self.opt.load_lm)['model']  # StackedLSTMEncoder
        pretrained_lm = pretrained_lm.module if isinstance(pretrained_lm, nn.DataParallel) else pretrained_lm
        avg_nll = 0.0
        batch_size = self.hp.batch_size
        for i in range(0, len(summaries), batch_size):
            batch_summs = summaries[i: i+ batch_size]
            batch_texts = [d['docs'] for d in batch_summs]
            dummy_ratings = [torch.LongTensor([0]) for _ in range(len(batch_texts))]
            batch_x, _, _ = self.dataset.prepare_batch(batch_texts, dummy_ratings)
            nll = calc_lm_nll(pretrained_lm, batch_x)
            avg_nll = update_moving_avg(avg_nll, nll.item(), i + 1)

        # Print and save accuracies, summaries, etc.
        print('NLL: ', avg_nll)
        print('Accuracy: ', accuracy.item())
        print('Per rating accuracy: ', per_rating_acc)

        dataset_dir = self.opt.dataset if self.opt.az_cat is None else 'amazon_{}'.format(self.opt.az_cat)
        out_dir = os.path.join(OUTPUTS_EVAL_DIR, dataset_dir, 'n_docs_{}'.format(self.hp.n_docs), 'clf_baseline')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_fp = os.path.join(out_dir, 'summaries.json')
        save_file(summaries, out_fp)
        out_fp = os.path.join(out_dir, 'stats.json')
        save_file({'acc': accuracy.item(), 'per_rating_acc': per_rating_acc, 'nll': avg_nll}, out_fp)

    def run_varying_ndocs_evaluation(self):
        """Vary n_docs at inference and calculate various stats"""
        pass

    def run_std_vs_acc(self):
        """Plot standard deviation of reviews being summarized against classification accuracy of summary"""
        pass

    def collect_results(self):
        """
        Aggregate results so that we can easily compare them
        """
        # Just hard coding the ones we're most interested in right now
        # Only including these in the output file just makes it easier to scroll through and read
        METHODS = ['unsup_abstractive', 'lm_autoenc', 'extractive']
        for n_docs in [8]:
            ndocs_dir = os.path.join(OUTPUTS_EVAL_DIR, self.opt.dataset, 'n_docs_{}'.format(n_docs))

            # Collect all summaries
            all_summs = []
            for method in METHODS:
                summs_fp = os.path.join(ndocs_dir, method, 'summaries.json')
                summs = load_file(summs_fp)
                all_summs.append(summs)
            # Combine them
            assert len(set([len(summs) for summs in all_summs])) == 1, \
                   'All methods should be calculated on the same test set and have the same number of summaries'

            # Combine summaries
            aggregated = []
            n_summs = len(all_summs[0])
            for i in range(n_summs):
                docs = [all_summs[j][i]['docs'] for j in range(len(METHODS))]
                assert len(set(docs)) == 1, 'The documents being summarized should be the same / in the same order'

                agg = {}
                # Get data related to original reviews
                for k, v in all_summs[0][i].items():
                    if k not in ['summary', 'pred_rating', 'pred_prob']:
                        agg[k] = v
                # Add summary data for each method
                for j in range(len(METHODS)):
                    agg[METHODS[j]] = {'summary': all_summs[j][i]['summary'],
                                       'pred_rating': all_summs[j][i]['pred_rating'],
                                       'pred_prob': all_summs[j][i]['pred_prob']}
                aggregated.append(agg)

            # Combine them
            out_fp = os.path.join(ndocs_dir, 'aggregated', 'summaries.json')
            save_file(aggregated, out_fp)


if __name__ == '__main__':
    hp = HParams()
    hp, run_name, parser = create_argparse_and_update_hp(hp)

    parser.add_argument('--dataset', default='yelp',
                        help='yelp,amazon')
    parser.add_argument('--az_cat', default=None,
                        help='"Movies_and_TV" or "Electronics"'
                             'Only test on one category')

    parser.add_argument('--summ_baselines', default='',
                        help='comma delimited strs: extractive,ledes-n,best_review,worst_review,lm_autoenc')
    parser.add_argument('--clf_baseline', action='store_true')

    parser.add_argument('--load_lm', default=None,
                        help='Path to pretrained language model;'
                             'Used for lm_autoenc baseline and calculating NLL of summaries')
    parser.add_argument('--load_clf', default=None)

    parser.add_argument('--show_figs', action='store_true')
    parser.add_argument('--collect_results', action='store_true')
    parser.add_argument('--gpus', default='0',
                        help="CUDA visible devices, e.g. 2,3")

    opt = parser.parse_args()
    setup_gpus(opt.gpus, hp.seed)


    # Set some default paths. It's dataset dependent, which is why we do it here, as dataset is also a
    # command line argument
    ds_conf = DatasetConfig(opt.dataset)
    if opt.load_lm is None:
        opt.load_lm = ds_conf.lm_path
    if opt.load_clf is None:
        opt.load_clf = ds_conf.clf_path

    # Run evaluations
    evaluations = Evaluations(hp, opt)
    if len(opt.summ_baselines) > 1:
        for method in opt.summ_baselines.split(','):
            evaluations.run_summarization_baseline(method)
    if opt.clf_baseline:
        evaluations.run_clf_baseline()

    if opt.collect_results:
        evaluations.collect_results()
