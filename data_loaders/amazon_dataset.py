# amazon_dataset.py

"""
Data preparation and loaders for Amazon dataset. Here, an item is one "product" being sold
"""

from collections import defaultdict
import json
import math
import numpy as np
import os
import pdb

from torch.utils.data import Dataset, DataLoader

from data_loaders.summ_dataset import SummReviewDataset, SummDataset
from project_settings import HParams, DatasetConfig
from utils import load_file, save_file


class AmazonPytorchDataset(Dataset):
    """
    Implements Pytorch Dataset

    One data point for model is n_docs reviews for one item. When training, we want to have batch_size items and
    sample n_docs reviews for each item. If a item has less than n_docs reviews, we sample with replacement
    (sampling with replacement as then you'll be summarizing repeated reviews, but this shouldn't happen right now
    as only items with a minimum number of reviews is used (50). These items and theiR reviews are selected
    in AmazonDataset.save_processed_splits().
    """
    def __init__(self, split=None, n_docs=None,
                 subset=None,
                 seed=0,
                 sample_reviews=True,
                 category=None,
                 item_max_reviews=None):
        """
        Args:
            split: str ('train', val', 'test')
            n_docs: int
            subset: float (Value in [0.0, 1.0]. If given, then dataset is truncated to subset of the businesses
            seed: int (set seed because we will be using np.random.choice to sample reviews if sample_reviews=True)
            sample_reviews: boolean
                - When True, __getitem_ will sample n_docs reviews for each item. The number of times a item appears
                in the dataset is dependent on uniform_items.
                - When False, each item will appear math.floor(number of reviews item has / n_docs) times
                so that almost every review is seen (with up to n_docs - 1 reviews not seen).
                    - Setting False is useful for (a) validation / test, and (b) simply iterating over all the reviews
                    (e.g. to build the vocabulary).
            item_max_reviews: int (maximum number of reviews a item can have)
                - This is used to remove outliers from the data. This is especially important if uniform_items=False,
                as there may be a large number of reviews in a training epoch coming from a single item. This also
                still matters when uniform_items=True, as items an outlier number of reviews will have reviews
                that are never sampled.
                - For the Amazon dataset, there are 11,870 items in the training set with at least 50 reviews
                no longer than 150 subtokens. The breakdown of the distribution in the training set is:
                    Percentile  |  percentile_n_reviews  |  n_items  |  total_revs
                    TODO?
        """
        self.split = split
        self.n_docs = n_docs
        self.subset = subset
        self.sample_reviews = sample_reviews
        item_max_reviews = float('inf') if item_max_reviews is None else item_max_reviews
        self.item_max_reviews = item_max_reviews

        self.ds_conf = DatasetConfig('amazon')

        # Set random seed so that choice is always the same across experiments
        # Especially necessary for test set (along with shuffle=False in the DataLoader)
        np.random.seed(seed)

        # Create map from idx-th data point to item
        item_to_nreviews = load_file(
            os.path.join(self.ds_conf.processed_path, '{}/item-to-nreviews.json'.format(split)))
        self.idx_to_item = {}

        # Filter to only one category
        if category:
            print('Filtering to only reviews in: {}'.format(category))
            _, item_to_reviews = AmazonDataset.load_all_reviews()
            for item in item_to_reviews:
                cat = item_to_reviews[item][0]['category']
                if cat != category:
                    if item in item_to_nreviews:  # item_to_nreviews is only for items and reviews in processed splits
                        del item_to_nreviews[item]

        if sample_reviews:
            # Get the number of times each item will appear in a pass through this dataset
            item_min_reviews = min(item_to_nreviews.values())
            if item_max_reviews == float('inf'):
                n_per_item = math.ceil(item_min_reviews / n_docs)
            else:
                n_per_item = np.mean([n for n in item_to_nreviews.values() if n <= item_max_reviews])
                n_per_item = math.ceil(n_per_item / n_docs)
            # print('Each item will appear {} times'.format(n_per_item))

            idx = 0
            for item, n_reviews in item_to_nreviews.items():
                if n_reviews <= item_max_reviews:
                    for _ in range(n_per_item):
                        self.idx_to_item[idx] = item
                        idx += 1
        else:
            # __getitem__ will not sample
            idx = 0
            self.idx_to_item_startidx = {}
            # idx items idx of one dataset item. item_startidx is the idx within that item's reviews.
            tot = 0
            for item, n_reviews in item_to_nreviews.items():
                if n_reviews <= item_max_reviews:
                    tot += n_reviews
                    item_startidx = 0
                    for _ in range(math.floor(n_reviews / n_docs)):
                        self.idx_to_item[idx] = item
                        self.idx_to_item_startidx[idx] = item_startidx
                        idx += 1
                        item_startidx += n_docs

        if self.subset:
            end = int(self.subset * len(self.idx_to_item))
            for idx in range(end, len(self.idx_to_item)):
                del self.idx_to_item[idx]

        self.n = len(self.idx_to_item)

    def __getitem__(self, idx):
        # Map idx to item and load reviews
        item = self.idx_to_item[idx]
        fp = os.path.join(self.ds_conf.processed_path, '{}/{}_reviews.json'.format(self.split, item))
        reviews = load_file(fp)

        # Get reviews from item
        if self.sample_reviews:
            if len(reviews) < self.n_docs:
                reviews = np.random.choice(reviews, size=self.n_docs, replace=True)
            else:
                reviews = np.random.choice(reviews, size=self.n_docs, replace=False)
        else:
            start_idx = self.idx_to_item_startidx[idx]
            reviews = reviews[start_idx:start_idx + self.n_docs]

        # Collect data to be returned
        texts, ratings = zip(*[(s['reviewText'], s['overall']) for s in reviews])
        texts = SummDataset.concat_docs(texts, edok_token=True)
        avg_rating = int(np.round(np.mean(ratings)))
        metadata = {'item': item, 'category': reviews[0]['category']}
        # all the reviews are for the same item, each review will have same category so use 0-th

        return texts, avg_rating, metadata

    def __len__(self):
        return self.n


class AmazonDataset(SummReviewDataset):
    def __init__(self):
        super(AmazonDataset, self).__init__()
        self.name = 'amazon'
        self.conf = DatasetConfig('amazon')
        self.n_ratings_labels = 5
        self.reviews = None
        self.subwordenc = load_file(self.conf.subwordenc_path)

    @staticmethod
    def load_all_reviews():
        """
        Returns:
            reviews: list of dicts
            item_to_reviews: dict, key=str (item id), value=list of dicts
        """
        reviews = []
        item_to_reviews = defaultdict(list)
        amazon_dir = DatasetConfig('amazon').dir_path
        for fn in os.listdir(amazon_dir):
            if fn.endswith('.json'):
                cat = fn.split('.json')[0].replace('_5', '')
                # print(cat)
                for line in open(os.path.join(amazon_dir, fn), 'r').readlines():
                    rev = json.loads(line)
                    rev['category'] = cat
                    reviews.append(rev)
                    item_to_reviews[rev['asin']].append(rev)

        return reviews, item_to_reviews

    def get_data_loader(self, split='train', n_docs=8, subset=None, sample_reviews=True,
                        category=None,
                        batch_size=64, shuffle=True, num_workers=4):
        """
        Return iterator over specific split in dataset
        """
        ds = AmazonPytorchDataset(split=split, n_docs=n_docs, subset=subset, sample_reviews=sample_reviews,
                                  category=category,
                                  item_max_reviews=self.conf.item_max_reviews)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        return loader

    ####################################
    #
    # One off functions
    #
    ####################################
    def save_processed_splits(self):
        """
        Save train, val, and test splits. Splits are across items (e.g. a item is either in train, val, or test).
        Iterates over all reviews in the original dataset. Tries to get close to a 80-10-10 split.

        Args:
            review_max_len: int (maximum length in subtokens a review can be)
            item_min_reviews: int (min number of reviews a item must have)
            out_dir: str (path to save splits to, e.g. datasets/amazon_dataset/proccessed/)
        """
        review_max_len = self.conf.review_max_len
        item_min_reviews = self.conf.item_min_reviews

        print('Saving processed splits')
        if self.reviews is None:
            self.reviews, _ = AmazonDataset.load_all_reviews()

        # # Note: we actually do more filtering in the Pytorch dataset class
        print('Filtering reviews longer than: {}'.format(review_max_len))
        item_to_reviews = defaultdict(list)
        for r in self.reviews:
            if len(self.subwordenc.encode(r['reviewText'])) < review_max_len:
                item_to_reviews[r['asin']].append(r)

        # Calculate target amount of reviews per item
        n = sum([len(revs) for revs in item_to_reviews.values()])
        print('Total number of reviews before filtering: {}'.format(len(self.reviews)))
        print('Total number of reviews after filtering: {}'.format(n))

        # Note: we actually do more filtering in the Pytorch Dataset class
        print('Filtering items with less than {} reviews'.format(item_min_reviews))
        item_to_n = {}
        for item in list(item_to_reviews.keys()):  # have to do list and keys for python3 to delete in-place
            n = len(item_to_reviews[item])
            if n < self.conf.item_min_reviews:
                del item_to_reviews[item]
            else:
                item_to_n[item] = n
        n = sum(item_to_n.values())
        print('Total number of reviews after filtering: {}'.format(n))
        print('Total number of items after filtering: {}'.format(len(item_to_n)))

        # Construct splits
        n_tr, n_val, n_te = int(0.8 * n), int(0.1 * n), int(0.1 * n)
        cur_n_tr, cur_n_val, cur_n_te = 0, 0, 0
        split_to_item_to_nreviews = {'train': {}, 'val': {}, 'test': {}}
        # In descending order of number of reviews per item
        for i, (item, n) in enumerate(sorted(item_to_n.items(), key=lambda x: -x[1])):
            # once every ten items, save to val / test if we haven't yet hit the target number
            if (i % 10 == 8) and (cur_n_val < n_val):
                split = 'val'
                cur_n_val += n
            elif (i % 10 == 9) and (cur_n_te < n_te):
                split = 'test'
                cur_n_te += n
            else:
                split = 'train'
                cur_n_tr += n

            out_fp = os.path.join(self.conf.processed_path, '{}/{}_reviews.json'.format(split, item))
            save_file(item_to_reviews[item], out_fp, verbose=False)

            split_to_item_to_nreviews[split][item] = n

        print('Number of train reviews: {} / {}'.format(cur_n_tr, n_tr))
        print('Number of val reviews: {} / {}'.format(cur_n_val, n_val))
        print('Number of test reviews: {} / {}'.format(cur_n_te, n_te))

        # This file is used by AmazonPytorchDataset
        for split, item_to_nreviews in split_to_item_to_nreviews.items():
            out_fp = os.path.join(self.conf.processed_path, '{}/item-to-nreviews.json'.format(split))
            save_file(item_to_nreviews, out_fp)

    def print_filtered_data_stats(self):
        """
        Calculate and print some statistics on the filtered dataset. This is what we use for
        training, validation, and testing.
        """
        for split in ['train', 'val', 'test']:
            all_rev_lens = []
            rating_to_count = defaultdict(int)
            cat_to_count = defaultdict(int)
            dl = self.get_data_loader(split=split, n_docs=1, sample_reviews=False,
                                      batch_size=1, num_workers=0, shuffle=False)
            for texts, ratings, metadata in dl:
                for i, text in enumerate(texts):  # note: this just loops once right now as n_docs=1
                    rev_len = len(self.subwordenc.encode(text))
                    all_rev_lens.append(rev_len)
                    rating_to_count[ratings[i].item()] += 1
                    cat_to_count[metadata['category'][0]] += 1
                    # data loader maps metadata dict into key: list of values

            print('', '=' * 50, '')
            print('Split: {}'.format(split))
            print('Total number of reviews: {}'.format(len(all_rev_lens)))
            print('Number of reviews per star rating:')
            for rating, count in sorted(rating_to_count.items()):
                print('-- {} stars: {:.2f} reviews; {} of dataset'
                      .format(rating, count, float(count) / len(all_rev_lens)))

            print('Number of reviews per category:')
            for cat, count in sorted(cat_to_count.items()):
                print('-- {}: {:.2f} reviews; {} of dataset'
                      .format(cat, count, float(count) / len(all_rev_lens)))

            print('Length of review:')
            print('-- mean: {}'.format(np.mean(all_rev_lens)))
            print('-- 75th percentile: {}'.format(np.percentile(all_rev_lens, 75)))
            print('-- 90th percentile: {}'.format(np.percentile(all_rev_lens, 90)))

    def print_original_data_stats(self):
        """
        Calculate and print some statistics on the original dataset
        """
        if self.reviews is None:
            self.reviews, self.item_to_reviews = AmazonDataset.load_all_reviews()
        lens = []
        for rev in self.reviews:
            lens.append(len(self.subwordenc.encode(rev['reviewText'])))

        print(np.median(lens))
        print(np.percentile(lens, 75))
        print(np.percentile(lens, 90))
        pdb.set_trace()


if __name__ == '__main__':
    from data_loaders.summ_dataset_factory import SummDatasetFactory

    hp = HParams()
    ds = SummDatasetFactory.get('amazon')
    # ds.save_processed_splits()
    ds.print_original_data_stats()
    # ds.print_filtered_data_stats()

    # test_dl = ds.get_data_loader(split='test', n_docs=8, sample_reviews=True,
    #                              category='Electronics',
    #                              batch_size=4, shuffle=True)
    # for texts, ratings, metadata in test_dl:
    #     x, lengths, labels = ds.prepare_batch(texts, ratings)
    #     pdb.set_trace()
