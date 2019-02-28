# project_settings.py

import os

SAVED_MODELS_DIR = 'checkpoints/'

OUTPUTS_DIR = 'outputs/'
OUTPUTS_EVAL_DIR = os.path.join(OUTPUTS_DIR, 'eval/')

WORD2VEC_PATH = 'datasets/GoogleNews-vectors-negative300.bin'  # for extractive method

PAD_TOK, EOS_TOK, GO_TOK, OOV_TOK, SDOC_TOK, EDOC_TOK = \
    '<pad>', '<EOS>', '<GO>', '<OOV>', '<DOC>', '</DOC>'
PAD_ID, EOS_ID, GO_ID, OOV_ID, SDOC_ID, EDOC_ID = \
    0, 1, 2, 3, 4, 5
RESERVED_TOKENS = [PAD_TOK, EOS_TOK, GO_TOK, OOV_TOK, SDOC_TOK, EDOC_TOK]


class DatasetConfig(object):
    def __init__(self, name):
        self.name = name

        if name == 'yelp':
            self.review_max_len = 150
            self.extractive_max_len = 38  # 99.5th percentile of reviews
            self.item_min_reviews = 50
            self.item_max_reviews = 260  # 90th percentile
            self.vocab_size = 32000  # target vocab size when building subwordenc

            # Paths
            self.dir_path = 'datasets/yelp_dataset/'
            self.reviews_path = 'datasets/yelp_dataset/review.json'
            self.businesses_path = 'datasets/yelp_dataset/business.json'
            self.processed_path = 'datasets/yelp_dataset/processed/'
            self.subwordenc_path = 'datasets/yelp_dataset/processed/subwordenc_32000_maxrevs260_fixed.pkl'

            # Trained models
            self.lm_path = 'stable_checkpoints/lm/mlstm/yelp/batch_size_512-lm_lr_0.001-notes_data260_fixed/' \
                           'lm_e24_2.88.pt'
            self.clf_path = 'stable_checkpoints/clf/cnn/yelp/batch_size_256-notes_data260_fixed/' \
                            'clf_e10_l0.6760_a0.7092.pt'
            self.sum_path = 'stable_checkpoints/sum/mlstm/yelp/' \
                            'batch_size_16-notes_cycloss_honly-sum_lr_0.0005-tau_2.0/' \
                            'sum_e0_tot3.32_r1f0.27.pt'
            self.autoenc_path = 'stable_checkpoints/sum/mlstm/yelp/' \
                    'autoenc_only_True-batch_size_16-sum_cycle_False-sum_lr_0.0005-tau_2.0/sum_e22_tot2.16_r1f0.03.pt'

        elif name == 'amazon':
            # Params
            self.review_max_len = 150
            self.extractive_max_len = 38  # 99.5th percentile of reviews
            self.item_min_reviews = 50
            self.item_max_reviews = 260  # 90th percentile
            self.vocab_size = 32000  # target vocab size when building subwordenc

            # Paths
            self.dir_path = 'datasets/amazon_dataset/'
            self.processed_path = 'datasets/amazon_dataset/processed/'
            self.subwordenc_path = 'datasets/amazon_dataset/processed/subwordenc_32000_secondpass.pkl'

            # Trained models
            self.lm_path = 'stable_checkpoints/lm/mlstm/amazon/batch_size_256-lm_lr_0.001/lm_e25_3.08.pt'
            self.clf_path = 'stable_checkpoints/clf/cnn/amazon/batch_size_256-clf_lr_0.0001/' \
                            'clf_e15_l0.7415_a0.7115_d0.0000.pt'
            self.sum_path = 'stable_checkpoints/sum/mlstm/amazon/batch_size_16-notes_both-sum_lr_0.0005-tau_2.0/' \
                            'sum_e1_tot4.14_r1f0.26.pt'
            self.autoenc_path = None


class HParams(object):
    def __init__(self):
        ###############################################
        #
        # MODEL GENERAL
        #
        ###############################################
        self.model_type = 'mlstm'  # mlstm, transformer
        self.emb_size = 256
        self.hidden_size = 512

        # transformer
        self.tsfr_blocks = 6
        self.tsfr_ff_size = 2048
        self.tsfr_nheads = 8
        self.tsfr_dropout = 0.1
        self.tsfr_tie_embs = False
        self.tsfr_label_smooth = 0.1  # range from [0.0, 1.0]; -1 means use regular CrossEntropyLoss

        # (m)lstm
        self.lstm_layers = 1
        self.lstm_dropout = 0.1
        self.lstm_ln = True  # layer normalization

        # TextCNN
        self.cnn_filter_sizes = [3, 4, 5]
        self.cnn_n_feat_maps = 128
        self.cnn_dropout = 0.5

        #
        # Decoding (sampling words)
        #
        self.tau = 2.0  # temperature for softmax
        self.g_eps = 1e-10  # Gumbel softmax

        ###############################################
        # SUMMARIZATION MODEL SPECIFIC
        ###############################################
        self.sum_cycle = True  # use cycle loss
        self.cycle_loss = 'enc'  # When 'rec', reconstruct original texts. When 'enc', compare rev_enc and sum_enc embs
        self.early_cycle = False  # When True, compute CosSim b/n mean and individual representations
        self.extract_loss = False  # use loss comparing summary to extractive summary
        self.autoenc_docs = True  # add autoencoding loss
        self.autoenc_only = False  # only perform autoencoding of reviews (would be used to pretrain)
        self.autoenc_docs_tie_dec = True  # use same decoder for summaries and review autoencoder
        self.tie_enc = True  # use same encoder for encoding documents and encoding summary
        self.sum_label_smooth = False  # for autoenc_loss and reconstruction cycle_loss
        self.sum_label_smooth_val = 0.1
        self.load_ae_freeze = False  # load pretrained autoencoder and freeze
        self.cos_wgt = 1.0  # weight for cycle loss and early cycle loss
        self.cos_honly = True  # compute cosine similarity losses using hiddens only, not hiddens + cells

        self.track_ppl = True  # use a fixed (pretraind) language model to calculate NLL of summaries

        # Discriminator
        self.sum_discrim = False  # add Discriminator loss
        self.wgan_lam = 10.0
        self.discrim_lr = 0.0001
        self.discrim_clip = 5.0
        self.discrim_model = 'cnn'
        self.discrim_onehot = True

        self.sum_clf = True  # calculate classification loss and accuracy
        self.sum_clf_lr = 0.0  # when 0, don't backwards() etc

        self.sum_lr = 0.0001
        self.sum_clip = 5.0  # clip gradients
        self.train_subset = 1.0  # train on this ratio of the training set (speed up experimentation, try to overfit)
        self.freeze_embed = True  # don't further train embedding layers

        self.concat_docs = False  # for one item, concatenate docs into long doc; else encode reviews separately
        self.combine_encs = 'mean'  # Combining separately encoded reviews: 'ff' for feedforward, 'mean' for mean, 'gru'
        self.combine_tie_hc = True  # Use the same FF / GRU to combine the hidden states and cell states
        self.combine_encs_gru_bi = True  # bidirectional gru to combine reviews
        self.combine_encs_gru_nlayers = 1
        self.combine_encs_gru_dropout = 0.1

        self.decay_tau = False
        self.decay_interval_size = 1000
        self.decay_tau_alpha = 0.1
        self.decay_tau_method = 'minus'
        self.min_tau = 0.4

        self.docs_attn = False
        self.docs_attn_hidden_size = 32
        self.docs_attn_learn_alpha = True

        ###############################################
        # LANGUAGE MODEL SPECIFIC
        ###############################################
        self.lm_lr = 0.0005
        self.lm_seq_len = 256

        # language model and mlstm (transformer has its own schedule)
        self.lm_clip = 5.0  # clip gradients
        # decay at end of epoch
        self.lm_lr_decay = 1.0  # 1 = no decay for 'times'
        self.lm_lr_decay_method = 'times'  # 'times', 'minus'

        ###############################################
        # CLASSIFIER SPECIFIC
        ###############################################
        self.clf_lr = 0.0001
        self.clf_clip = 5.0
        self.clf_onehot = True
        self.clf_mse = False  # treat as regression problem and use MSE instead of cross entropy

        ###############################################
        # TRAINING AND DATA REPRESENTATION
        ###############################################
        self.seed = 1234
        self.batch_size = 128
        self.n_docs = 8
        self.n_docs_min = -1
        self.n_docs_max = -1
        self.max_nepochs = 50
        self.notes = ''  # notes about run

        self.optim = 'normal'  # normal or noam
        self.noam_warmup = 4000  # number of warmup steps to linearly increase learning rate before decaying it

        #
        # UTILS / MISCELLANEOUS
        #
        self.debug = False

        ###############################################
        # EVALUATION
        ###############################################
        self.use_stemmer = True  # when calculating rouge
        self.remove_stopwords = False  # when calculating rouge
