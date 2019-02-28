# mlstm.py

"""
Multiplicative LSTM (used in 'Learning to Generate Reviews and Discovering Sentiment')
"""
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.nn_utils import move_to_cuda, logits_to_prob, prob_to_vocab_id
from project_settings import EOS_ID, PAD_ID


class mLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, layer_norm=False):
        super(mLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer_norm = layer_norm

        self.wx = nn.Linear(input_size, 4 * hidden_size, bias=False)
        self.wh = nn.Linear(hidden_size, 4 * hidden_size, bias=True)
        self.wmx = nn.Linear(input_size, hidden_size, bias=False)
        self.wmh = nn.Linear(hidden_size, hidden_size, bias=False)

        if layer_norm:
            self.wx_norm = nn.LayerNorm(input_size)
            self.wh_norm = nn.LayerNorm(hidden_size)
            self.wmx_norm = nn.LayerNorm(input_size)
            self.wmh_norm = nn.LayerNorm(hidden_size)

    def forward(self, data, hidden, cell):
        hx, cx = hidden, cell

        hx = self.wmh_norm(hx) if self.layer_norm else hx
        data_wm = self.wmx_norm(data) if self.layer_norm else data
        m = self.wmx(data_wm) * self.wmh(hx)

        m = self.wh_norm(m) if self.layer_norm else m
        data_w = self.wx_norm(data) if self.layer_norm else data
        gates = self.wx(data_w) + self.wh(m)

        i, f, o, u = gates.chunk(4, 1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        u = torch.tanh(u)
        o = torch.sigmoid(o)

        cy = f * cx + i * u
        hy = o * torch.tanh(cy)

        return hy, cy


class StackedLSTM(nn.Module):
    def __init__(self, cell, num_layers, input_size, hidden_size, output_size, dropout, layer_norm=False):
        super(StackedLSTM, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.layer_norm = layer_norm

        self.dropout = nn.Dropout(dropout)
        self.h2o = nn.Linear(hidden_size, output_size)
        if layer_norm:
            self.h2o_norm = nn.LayerNorm(hidden_size)

        self.layers = []
        for i in range(num_layers):
            layer = cell(input_size, hidden_size, layer_norm=layer_norm)
            self.add_module('layer_%d' % i, layer)
            self.layers += [layer]
            input_size = hidden_size

    def forward(self, input, hidden, cell):
        """
        One time step

        Args:
            input: [batch_size, input_size]
            hidden: [batch_size, n_layers, hidden]
            cell: [batch_size, n_layers, hidden]

        Returns:
            hidden: [batch_size, n_layers, hidden]
            cell: [batch_size, n_layers, hidden]
            output: [batch_size, output_size]
        """
        h_0, c_0 = hidden, cell
        h_1, c_1 = [], []
        for i in range(self.num_layers):
            layer = getattr(self, 'layer_{}'.format(i))
            h_1_i, c_1_i = layer(input, h_0[:, i, :], c_0[:, i, :])
            if i == 0:
                input = h_1_i
            else:
                input = input + h_1_i
            if i != len(self.layers):
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1, dim=1)  # [batch, n_layers, hidden]
        c_1 = torch.stack(c_1, dim=1)  # [batch, n_layers, hidden]

        if self.layer_norm:
            input = self.h2o_norm(input)
        output = self.h2o(input)

        return h_1, c_1, output

    def state0(self, batch_size):
        h_0 = torch.zeros(batch_size, self.num_layers, self.hidden_size, requires_grad=False)
        c_0 = torch.zeros(batch_size, self.num_layers, self.hidden_size, requires_grad=False)
        return h_0, c_0


class StackedLSTMEncoder(nn.Module):
    def __init__(self, embed, rnn):
        super(StackedLSTMEncoder, self).__init__()
        self.embed = embed
        self.rnn = rnn

    def forward(self, input, hidden, cell,
                all_hiddens=False, all_cells=False, all_outputs=True):
        """
        Embed input and forward through rnn seq_len time steps

        Args:
            input: [batch_size, seq_len] or [batch_size, seq_len, vocab_size]
            hidden: [batch_size, n_layers, hidden_size]
            cell: [batch_size, n_layers, hidden_size]

            all_hiddens: boolean (return hidden state at every time step, otherwise return last hidden)
            all_cells: boolean (return cell state at every time step, otherwise return last cell)
            all_outputs: boolean (return output at every time step, otherwise return last output)

        Returns:
            hiddens: length seq_len list of hidden states
            cells:  length seq_len list of cell states
            outputs: length seq_len list of [batch, output_size] tensors

        """
        hiddens, cells, outputs = [], [], []
        seq_len = input.size(1)
        for t in range(seq_len):
            if input.dim() == 2:
                emb = self.embed(input[:, t])
            elif input.dim() == 3:  # e.g. Gumbel softmax summaries
                emb = torch.matmul(input[:, t, :], self.embed.weight)

            hidden, cell, output = self.rnn(emb, hidden, cell)
            if all_hiddens or (t == seq_len - 1):
                hiddens.append(hidden)
            if all_cells or (t == seq_len - 1):
                cells.append(cell)
            if all_outputs or (t == seq_len - 1):
                outputs.append(output)

        return hiddens, cells, outputs


class StackedLSTMDecoder(nn.Module):
    def __init__(self, embed, rnn,
                 use_docs_attn=False, attn_emb_size=None, attn_hidden_size=None, attn_learn_alpha=False):
        """

        Args:
            embed: nn.Embedding
            rnn: StackedLSTM
            use_emb_attn: boolean (whether to attend over input embedding's of size [batch, attn_emb_size])
                - can be used to attend over multiple document representations
            attn_emb_size: int (size of embeddings being attended over, e.g. hidden_size for documents)
            attn_hidden_size: int (size of intermediate attention representation)
            attn_learn_alpha: boolean (whether to average the context with previous hidden state or learn the weighting)
        """
        super(StackedLSTMDecoder, self).__init__()
        self.embed = embed
        self.rnn = rnn

        self.use_docs_attn = use_docs_attn
        self.attn_emb_size = attn_emb_size
        self.attn_hidden_size = attn_hidden_size
        self.attn_learn_alpha = attn_learn_alpha

        if use_docs_attn:
            self.attn_lin1 = nn.Linear(attn_emb_size ,attn_hidden_size)
            self.attn_act1 = nn.Tanh()
            self.attn_lin2 = nn.Linear(attn_hidden_size, 1)
            self.context_alpha = nn.Parameter(torch.Tensor([0.5])) if self.attn_learn_alpha else 0.5

    def forward(self, init_hidden, init_cell, init_input,
                targets=None,
                seq_len=None, eos_id=EOS_ID, non_pad_prob_val=0,
                softmax_method='softmax', sample_method='sample',
                tau=1.0, eps=1e-10, gumbel_hard=False,
                encoder_hiddens=None, encoder_inputs=None, attend_to_embs=None,
                subwordenc=None,
                return_last_state=False, k=1):
        """
        Decode. If targets is given, then use teacher forcing.

        Notes:
            This is also used by beam search by setting seq_len=1 and k=beam_size.
            The comments talk about [batch * k^(t+1), but in practice this should only ever
            be called with seq_len=1 (and hence t=0). The results from one beam step are then pruned,
            before beam search repeats the step.

        Args:
            init_hidden: [batch_size, n_layers, hidden_size]
            init_cell: [batch_size, n_layers, hidden_size]
            init_input: [batch_size] (e.g. <EDOC> ids)

            # For teacher forcing
            targets: [batch_size, trg_seq_len]

            # For non-teacher forcing
            seq_len: int (length to generate)
            eos_id: int (generate until every sequence in batch has generated eos, or until seq_len)
            non_pad_prob_val: float
                When replacing tokens after eos_id, set probability of non-pad tokens to this value
                A small epsilon may be used if the log of the probs will be computed for a NLLLoss in order
                to prevent Nans.

            # Sampling, temperature, etc.
            softmax_method: str (which version of softmax to get probabilities; 'gumbel' or 'softmax')
            sample_method: str (how to sample words given probabilities; 'greedy', 'sample')
            tau: float (temperature for softmax)
            eps: float (controls sampling from Gumbel)
            gumbel_hard: boolean (whether to produce one hot encodings for Gumbel Softmax)
            subwordenc: SubwordTokenizer
                (returns text if given)

            # Additional inputs
            encoder_hiddens: [batch_size, seq_len, hidden_size]
                Hiddens at each time step. Would be used for attention
            encoder_inputs: [batch_size, seq_len]
                Would be used for a pointer network
            attend_to_embs: [batch_size, n_docs, n_layers, hidden_size]
                maybe just [batch, *, hidden]?
                Embs to attend to. Could be last hidden states (i.e. document representations)

            # Beam search
            return_last_state: bool
                (states used for beam search)
            k: int (i.e. beam width)

        Returns:
            decoded_probs: [batch * k^(gen_len), gen_len, vocab]
            decoded_ids: [batch * k^(gen_len), gen_len]
            decoded_texts: list of str's if subwordenc is given
            extra: dict of additional outputs
        """
        batch_size = init_input.size(0)
        output_len = seq_len if seq_len is not None else targets.size(1)
        vocab_size = self.rnn.h2o.out_features

        decoded_probs = move_to_cuda(torch.zeros(batch_size * k, output_len, vocab_size))
        decoded_ids = move_to_cuda(torch.zeros(batch_size * k, output_len).long())
        extra = {}

        rows_with_eos = move_to_cuda(torch.zeros(batch_size * k).long())  # track which sequences have generated eos_id
        pad_ids = move_to_cuda(torch.Tensor(batch_size * k).fill_(PAD_ID).long())
        pad_prob = move_to_cuda(torch.zeros(batch_size * k, vocab_size)).fill_(non_pad_prob_val)
        pad_prob[:, PAD_ID] = 1.0

        hidden, cell = init_hidden, init_cell  # [batch, n_layers, hidden]
        input = init_input.long()

        for t in range(output_len):
            if gumbel_hard and t != 0:
                input_emb = torch.matmul(input, self.embed.weight)
            else:
                input_emb = self.embed(input)  # [batch, emb_size]

            if self.use_docs_attn:
                attn_wts = self.attn_lin1(attend_to_embs)  # [batch, n_docs, n_layers, attn_size]
                attn_wts = self.attn_lin2(self.attn_act1(attn_wts))  # [batch, n_docs, n_layers, 1]
                attn_wts = F.softmax(attn_wts, dim=1)  # [batch, n_docs, n_layers, 1]
                context = attn_wts * attend_to_embs  # [batch, n_docs, n_layers, hidden]
                context = context.sum(dim=1)  # [batch, n_layers, hidden]
                hidden = self.context_alpha * context + (1 - self.context_alpha) * hidden

            hidden, cell, output = self.rnn(input_emb, hidden, cell)
            prob = logits_to_prob(output, softmax_method,
                                  tau=tau, eps=eps, gumbel_hard=gumbel_hard)  # [batch, vocab]
            prob, id = prob_to_vocab_id(prob, sample_method, k=k)  # [batch * k^(t+1)]

            # If sequence (row) has *previously* produced an EOS,
            # replace prob with one hot (probability one for pad) and id with pad
            prob = torch.where((rows_with_eos == 1).unsqueeze(1), pad_prob, prob)  # unsqueeze to broadcast
            id = torch.where(rows_with_eos == 1, pad_ids, id)
            # Now update rows_with_eos to include this time step
            # This has to go after the above! Otherwise EOS is replaced as well
            rows_with_eos = rows_with_eos | (id == eos_id).long()

            decoded_probs[:, t, :] = prob
            decoded_ids[:, t] = id

            # Get next input
            if targets is not None:  # teacher forcing
                input = targets[:, t]  # [batch]
            else:  # non-teacher forcing
                if gumbel_hard:
                    input = prob
                else:
                    input = id  # [batch * k^(t+1)]

            # Terminate early if not teacher forcing and all sequences have generated an eos
            if targets is None:
                if rows_with_eos.sum().item() == (batch_size * k):
                    break

        # if return_last_state:
        #     extra['last_state'] = states

        decoded_texts = []
        if subwordenc:
            for i in range(batch_size):
                decoded_texts.append(subwordenc.decode(decoded_ids[i].long().tolist()))

        return decoded_probs, decoded_ids, decoded_texts, extra


class StackedLSTMEncoderDecoder(nn.Module):
    def __init__(self, embed, rnn):
        super(StackedLSTMEncoderDecoder, self).__init__()
        self.embed = embed
        self.rnn = rnn
        self.encoder = StackedLSTMEncoder(embed, rnn)
        self.decoder = StackedLSTMDecoder(embed, rnn)

    def forward(self, input,
                enc_init_h=None, enc_init_c=None,
                dec_init_input=None, dec_kwargs={}):
        """
        Args:
            input: [batch_size, seq_len]
            enc_init_h: [batch_size, n_layers, hidden_size]
            enc_init_c: [batch_size, n_layers, hidden_size]
            dec_init_input: [batch_size]
            dec_kwargs: dict

        Returns:
            Output of StackedLSTMDecoder's forward()
        """
        if (enc_init_h is None) and (enc_init_c is None):
            batch_size = input.size(0)
            enc_init_h, enc_init_c = self.rnn.state0(batch_size)
            enc_init_h, enc_init_c = move_to_cuda(enc_init_h), move_to_cuda(enc_init_c)

        hiddens, cells, outputs = self.encoder(input, enc_init_h, enc_init_c)

        # Get states and input for decoder
        last_hidden, last_cell, last_logits = hiddens[-1], cells[-1], outputs[-1]
        if dec_init_input is None:
            last_probs = logits_to_prob(last_logits, method='softmax')  # [batch, vocab]
            _, dec_init_input = prob_to_vocab_id(last_probs, 'greedy')  # [batch]

        probs, ids, texts, extra = self.decoder(last_hidden, last_cell, dec_init_input, **dec_kwargs)
        return probs, ids, texts, extra
