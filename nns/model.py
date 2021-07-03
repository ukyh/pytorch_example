import torch
import torch.nn as nn
import torch.nn.functional as F

from nns.modules import BiLSTM


class Model(nn.Module):

    def __init__(self, args):
        super(Model, self).__init__()
        # Embedder
        self.embedder = nn.Embedding(args.vocab_size, args.dim_tok, padding_idx=0)

        # Encoder
        self.seq_encoder = BiLSTM(
            args.dim_tok, args.dim_hid, args.nlayer_enc, args.drop_seq, args.out_pad
        )
        self.seq_dropout = nn.Dropout(p=args.drop_seq)

        self.att_query = nn.Linear(args.dim_hid * 2, 1)
        self.pred_proj = nn.Linear(args.dim_hid * 2, 2)

        self.device = args.device

    def forward(self, batch):
        """Predict whether a given sentence is subjective or objective.
        
        Args:
            text_tensor: LongTensor (batch, max_slen)
            lengths: LongTensor (batch)
            labels: LongTensor (batch)

        Returns:
            preds: FloatTensor (batch, 2)
            labels: LongTensor (batch)
        """
        # Send input tensors to CPU/GPU
        text_tensor, lengths, labels = batch
        text_tensor = text_tensor.to(self.device)
        # lengths = lengths.to(self.device)
        labels = labels.to(self.device)

        # Embed tokens
        tok_embs = self.embedder(text_tensor)       # (batch, max_slen, dim_tok)

        # Encode sequences of token embeddings
        seq_embs, _ = self.seq_encoder(tok_embs, lengths)  # (batch, max_slen, dim_hid * 2)

        # Make padding mask for attention.
        mask_tensor = torch.empty_like(text_tensor)     # (batch, max_slen)
        mask_tensor.fill_(-1e+12)                       # initialize the mask with -inf
        for i, slen in enumerate(lengths):
            mask_tensor[i][:slen] = torch.zeros(slen)   # fill the words with zeros
        mask_tensor = mask_tensor.unsqueeze(1)          # (batch, 1, max_slen)
        mask_tensor = mask_tensor.to(self.device)       # send to cpu/gpu

        # Apply attention
        att_weights = self.att_query(seq_embs)          # (batch, max_slen, 1)
        att_weights = att_weights.transpose(1, 2)       # (batch, 1, max_slen)
        att_weights = att_weights + mask_tensor         # fill the paddings with -inf to make them 0 in the normalized attention weights
        att_weights = torch.softmax(
            att_weights, dim=-1
        )                                               # take softmax over the last dimension
        att_outs = torch.bmm(
            att_weights, seq_embs
        ).squeeze(1)                                    # (batch, 1, dim_hid * 2) -> (batch, dim_hid * 2)

        # Transform for prediction
        preds = self.pred_proj(att_outs)                # (batch, 2)

        return preds, labels
