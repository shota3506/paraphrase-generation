import math
from statistics import mean
from typing import Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .modules.positional_encoder import PositionalEncoder


class Transformer(nn.Module):
    """
    A transformer model
    "Attention Is All You Need". Ashish Vaswani, Noam Shazeer,
    Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and
    Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information
    Processing Systems, pages 6000-6010.

    Args:
        d_model: the number of expected features in the encoder/decoder inputs (default=512).
        nhead: the number of heads in the multiheadattention models (default=8).
        num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
        num_decoder_layers: the number of sub-decoder-layers in the decoder (default=6).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of encoder/decoder intermediate layer, relu or gelu (default=relu).
    """

    def __init__(
        self, 
        num_embeddings: int,
        d_model: int = 512, 
        nhead: int = 8, 
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6, 
        dim_feedforward: int = 2048, 
        dropout: float = 0.1,
        activation: str = "relu", 
    ) -> None:
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead

        self.embedding = nn.Embedding(num_embeddings, d_model)
        self.pos_encoder = PositionalEncoder(d_model, dropout)
        self.dropout = nn.Dropout(dropout)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation), 
            num_encoder_layers, 
            nn.LayerNorm(d_model))

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation), 
            num_decoder_layers, 
            nn.LayerNorm(d_model))

        self.fc = nn.Linear(d_model, num_embeddings)

        self._init_weights()

    def _init_weights(self):        
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self, 
        src: Tensor, 
        tgt: Tensor, 
        src_mask: Optional[Tensor] = None, 
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None, 
        src_key_padding_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None, 
        memory_key_padding_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Args:
            src: Tensor `(N, S)`, the sequence to the encoder (required).
            tgt: Tensor `(N, T)`, the sequence to the decoder (required).
            src_mask: Tensor `(S, S)`, the additive mask for the src sequence (optional).
            tgt_mask: Tensor `(T, T)`, the additive mask for the tgt sequence (optional).
            memory_mask: Tensor `(T, S)`, the additive mask for the encoder output (optional).
            src_key_padding_mask: Tensor `(N, S)`, the ByteTensor mask for src keys per batch (optional).
            tgt_key_padding_mask: Tensor `(N, T)`, the ByteTensor mask for tgt keys per batch (optional).
            memory_key_padding_mask: Tensor `(N, S)`, the ByteTensor mask for memory keys per batch (optional).
        Output:
            Tensor `(T, N, E)`

            where S is the source sequence length, T is the target sequence length, N is the
            batch size, E is the feature number
        """

        memory = self.encode(src, src_mask, src_key_padding_mask)        
        output = self.decode(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask)
        return output

    def encode(
        self, 
        src: Tensor, 
        src_mask: Optional[Tensor] = None, 
        src_key_padding_mask: Optional[Tensor] = None
    ) -> Tensor:
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = src.permute(1, 0, 2).contiguous()
        src = self.dropout(self.pos_encoder(src))
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        return memory

    def decode(
        self,
        tgt: Tensor, 
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None, 
        tgt_key_padding_mask: Optional[Tensor] = None, 
        memory_key_padding_mask: Optional[Tensor] = None
    ) -> Tensor:
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt = tgt.permute(1, 0, 2).contiguous()
        tgt = self.dropout(self.pos_encoder(tgt))

        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        output = output.permute(1, 0, 2).contiguous()
        output = self.fc(output)
        return output

    def step(self, last_predictions, state, timestep):
        memory = state['memory'].permute(1, 0, 2).contiguous()
        prev_output_tokens = state['prev_output_tokens']
        memory_key_padding_mask = state['memory_key_padding_mask']

        if prev_output_tokens is None:
            prev_output_tokens = last_predictions.unsqueeze(1)
        else:
            prev_output_tokens = torch.cat((prev_output_tokens, last_predictions.unsqueeze(1)), dim=1)

        tgt_mask = self.generate_square_subsequent_mask(prev_output_tokens.size(1)).to(prev_output_tokens.device)
        output = self.decode(prev_output_tokens, memory, tgt_mask=tgt_mask, memory_key_padding_mask=memory_key_padding_mask)
        output = output[:, -1, :]
        log_probabilities = F.log_softmax(output, dim=-1)

        memory = memory.permute(1, 0, 2)

        return log_probabilities, {'memory': memory, 'prev_output_tokens': prev_output_tokens,  'memory_key_padding_mask': memory_key_padding_mask}
        
    def generate_square_subsequent_mask(self, sz: int) -> Tensor:
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

