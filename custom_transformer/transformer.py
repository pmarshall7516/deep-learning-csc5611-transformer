"""
Patrick Marshall
CSC 5611 Deep Learning
01 May 2025

transformer.py
This file contains the Transformer class, which implements a simplified version of the
Transformer architecture. The class includes methods for initializing the model,
forward propagation, and parameter retrieval. The model consists of an encoder and decoder
with multi-head self-attention and feed-forward layers. The class also includes token and
positional embeddings, as well as a final linear layer for outputting predictions.
"""

import torch
from layers import *

class Transformer:
    def __init__(self, vocab_size, d_model, num_heads, d_ff, max_seq_len):
        # Embeddings
        self.tok_embed = Parameter(torch.randn(vocab_size, d_model).to(DEVICE))
        self.pos_embed = PositionalEncoding(max_seq_len, d_model)
        
        # Input
        self.input_layer = Input((1, max_seq_len, d_model))  # Placeholder input shape
        
        # Encoder
        self.encoder_norm1 = Parameter(torch.ones(1, 1, d_model).to(DEVICE))
        self.encoder_norm2 = Parameter(torch.ones(1, 1, d_model).to(DEVICE))
        
        self.W_q_enc = Parameter(torch.randn(d_model, d_model).to(DEVICE))
        self.W_k_enc = Parameter(torch.randn(d_model, d_model).to(DEVICE))
        self.W_v_enc = Parameter(torch.randn(d_model, d_model).to(DEVICE))
        self.W_o_enc = Parameter(torch.randn(d_model, d_model).to(DEVICE))
        
        self.W1_enc = Parameter(torch.randn(d_model, d_ff).to(DEVICE))
        self.b1_enc = Parameter(torch.randn(d_ff).to(DEVICE))
        self.W2_enc = Parameter(torch.randn(d_ff, d_model).to(DEVICE))
        self.b2_enc = Parameter(torch.randn(d_model).to(DEVICE))
        
        self.encoder = EncoderLayer(
            self.input_layer, d_model, num_heads, d_ff,
            self.W_q_enc, self.W_k_enc, self.W_v_enc, self.W_o_enc,
            self.W1_enc, self.b1_enc, self.W2_enc, self.b2_enc,
            self.encoder_norm1, self.encoder_norm2
        )
        
        # Decoder
        self.decoder_norm1 = Parameter(torch.ones(1, 1, d_model).to(DEVICE))
        self.decoder_norm2 = Parameter(torch.ones(1, 1, d_model).to(DEVICE))
        self.decoder_norm3 = Parameter(torch.ones(1, 1, d_model).to(DEVICE))
        
        self.W_q_dec_self = Parameter(torch.randn(d_model, d_model).to(DEVICE))
        self.W_k_dec_self = Parameter(torch.randn(d_model, d_model).to(DEVICE))
        self.W_v_dec_self = Parameter(torch.randn(d_model, d_model).to(DEVICE))
        self.W_o_dec_self = Parameter(torch.randn(d_model, d_model).to(DEVICE))
        
        self.W_q_dec_cross = Parameter(torch.randn(d_model, d_model).to(DEVICE))
        self.W_k_dec_cross = Parameter(torch.randn(d_model, d_model).to(DEVICE))
        self.W_v_dec_cross = Parameter(torch.randn(d_model, d_model).to(DEVICE))
        self.W_o_dec_cross = Parameter(torch.randn(d_model, d_model).to(DEVICE))
        
        self.W1_dec = Parameter(torch.randn(d_model, d_ff).to(DEVICE))
        self.b1_dec = Parameter(torch.randn(d_ff).to(DEVICE))
        self.W2_dec = Parameter(torch.randn(d_ff, d_model).to(DEVICE))
        self.b2_dec = Parameter(torch.randn(d_model).to(DEVICE))
        
        self.decoder = DecoderLayer(
            self.input_layer, self.encoder,
            d_model, num_heads, d_ff,
            self.W_q_dec_self, self.W_k_dec_self, self.W_v_dec_self, self.W_o_dec_self,
            self.W_q_dec_cross, self.W_k_dec_cross, self.W_v_dec_cross, self.W_o_dec_cross,
            self.W1_dec, self.b1_dec, self.W2_dec, self.b2_dec,
            self.decoder_norm1, self.decoder_norm2, self.decoder_norm3
        )
        
        # Final linear to vocab
        self.final_linear_W = Parameter(torch.randn(d_model, vocab_size).to(DEVICE))
        self.final_linear_b = Parameter(torch.randn(vocab_size).to(DEVICE))
        
    def forward(self, src):
        # Embed tokens
        embedded = src @ self.tok_embed.output  # Shape (B, T, d_model)
        embedded = self.pos_embed.add_to(Input(embedded.shape))
        
        self.input_layer.set(embedded)
        self.encoder.forward()
        self.decoder.forward()
        
        # Project to vocab
        final_inp = Input(self.decoder.output.shape)
        final_inp.set(self.decoder.output)
        self.final_linear = Linear(final_inp, self.final_linear_W, self.final_linear_b)
        self.final_linear.forward()
        
        return self.final_linear.output

    def parameters(self):
        return [
            self.tok_embed, self.final_linear_W, self.final_linear_b,
            self.W_q_enc, self.W_k_enc, self.W_v_enc, self.W_o_enc,
            self.W1_enc, self.b1_enc, self.W2_enc, self.b2_enc,
            self.W_q_dec_self, self.W_k_dec_self, self.W_v_dec_self, self.W_o_dec_self,
            self.W_q_dec_cross, self.W_k_dec_cross, self.W_v_dec_cross, self.W_o_dec_cross,
            self.W1_dec, self.b1_dec, self.W2_dec, self.b2_dec,
            self.encoder_norm1, self.encoder_norm2,
            self.decoder_norm1, self.decoder_norm2, self.decoder_norm3
        ]
