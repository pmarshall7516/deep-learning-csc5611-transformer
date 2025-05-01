import torch
import math
from layers import *

class Transformer:
    def __init__(self, vocab_size, d_model, num_heads, d_ff, max_seq_len):
        self.vocab_size = vocab_size
        self.d_model = d_model

        emb_std = math.sqrt(2.0 / (vocab_size + d_model))
        self.tok_embed = Parameter(torch.randn(vocab_size, d_model).to(DEVICE) * emb_std)
        self.pos_embed = PositionalEncoding(max_seq_len, d_model)

        self.input_layer = Input((1, max_seq_len, d_model))
        self.decoder_input = Input((1, max_seq_len, d_model))
        self.encoder_output = Input((1, max_seq_len, d_model))

        # Encoder weights
        self.encoder_norm1 = Parameter(torch.ones(d_model).to(DEVICE))
        self.encoder_norm2 = Parameter(torch.ones(d_model).to(DEVICE))
        attn_std = math.sqrt(2.0 / (2 * d_model))
        ff_std = math.sqrt(2.0 / (d_model + d_ff))
        self.W_q_enc = Parameter(torch.randn(d_model, d_model).to(DEVICE) * attn_std)
        self.W_k_enc = Parameter(torch.randn(d_model, d_model).to(DEVICE) * attn_std)
        self.W_v_enc = Parameter(torch.randn(d_model, d_model).to(DEVICE) * attn_std)
        self.W_o_enc = Parameter(torch.randn(d_model, d_model).to(DEVICE) * attn_std)
        self.W1_enc = Parameter(torch.randn(d_model, d_ff).to(DEVICE) * ff_std)
        self.b1_enc = Parameter(torch.zeros(d_ff).to(DEVICE))
        self.W2_enc = Parameter(torch.randn(d_ff, d_model).to(DEVICE) * ff_std)
        self.b2_enc = Parameter(torch.zeros(d_model).to(DEVICE))

        self.encoder = EncoderLayer(
            self.input_layer, d_model, num_heads, d_ff,
            self.W_q_enc, self.W_k_enc, self.W_v_enc, self.W_o_enc,
            self.W1_enc, self.b1_enc, self.W2_enc, self.b2_enc,
            self.encoder_norm1, self.encoder_norm2
        )

        # Decoder weights
        self.decoder_norm1 = Parameter(torch.ones(d_model).to(DEVICE))
        self.decoder_norm2 = Parameter(torch.ones(d_model).to(DEVICE))
        self.decoder_norm3 = Parameter(torch.ones(d_model).to(DEVICE))
        self.W_q_dec_self = Parameter(torch.randn(d_model, d_model).to(DEVICE) * attn_std)
        self.W_k_dec_self = Parameter(torch.randn(d_model, d_model).to(DEVICE) * attn_std)
        self.W_v_dec_self = Parameter(torch.randn(d_model, d_model).to(DEVICE) * attn_std)
        self.W_o_dec_self = Parameter(torch.randn(d_model, d_model).to(DEVICE) * attn_std)
        self.W_q_dec_cross = Parameter(torch.randn(d_model, d_model).to(DEVICE) * attn_std)
        self.W_k_dec_cross = Parameter(torch.randn(d_model, d_model).to(DEVICE) * attn_std)
        self.W_v_dec_cross = Parameter(torch.randn(d_model, d_model).to(DEVICE) * attn_std)
        self.W_o_dec_cross = Parameter(torch.randn(d_model, d_model).to(DEVICE) * attn_std)
        self.W1_dec = Parameter(torch.randn(d_model, d_ff).to(DEVICE) * ff_std)
        self.b1_dec = Parameter(torch.zeros(d_ff).to(DEVICE))
        self.W2_dec = Parameter(torch.randn(d_ff, d_model).to(DEVICE) * ff_std)
        self.b2_dec = Parameter(torch.zeros(d_model).to(DEVICE))

        self.decoder = DecoderLayer(
            self.decoder_input, self.encoder_output, d_model, num_heads, d_ff,
            self.W_q_dec_self, self.W_k_dec_self, self.W_v_dec_self, self.W_o_dec_self,
            self.W_q_dec_cross, self.W_k_dec_cross, self.W_v_dec_cross, self.W_o_dec_cross,
            self.W1_dec, self.b1_dec, self.W2_dec, self.b2_dec,
            self.decoder_norm1, self.decoder_norm2, self.decoder_norm3
        )

        out_std = math.sqrt(1.0 / d_model)
        self.final_linear_W = Parameter(torch.randn(d_model, vocab_size).to(DEVICE) * out_std)
        self.final_linear_b = Parameter(torch.zeros(vocab_size).to(DEVICE))

    def forward(self, src):
        batch_size, seq_len = src.shape
        embedded = torch.zeros(batch_size, seq_len, self.d_model, device=DEVICE)
        for b in range(batch_size):
            for t in range(seq_len):
                embedded[b, t] = self.tok_embed.output[src[b, t]]
        embedded *= math.sqrt(self.d_model)

        input_with_pos = Input(embedded.shape)
        input_with_pos.set(self.pos_embed.add_to(Input(embedded.shape)))
        self.input_layer.set(input_with_pos.output)
        self.decoder_input.set(input_with_pos.output)

        self.encoder.forward()
        self.encoder_output.set(self.encoder.output)

        self.decoder.forward()

        final_inp = Input(self.decoder.output.shape)
        final_inp.set(self.decoder.output)
        self.final_linear = Linear(final_inp, self.final_linear_W, self.final_linear_b)
        self.final_linear.forward()
        return self.final_linear.output

    def backward(self, loss_grad):
        self.final_linear.grad = loss_grad
        self.final_linear.backward()
        self.decoder.grad = self.final_linear.x.grad
        self.decoder.backward()
        self.encoder.grad = self.encoder_output.grad
        self.encoder.backward()

    def clear_gradients(self):
        for param in self.parameters():
            param.clear_grad()

    def step(self, learning_rate=0.0001):
        for param in self.parameters():
            param.step(learning_rate)

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
