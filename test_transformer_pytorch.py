import unittest
import torch
from transformer_pytorch import Transformer
from transformer_layers import (
    MultiHeadAttention, PositionWiseFeedForward,
    PositionalEncoding, EncoderLayer, DecoderLayer
)

class TestTransformer(unittest.TestCase):
    def setUp(self):
        self.vocab_size = 100
        self.embed_dim = 32
        self.num_heads = 4
        self.ff_hidden_dim = 64
        self.num_layers = 2
        self.max_len = 50
        self.model = Transformer(
            src_vocab_size=self.vocab_size,
            tgt_vocab_size=self.vocab_size,
            d_model=self.embed_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            d_ff=self.ff_hidden_dim,
            max_seq_length=self.max_len,
            dropout=0.1
        )

    def test_forward_shape(self):
        src = torch.randint(1, self.vocab_size, (4, 10))
        tgt = torch.randint(1, self.vocab_size, (4, 10))
        out = self.model(src, tgt)
        self.assertEqual(out.shape, (4, 10, self.vocab_size))

    def test_autograd(self):
        src = torch.randint(1, self.vocab_size, (2, 8))
        tgt = torch.randint(1, self.vocab_size, (2, 8))
        out = self.model(src, tgt)
        loss = out.mean()
        loss.backward()
        grads = [p.grad is not None for p in self.model.parameters() if p.requires_grad]
        self.assertTrue(all(grads))

    def test_forward_consistency(self):
        src = torch.randint(1, self.vocab_size, (1, 10))
        tgt = torch.randint(1, self.vocab_size, (1, 10))
        out1 = self.model(src, tgt)
        out2 = self.model(src, tgt)
        self.assertEqual(out1.shape, out2.shape)

class TestTransformerLayers(unittest.TestCase):
    def test_multihead_attention(self):
        mha = MultiHeadAttention(d_model=32, num_heads=4)
        x = torch.randn(2, 10, 32)
        out = mha(x, x, x)
        self.assertEqual(out.shape, (2, 10, 32))

    def test_feedforward(self):
        ffn = PositionWiseFeedForward(d_model=32, d_ff=64)
        x = torch.randn(2, 10, 32)
        out = ffn(x)
        self.assertEqual(out.shape, (2, 10, 32))

    def test_positional_encoding(self):
        pe = PositionalEncoding(d_model=32, max_seq_length=50)
        x = torch.randn(2, 10, 32)
        out = pe(x)
        self.assertEqual(out.shape, (2, 10, 32))

    def test_encoder_layer(self):
        enc = EncoderLayer(d_model=32, num_heads=4, d_ff=64, dropout=0.1)
        x = torch.randn(2, 10, 32)
        mask = torch.ones(2, 1, 1, 10)
        out = enc(x, mask)
        self.assertEqual(out.shape, (2, 10, 32))

    def test_decoder_layer(self):
        dec = DecoderLayer(d_model=32, num_heads=4, d_ff=64, dropout=0.1)
        x = torch.randn(2, 10, 32)
        enc_out = torch.randn(2, 10, 32)
        mask = torch.ones(2, 1, 1, 10)
        out = dec(x, enc_out, mask, mask)
        self.assertEqual(out.shape, (2, 10, 32))

if __name__ == '__main__':
    unittest.main()
