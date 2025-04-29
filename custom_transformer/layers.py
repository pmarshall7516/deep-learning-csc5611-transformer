"""
Patrick Marshall
CSC 5611 Deep Learning
01 May 2025

layers.py
This file contains the implementation of various layers used in a transformer model.
"""

import torch
import math

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Layer:
    def __init__(self):
        self.output = None 
        self.grad = None

    def accumulate_grad(self, grad):
        if self.grad is None:
            self.grad = grad.clone()
        else:
            self.grad += grad

    def clear_grad(self):
        if self.grad is not None:
            self.grad.zero_()

    def step(self, alpha=0.01):
        pass

    def forward(self):
        raise NotImplementedError("Forward method must be implemented in subclasses.")
    
class Input(Layer):
    def __init__(self, shape, train=False):
        Layer.__init__(self)
        self.output = torch.zeros(shape, device=DEVICE)
        self.train = train

    def set(self, output):
        if self.output.shape != output.shape:
            raise ValueError("Shape mismatch in Input layer.")
        self.output = output.to(DEVICE).clone()

    def randomize(self):
        self.output.normal_()

    def forward(self):
        pass

    def backward(self):
        pass

class Parameter(Input):
    def __init__(self, value):
        super().__init__(value.shape, train=True)
        self.set(value)
    
    def step(self, alpha=0.01):
        if self.train and self.grad is not None:
            self.output -= alpha * self.grad

    def accumulate_grad(self, grad):
        if self.grad is None:
            self.grad = grad.clone()
        else:
            self.grad += grad


class Linear(Layer):
    def __init__(self, input_layer, weight_layer, bias_layer):
        Layer.__init__(self)
        self.input_layer = input_layer
        self.weight_layer = weight_layer  
        self.bias_layer = bias_layer      

    def forward(self):
        self.output = self.input_layer.output @ self.weight_layer.output + self.bias_layer.output

    def backward(self):
        grad_out = self.grad
        grad_input = grad_out @ self.weight_layer.output.T
        grad_weights = self.input_layer.output.T @ grad_out
        grad_bias = grad_out.sum(dim=0)

        self.input_layer.accumulate_grad(grad_input)
        self.weight_layer.accumulate_grad(grad_weights)
        self.bias_layer.accumulate_grad(grad_bias)


class ReLU(Layer):
    def __init__(self, input_layer):
        Layer.__init__(self)
        self.input_layer = input_layer

    def forward(self):
        self.output = torch.relu(self.input_layer.output)

    def backward(self):
        grad_input = self.grad.clone()
        grad_input[self.input_layer.output <= 0] = 0
        self.input_layer.accumulate_grad(grad_input)

class MSELoss(Layer):
    def __init__(self, prediction_layer, target_layer):
        Layer.__init__(self)
        self.prediction_layer = prediction_layer
        self.target_layer = target_layer

    def forward(self):
        self.output = torch.mean((self.prediction_layer.output - self.target_layer.output) ** 2)

    def backward(self):
        grad_out = self.grad
        grad = self.grad * (self.prediction_layer.output - self.target_layer.output)
        self.prediction_layer.accumulate_grad(grad)


class Regularization(Layer):
    def __init__(self, input_layer, lambda_):
        Layer.__init__(self)
        self.input_layer = input_layer
        self.lambda_ = lambda_

    def forward(self):
        self.output = self.lambda_ * torch.sum(self.input_layer.output ** 2)

    def backward(self):
        grad_out = self.grad
        grad = 2 * self.lambda_ * grad_out * self.input_layer.output
        self.input_layer.accumulate_grad(grad)

class Softmax(Layer):
    def __init__(self, input_layer):
        super().__init__()
        self.input_layer = input_layer

    def forward(self):
        exp_vals = torch.exp(self.input_layer.output)
        self.output = exp_vals / torch.sum(exp_vals, dim=-1, keepdim=True)

    def backward(self):
        grad_out = self.grad
        softmax = self.output
        # Softmax backward: Jacobian-vector product
        grad_input = grad_out - (grad_out * softmax).sum(dim=-1, keepdim=True) * softmax
        self.input_layer.accumulate_grad(grad_input)

class CrossEntropyLoss(Layer):
    def __init__(self, input_layer, target_layer):
        super().__init__()
        self.input_layer = input_layer  # Output from softmax
        self.target_layer = target_layer  # One-hot encoded target

    def forward(self):
        self.output = -torch.sum(self.target_layer.output * torch.log(self.input_layer.output + 1e-8), dim=-1).mean()

    def backward(self):
        grad_out = self.grad
        grad = - (self.target_layer.output / (self.input_layer.output + 1e-8)) / self.target_layer.output.shape[0]
        self.input_layer.accumulate_grad(grad_out * grad)

class SoftmaxCrossEntropy(Layer):
    def __init__(self, input_layer):
        Layer.__init__(self)
        self.input_layer = input_layer
        self.classifications = None
        self.targets = None

    def forward(self):
        exp_vals = torch.exp(self.input_layer.output)
        self.classifications = exp_vals / torch.sum(exp_vals, dim=-1, keepdim=True) # Softmax
        self.output = -torch.sum(self.targets * torch.log(self.classifications + 1e-8), dim=1) # Cross-entropy loss

    def backward(self):
        grad_out = self.grad
        grad = grad_out * (self.classifications - self.targets)
        self.input_layer.accumulate_grad(grad)


class Sum(Layer):
    def __init__(self, *input_layers):
        Layer.__init__(self)
        self.input_layers = list(input_layers)

    def forward(self):
        self.output = sum(layer.output for layer in self.input_layers)

    def backward(self):
        for layer in self.input_layers:
            layer.accumulate_grad(self.grad)

class MatMul(Layer):
    def __init__(self, A_layer, B_layer):
        super().__init__()
        self.A_layer = A_layer
        self.B_layer = B_layer

    def forward(self):
        self.output = self.A_layer.output @ self.B_layer.output

    def backward(self):
        grad_output = self.grad
        grad_A = grad_output @ self.B_layer.output.T
        grad_B = self.A_layer.output.T @ grad_output
        self.A_layer.accumulate_grad(grad_A)
        self.B_layer.accumulate_grad(grad_B)
    

class RMSNorm(Layer):
    """
    This performs RMS Normalization

    g is the scaling factor from the paper.

    The number of dimensions in g indicate which of the last
    dimensions in x indicate a sample passing through the layer.

    https://arxiv.org/pdf/1910.07467
    """
    def __init__(self,x,gamma):
        """
        :param x: input to be normalized
        """
        Layer.__init__(self)    
        # Either the dimensions of g should match the last dimensions of x.
        assert gamma.output.shape[-1] == x.output.shape[-1], "gamma and x must match in the last dimension only"
        self.x = x
        self.gamma = gamma

    def forward(self):
        self.std = torch.sqrt((self.x.output**2).sum(axis=-1, keepdims=True) / self.x.output.shape[-1])
        self.output = self.x.output/(self.std+1e-9)*self.gamma.output

    def backward(self):
        dJdgamma = (self.x.output/(self.std+1e-9)*self.grad).sum(axis=list(range(0,len(self.x.output.shape)-1)),keepdims=True)
        simple = self.grad * self.gamma.output / self.std

        dJdx = simple - self.x.output * (simple * self.x.output).sum(axis=-1,keepdims=True) / (
            self.x.output.shape[-1] * self.std**2
        )
        self.gamma.accumulate_grad(dJdgamma)
        self.x.accumulate_grad(dJdx)

class MultiHeadAttention(Layer):
    def __init__(self, query, key, value, d_model, num_heads, W_q, W_k, W_v, W_o, mask=None):
        super().__init__()
        assert d_model % num_heads == 0
        self.query, self.key, self.value = query, key, value
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q, self.W_k, self.W_v, self.W_o = W_q, W_k, W_v, W_o
        self.mask = mask

    def split_heads(self, x):
        B, T, C = x.shape
        return x.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        B, H, T, D = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * D)

    def forward(self):
        # Linear projections
        q_linear = Linear(self.query, self.W_q, Parameter(torch.zeros(self.d_model).to(DEVICE)))
        k_linear = Linear(self.key, self.W_k, Parameter(torch.zeros(self.d_model).to(DEVICE)))
        v_linear = Linear(self.value, self.W_v, Parameter(torch.zeros(self.d_model).to(DEVICE)))
        q_linear.forward()
        k_linear.forward()
        v_linear.forward()

        Q = self.split_heads(q_linear.output)
        K = self.split_heads(k_linear.output)
        V = self.split_heads(v_linear.output)

        # Scaled Dot-Product
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        if self.mask is not None:
            scores = scores.masked_fill(self.mask == 0, -1e9)

        # Use your custom Softmax layer
        scores_input = Input(scores.shape)
        scores_input.set(scores)

        self.softmax = Softmax(scores_input)
        self.softmax.forward()
        attn_probs = self.softmax.output

        attn_output = attn_probs @ V

        combined = self.combine_heads(attn_output)

        # Output projection
        o_input = Input(combined.shape)
        o_input.set(combined.to(DEVICE))
        self.o_proj = Linear(o_input, self.W_o, Parameter(torch.zeros(self.d_model).to(DEVICE)))
        self.o_proj.forward()

        self.output = self.o_proj.output

    def backward(self):
        self.o_proj.grad = self.grad
        self.o_proj.backward()

        self.softmax.grad = None
        self.softmax.backward()

class PositionWiseFeedForward(Layer):
    def __init__(self, x, W1, b1, W2, b2):
        super().__init__()
        self.x = x
        self.linear1 = Linear(x, W1, b1)
        self.relu = ReLU(self.linear1)
        self.linear2 = Linear(self.relu, W2, b2)

    def forward(self):
        self.linear1.forward()
        self.relu.forward()
        self.linear2.forward()
        self.output = self.linear2.output

    def backward(self):
        self.linear2.grad = self.grad
        self.linear2.backward()
        self.relu.backward()
        self.linear1.backward()

class PositionalEncoding(Layer):
    def __init__(self, max_seq_len, d_model):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model, device=DEVICE)
        position = torch.arange(0, max_seq_len, device=DEVICE).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2, device=DEVICE).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # Shape: (1, max_seq_len, d_model)

    def add_to(self, x_layer):
        self.output = x_layer.output + self.pe[:, :x_layer.output.shape[1]]
        return self.output
    
class EncoderLayer(Layer):
    def __init__(self, x, d_model, num_heads, d_ff, W_q, W_k, W_v, W_o, W1, b1, W2, b2, norm1_weight, norm2_weight):
        super().__init__()
        self.attn = MultiHeadAttention(x, x, x, d_model, num_heads, W_q, W_k, W_v, W_o)
        self.ff = PositionWiseFeedForward(x, W1, b1, W2, b2)
        self.norm1 = RMSNorm(x, norm1_weight)
        self.norm2 = RMSNorm(x, norm2_weight)
        self.x_input = x

    def forward(self):
        # Attention
        self.attn.forward()
        attn_output = self.attn.output

        # Residual connection + normalization
        add1 = Input(self.x_input.output.shape)
        add1.set(self.x_input.output + attn_output)
        self.norm1.x = add1
        self.norm1.forward()

        # Feed-forward
        self.ff.x = Input(self.norm1.output.shape)
        self.ff.x.set(self.norm1.output.clone())
        self.ff.forward()
        ff_output = self.ff.output

        # Second residual connection + normalization
        add2 = Input(self.norm1.output.shape)
        add2.set(self.norm1.output + ff_output)
        self.norm2.x = add2
        self.norm2.forward()

        self.output = self.norm2.output

    def backward(self):
        self.norm2.grad = self.grad
        self.norm2.backward()

        self.ff.grad = self.norm2.x.grad
        self.ff.backward()

        self.norm1.grad = self.ff.x.grad
        self.norm1.backward()

        self.attn.grad = self.norm1.x.grad
        self.attn.backward()

class DecoderLayer(Layer):
    def __init__(self, x, enc_output, d_model, num_heads, d_ff, 
                 W_q_self, W_k_self, W_v_self, W_o_self,
                 W_q_cross, W_k_cross, W_v_cross, W_o_cross,
                 W1, b1, W2, b2,
                 norm1_weight, norm2_weight, norm3_weight):
        super().__init__()
        # Self-attention
        self.self_attn = MultiHeadAttention(x, x, x, d_model, num_heads, W_q_self, W_k_self, W_v_self, W_o_self)
        # Cross-attention
        self.cross_attn = MultiHeadAttention(x, enc_output, enc_output, d_model, num_heads, W_q_cross, W_k_cross, W_v_cross, W_o_cross)
        # Feed-forward
        self.ff = PositionWiseFeedForward(x, W1, b1, W2, b2)

        self.norm1 = RMSNorm(x, norm1_weight)
        self.norm2 = RMSNorm(x, norm2_weight)
        self.norm3 = RMSNorm(x, norm3_weight)

        self.x_input = x
        self.enc_output = enc_output

    def forward(self):
        # Self-attention
        self.self_attn.forward()
        attn_self = self.self_attn.output

        add1 = Input(self.x_input.output.shape)
        add1.set(self.x_input.output + attn_self)
        self.norm1.x = add1
        self.norm1.forward()

        # Cross-attention
        self.cross_attn.query = Input(self.norm1.output.shape)
        self.cross_attn.query.set(self.norm1.output)
        self.cross_attn.forward()
        attn_cross = self.cross_attn.output

        add2 = Input(self.norm1.output.shape)
        add2.set(self.norm1.output + attn_cross)
        self.norm2.x = add2
        self.norm2.forward()

        # Feed-forward
        self.ff.x = Input(self.norm2.output.shape)
        self.ff.x.set(self.norm2.output)
        self.ff.forward()

        add3 = Input(self.norm2.output.shape)
        add3.set(self.norm2.output + self.ff.output)
        self.norm3.x = add3
        self.norm3.forward()

        self.output = self.norm3.output

    def backward(self):
        self.norm3.grad = self.grad
        self.norm3.backward()

        self.ff.grad = self.norm3.x.grad
        self.ff.backward()

        self.norm2.grad = self.ff.x.grad
        self.norm2.backward()

        self.cross_attn.grad = self.norm2.x.grad
        self.cross_attn.backward()

        self.norm1.grad = self.cross_attn.query.grad
        self.norm1.backward()

        self.self_attn.grad = self.norm1.x.grad
        self.self_attn.backward()

