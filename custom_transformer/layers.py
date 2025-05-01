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
        
        # Add gradient clipping to prevent NaN
        if self.grad is not None:
            torch.nn.utils.clip_grad_norm_(self.grad.reshape(-1), max_norm=1.0)

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
        # Initialize with scaled values to improve convergence
        if len(value.shape) > 1:
            # Xavier initialization for weight matrices
            std = math.sqrt(2.0 / sum(value.shape))
            self.output = torch.randn_like(value) * std
        else:
            # Initialize biases to zero
            self.output = torch.zeros_like(value)
    
    def step(self, alpha=0.01):
        if self.train and self.grad is not None:
            # Apply gradient clipping before updating
            torch.nn.utils.clip_grad_norm_(self.grad.reshape(-1), max_norm=1.0)
            self.output -= alpha * self.grad

    def accumulate_grad(self, grad):
        if self.grad is None:
            self.grad = grad.clone()
        else:
            self.grad += grad
        
        # Add gradient clipping
        if self.grad is not None:
            torch.nn.utils.clip_grad_norm_(self.grad.reshape(-1), max_norm=1.0)


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
        # Add small epsilon to prevent exactly zero loss
        self.diff = self.prediction_layer.output - self.target_layer.output
        self.output = torch.mean(self.diff ** 2) + 1e-8

    def backward(self):
        batch_size = self.prediction_layer.output.shape[0]
        # Scale by batch size for better stability
        grad = 2 * self.diff / batch_size
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
        # Numerical stability: Subtract max value
        x = self.input_layer.output
        x_max, _ = torch.max(x, dim=-1, keepdim=True)
        exp_vals = torch.exp(x - x_max)
        self.output = exp_vals / torch.sum(exp_vals, dim=-1, keepdim=True)

    def backward(self):
        if self.grad is None:
            return
            
        softmax = self.output
        # Efficient Jacobian-vector product for softmax backward
        grad_input = self.grad - (self.grad * softmax).sum(dim=-1, keepdim=True) * softmax
        self.input_layer.accumulate_grad(grad_input)

class CrossEntropyLoss(Layer):
    def __init__(self, input_layer, target_layer):
        super().__init__()
        self.input_layer = input_layer  # Output from softmax
        self.target_layer = target_layer  # One-hot encoded target

    def forward(self):
        # Add epsilon for numerical stability
        epsilon = 1e-8
        self.output = -torch.sum(self.target_layer.output * 
                                torch.log(self.input_layer.output + epsilon), 
                                dim=-1).mean()

    def backward(self):
        batch_size = self.target_layer.output.shape[0]
        epsilon = 1e-8
        # Scale by batch size for stability
        grad = -(self.target_layer.output / (self.input_layer.output + epsilon)) / batch_size
        self.input_layer.accumulate_grad(self.grad * grad)

class SoftmaxCrossEntropy(Layer):
    def __init__(self, input_layer, target_layer=None):
        Layer.__init__(self)
        self.input_layer = input_layer
        self.target_layer = target_layer
        self.classifications = None

    def set_targets(self, targets):
        self.targets = targets

    def forward(self):
        # Numerical stability: Subtract max value
        x = self.input_layer.output
        x_max, _ = torch.max(x, dim=-1, keepdim=True)
        exp_vals = torch.exp(x - x_max)
        self.classifications = exp_vals / torch.sum(exp_vals, dim=-1, keepdim=True)
        
        if self.target_layer is not None:
            self.targets = self.target_layer.output
            
        # Add epsilon for numerical stability
        epsilon = 1e-8
        self.output = -torch.sum(self.targets * torch.log(self.classifications + epsilon), dim=-1).mean()

    def backward(self):
        batch_size = self.targets.shape[0]
        # Scale by batch size for better stability
        grad = self.grad * (self.classifications - self.targets) / batch_size
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
    This performs RMS Normalization with numerical stability improvements
    """
    def __init__(self, x, gamma, epsilon=1e-5):
        Layer.__init__(self)    
        # Either the dimensions of g should match the last dimensions of x.
        assert gamma.output.shape[-1] == x.output.shape[-1], "gamma and x must match in the last dimension only"
        self.x = x
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self):
        # Calculate RMS with epsilon for numerical stability
        self.rms = torch.sqrt(torch.mean(self.x.output**2, dim=-1, keepdim=True) + self.epsilon)
        self.output = self.x.output / self.rms * self.gamma.output

    def backward(self):
        if self.grad is None:
            return
            
        # More stable backward computation
        n = self.x.output.shape[-1]
        x_normalized = self.x.output / self.rms
        
        # Gradient for gamma
        dJdgamma = (x_normalized * self.grad).sum(axis=tuple(range(len(self.x.output.shape)-1)), keepdim=True)
        
        # Gradient for input
        dx_normalized = self.grad * self.gamma.output
        drms = -torch.sum(dx_normalized * self.x.output, dim=-1, keepdim=True) / (self.rms * n)
        dx = dx_normalized / self.rms + drms * self.x.output / self.rms
        
        self.gamma.accumulate_grad(dJdgamma)
        self.x.accumulate_grad(dx)

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
        self.attention_weights = None  # Store for backward pass

    def split_heads(self, x):
        B, T, C = x.shape
        return x.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        B, H, T, D = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * D)

    def forward(self):
        # Linear projections
        self.q_linear = Linear(self.query, self.W_q, Parameter(torch.zeros(self.d_model).to(DEVICE)))
        self.k_linear = Linear(self.key, self.W_k, Parameter(torch.zeros(self.d_model).to(DEVICE)))
        self.v_linear = Linear(self.value, self.W_v, Parameter(torch.zeros(self.d_model).to(DEVICE)))
        
        self.q_linear.forward()
        self.k_linear.forward()
        self.v_linear.forward()

        self.Q = self.split_heads(self.q_linear.output)
        self.K = self.split_heads(self.k_linear.output)
        self.V = self.split_heads(self.v_linear.output)

        # Scaled Dot-Product with improved numerical stability
        self.scores = self.Q @ self.K.transpose(-2, -1) / math.sqrt(self.d_k)
        
        if self.mask is not None:
            self.scores = self.scores.masked_fill(self.mask == 0, -1e9)

        # Softmax with numerical stability
        self.scores_max, _ = torch.max(self.scores, dim=-1, keepdim=True)
        self.exp_scores = torch.exp(self.scores - self.scores_max)
        self.softmax_denom = torch.sum(self.exp_scores, dim=-1, keepdim=True)
        self.attention_weights = self.exp_scores / self.softmax_denom

        self.attn_output = self.attention_weights @ self.V
        self.combined = self.combine_heads(self.attn_output)

        # Output projection
        self.o_input = Input(self.combined.shape)
        self.o_input.set(self.combined)
        self.o_proj = Linear(self.o_input, self.W_o, Parameter(torch.zeros(self.d_model).to(DEVICE)))
        self.o_proj.forward()

        self.output = self.o_proj.output

    def backward(self):
        if self.grad is None:
            return
            
        # Backward through output projection
        self.o_proj.grad = self.grad
        self.o_proj.backward()
        
        # Backward through attention mechanism
        d_combined = self.o_input.grad
        
        # Reshape for multi-head format
        d_attn_output = d_combined.view(d_combined.size(0), -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Gradient through attention
        d_V = self.attention_weights.transpose(-2, -1) @ d_attn_output
        d_attn_weights = d_attn_output @ self.V.transpose(-2, -1)
        
        # Gradient through softmax
        d_scores = d_attn_weights * self.attention_weights
        d_scores_sum = torch.sum(d_scores, dim=-1, keepdim=True)
        d_scores = d_scores - self.attention_weights * d_scores_sum
        
        # Scale back
        d_scores = d_scores / math.sqrt(self.d_k)
        
        # Gradients for Q, K, V
        d_Q = d_scores @ self.K
        d_K = d_scores.transpose(-2, -1) @ self.Q
        
        # Reshape back
        d_q = d_Q.transpose(1, 2).contiguous().view(d_Q.size(0), -1, self.d_model)
        d_k = d_K.transpose(1, 2).contiguous().view(d_K.size(0), -1, self.d_model)
        d_v = d_V.transpose(1, 2).contiguous().view(d_V.size(0), -1, self.d_model)
        
        # Set gradients for linear layers
        self.q_linear.grad = d_q
        self.k_linear.grad = d_k
        self.v_linear.grad = d_v
        
        # Backward through linear layers
        self.q_linear.backward()
        self.k_linear.backward()
        self.v_linear.backward()

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
        if self.grad is None:
            return
            
        self.linear2.grad = self.grad
        self.linear2.backward()
        self.relu.backward()
        self.linear1.backward()

class PositionalEncoding(Layer):
    def __init__(self, max_seq_len, d_model):
        super().__init__()
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model, device=DEVICE)
        position = torch.arange(0, max_seq_len, device=DEVICE).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2, device=DEVICE).float() * -(math.log(10000.0) / d_model))
        
        # Calculate sin and cos components
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension
        self.pe = pe.unsqueeze(0)  # Shape: (1, max_seq_len, d_model)
        self.max_seq_len = max_seq_len
        self.d_model = d_model

    def add_to(self, x_layer):
        # Ensure we don't go out of bounds
        seq_len = min(self.max_seq_len, x_layer.output.shape[1])
        self.output = x_layer.output + self.pe[:, :seq_len]
        return self.output
    
class EncoderLayer(Layer):
    def __init__(self, x, d_model, num_heads, d_ff, W_q, W_k, W_v, W_o, W1, b1, W2, b2, norm1_weight, norm2_weight):
        super().__init__()
        self.x_input = x
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Self-attention components
        self.attn = MultiHeadAttention(x, x, x, d_model, num_heads, W_q, W_k, W_v, W_o)
        
        # Feed-forward components
        self.ff = PositionWiseFeedForward(Input(x.output.shape), W1, b1, W2, b2)
        
        # Normalization layers
        self.norm1 = RMSNorm(Input(x.output.shape), norm1_weight)
        self.norm2 = RMSNorm(Input(x.output.shape), norm2_weight)

    def forward(self):
        # Store input for residual connection
        x_copy = self.x_input.output.clone()
        
        # Self-attention with pre-normalization
        self.norm1.x.set(x_copy)
        self.norm1.forward()
        
        self.attn.query = Input(self.norm1.output.shape)
        self.attn.query.set(self.norm1.output)
        self.attn.key = self.attn.query
        self.attn.value = self.attn.query
        self.attn.forward()
        
        # First residual connection
        attn_output = x_copy + self.attn.output
        
        # Feed-forward with pre-normalization
        self.norm2.x.set(attn_output)
        self.norm2.forward()
        
        self.ff.x.set(self.norm2.output)
        self.ff.forward()
        
        # Second residual connection
        self.output = attn_output + self.ff.output

    def backward(self):
        if self.grad is None:
            return
            
        # Split gradient for residual connection
        d_ff_output = self.grad
        d_attn_output = self.grad
        
        # Backward through feed-forward path
        self.ff.grad = d_ff_output
        self.ff.backward()
        
        # Gradient to normalization
        d_norm2_input = self.ff.x.grad
        self.norm2.grad = d_norm2_input
        self.norm2.backward()
        
        # Add gradient from second residual connection
        d_attn_output_total = d_attn_output + self.norm2.x.grad
        
        # Backward through attention path
        self.attn.grad = d_attn_output_total
        self.attn.backward()
        
        # Gradient to first normalization
        d_norm1_input = self.attn.query.grad
        self.norm1.grad = d_norm1_input
        self.norm1.backward()
        
        # Add gradient from first residual connection
        d_x = self.norm1.x.grad
        
        # Accumulate gradient to input
        self.x_input.accumulate_grad(d_x)

class DecoderLayer(Layer):
    def __init__(self, x, enc_output, d_model, num_heads, d_ff, 
                 W_q_self, W_k_self, W_v_self, W_o_self,
                 W_q_cross, W_k_cross, W_v_cross, W_o_cross,
                 W1, b1, W2, b2,
                 norm1_weight, norm2_weight, norm3_weight):
        super().__init__()
        self.x_input = x
        self.enc_output = enc_output
        
        # Self-attention
        self.self_attn = MultiHeadAttention(x, x, x, d_model, num_heads, 
                                          W_q_self, W_k_self, W_v_self, W_o_self)
        
        # Cross-attention
        self.cross_attn = MultiHeadAttention(x, enc_output, enc_output, d_model, num_heads,
                                           W_q_cross, W_k_cross, W_v_cross, W_o_cross)
        
        # Feed-forward
        self.ff = PositionWiseFeedForward(Input(x.output.shape), W1, b1, W2, b2)

        # Layer normalization
        self.norm1 = RMSNorm(Input(x.output.shape), norm1_weight)
        self.norm2 = RMSNorm(Input(x.output.shape), norm2_weight)
        self.norm3 = RMSNorm(Input(x.output.shape), norm3_weight)

    def forward(self):
        # Store input for residual connections
        x_copy = self.x_input.output.clone()
        
        # Self-attention with pre-normalization
        self.norm1.x.set(x_copy)
        self.norm1.forward()
        
        self.self_attn.query = Input(self.norm1.output.shape)
        self.self_attn.query.set(self.norm1.output)
        self.self_attn.key = self.self_attn.query
        self.self_attn.value = self.self_attn.query
        self.self_attn.forward()
        
        # First residual connection
        attn_self_output = x_copy + self.self_attn.output
        
        # Cross-attention with pre-normalization
        self.norm2.x.set(attn_self_output)
        self.norm2.forward()
        
        self.cross_attn.query = Input(self.norm2.output.shape)
        self.cross_attn.query.set(self.norm2.output)
        self.cross_attn.key = Input(self.enc_output.output.shape)
        self.cross_attn.key.set(self.enc_output.output)
        self.cross_attn.value = self.cross_attn.key
        self.cross_attn.forward()
        
        # Second residual connection
        attn_cross_output = attn_self_output + self.cross_attn.output
        
        # Feed-forward with pre-normalization
        self.norm3.x.set(attn_cross_output)
        self.norm3.forward()
        
        self.ff.x.set(self.norm3.output)
        self.ff.forward()
        
        # Third residual connection
        self.output = attn_cross_output + self.ff.output

    def backward(self):
        if self.grad is None:
            return
            
        # Split gradient for residual connections
        d_ff_output = self.grad
        d_cross_attn_output = self.grad
        d_self_attn_output = self.grad
        
        # Backward through feed-forward path
        self.ff.grad = d_ff_output
        self.ff.backward()
        
        # Gradient to third normalization
        d_norm3_input = self.ff.x.grad
        self.norm3.grad = d_norm3_input
        self.norm3.backward()
        
        # Add gradient from third residual connection
        d_cross_attn_output_total = d_cross_attn_output + self.norm3.x.grad
        
        # Backward through cross-attention path
        self.cross_attn.grad = d_cross_attn_output_total
        self.cross_attn.backward()
        
        # Gradient to second normalization
        d_norm2_input = self.cross_attn.query.grad
        self.norm2.grad = d_norm2_input
        self.norm2.backward()
        
        # Add gradient from second residual connection
        d_self_attn_output_total = d_self_attn_output + self.norm2.x.grad
        
        # Backward through self-attention path
        self.self_attn.grad = d_self_attn_output_total
        self.self_attn.backward()
        
        # Gradient to first normalization
        d_norm1_input = self.self_attn.query.grad
        self.norm1.grad = d_norm1_input
        self.norm1.backward()
        
        # Add gradient from first residual connection
        d_x = self.norm1.x.grad
        
        # Accumulate gradient to input
        self.x_input.accumulate_grad(d_x)