import logging

import torch
import torch.nn as nn

logger = logging.Logger("Model logger")


class ForgetGate(nn.Module):
    """
    Forget gate of an LSTM.
    Calculates F_t = sigmoid(X_t@W_xf + H_t-1@W_hf + b_f)
    """
    def __init__(self, input_dim, hidden_dim, sigma):
        super().__init__()
        self.W_xf = nn.Parameter(torch.randn(input_dim, hidden_dim)*sigma, requires_grad=True) # Will transform the input tensor X of size input_dim to hidden_dim.
        self.W_hf = nn.Parameter(torch.randn(hidden_dim, hidden_dim)*sigma, requires_grad=True) # Takes the previous hidden state as input, therefore input dim is hidden_dim and output dim is hidden_dim.
        self.b_f = nn.Parameter(torch.randn(1, hidden_dim)*sigma, requires_grad=True)

        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, previous_hidden_state):
        return self.sigmoid(torch.matmul(x, self.W_xf) + torch.matmul(previous_hidden_state, self.W_hf) + self.b_f)
    

class InputGate(nn.Module):
    """
    Input gate of an LSTM.
    Calculates I_t = sigmoid(X_t@W_xi + H_t-1@W_hi + b_i)
    """
    def __init__(self, input_dim, hidden_dim, sigma):
        super().__init__()
        self.W_xi = nn.Parameter(torch.randn(input_dim, hidden_dim)*sigma, requires_grad=True) # Will transform the input tensor X of size input_dim to hidden_dim.
        self.W_hi = nn.Parameter(torch.randn(hidden_dim, hidden_dim)*sigma, requires_grad=True) # Takes the previous hidden state as input, therefore input dim is hidden_dim and output dim is hidden_dim.
        self.b_i = nn.Parameter(torch.randn(1, hidden_dim)*sigma, requires_grad=True)

        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, previous_hidden_state):
        return self.sigmoid(torch.matmul(x, self.W_xi) + torch.matmul(previous_hidden_state, self.W_hi) + self.b_i)


class OutputGate(nn.Module):
    """
    Output gate of an LSTM.
    Calculates O_t = sigmoid(X_t@W_xo + H_t-1@W_ho + b_o)
    """
    def __init__(self, input_dim, hidden_dim, sigma):
        super().__init__()
        self.W_xo = nn.Parameter(torch.randn(input_dim, hidden_dim)*sigma, requires_grad=True) # Will transform the input tensor X of size input_dim to hidden_dim.
        self.W_ho = nn.Parameter(torch.randn(hidden_dim, hidden_dim)*sigma, requires_grad=True) # Takes the previous hidden state as input, therefore input dim is hidden_dim and output dim is hidden_dim.
        self.b_o = nn.Parameter(torch.randn(1, hidden_dim)*sigma, requires_grad=True)

        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, previous_hidden_state):
        return self.sigmoid(torch.matmul(x, self.W_xo) + torch.matmul(previous_hidden_state, self.W_ho) + self.b_o)
    

class InputNode(nn.Module):
    """
    Input node of an LSTM.
    Calculates ~C = tanh(X_t@W_xc + H_t-1@W_hc + b_c)
    """
    def __init__(self, input_dim, hidden_dim, sigma):
        super().__init__()
        self.W_xc = nn.Parameter(torch.randn(input_dim, hidden_dim)*sigma, requires_grad=True)
        self.W_hc = nn.Parameter(torch.randn(hidden_dim, hidden_dim)*sigma, requires_grad=True)
        self.b_c = nn.Parameter(torch.randn(1, hidden_dim)*sigma, requires_grad=True)

        self.tanh = nn.Tanh()
    
    def forward(self, x, previous_hidden_state):
        return self.tanh(torch.matmul(x, self.W_xc) + torch.matmul(previous_hidden_state, self.W_hc) + self.b_c)
    

class LSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, sigma=0.01):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.input_gate = InputGate(input_dim, hidden_dim, sigma)
        self.output_gate = OutputGate(input_dim, hidden_dim, sigma)
        self.input_node = InputNode(input_dim, hidden_dim, sigma)
        self.forget_gate = ForgetGate(input_dim, hidden_dim, sigma)

        self.tanh = nn.Tanh()

    def forward(self, x, previous_hidden_state, previous_memory_state):
        forget_gate_output = self.forget_gate(x, previous_hidden_state)
        output_gate_output = self.output_gate(x ,previous_hidden_state)
        input_gate_output = self.input_gate(x, previous_hidden_state)
        input_node_output = self.input_node(x, previous_hidden_state)

        memory_cell_internal_state = forget_gate_output*previous_memory_state + input_gate_output*input_node_output # C_t = F_t*C_t-1 + I_t*~C_t
        hidden_state = output_gate_output*self.tanh(memory_cell_internal_state)

        return hidden_state, memory_cell_internal_state