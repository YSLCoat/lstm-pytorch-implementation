import torch
import torch.nn as nn

from lstm import LSTMCell


class SingleLayerLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm_cell = LSTMCell(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        hidden_state = torch.zeros((x.shape[0], self.hidden_dim), device=x.device)
        memory_state = torch.zeros((x.shape[0], self.hidden_dim), device=x.device)

        for t in range(x.shape[1]):
            x_t = x[:, t, :]
            hidden_state, memory_state = self.lstm_cell(x_t, hidden_state, memory_state)

        output = self.output_layer(hidden_state)
        return output
    

class MultiLayerLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.lstm_layers = nn.ModuleList()
        self.lstm_layers.append(LSTMCell(input_dim, hidden_dim))

        for _ in range(num_layers-1):
            self.lstm_layers.append(LSTMCell(hidden_dim, hidden_dim))

        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        hidden_states = [torch.zeros((x.shape[0], self.hidden_dim), device=x.device) for _ in range(self.num_layers)]
        memory_states = [torch.zeros((x.shape[0], self.hidden_dim), device=x.device) for _ in range(self.num_layers)]

        for t in range(x.shape[1]):
            x_t = x[:, t, :]

            for layer in range(self.num_layers):
                lstm_layer = self.lstm_layers[layer]
                previous_hidden_state = hidden_states[layer]
                previous_memory_state = memory_states[layer]

                hidden_state, memory_state = lstm_layer(x_t, previous_hidden_state, previous_memory_state)
                hidden_states[layer] = hidden_state
                memory_states[layer] = memory_state

                x_t = hidden_state 

        output = self.output_layer(hidden_states[-1])
        return output