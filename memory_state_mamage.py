# We are import the require library
import torch
import torch.nn as nn
import torch.nn.functional as F

# We are define an class name Attention
class Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Attention,self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.W = nn.Linear(input_dim, hidden_dim)
        self.U = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim,1)
        
    
    def forward(self, inputs):
        energy = torch.tanh(self.W(inputs))
        attention_scores = self.v(energy).squeeze(2)
        attention_weights = F.softmax(attention_scores, dim=1)
        context_vector = torch.bmm(attention_weights.unsqueeze(1),inputs).squeeze(1)
        return context_vector, attention_weights
    

# We are define the class with LLMWithMemory
class LLMWithMemory(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LLMWithMemory, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.attention = Attention(input_dim, hidden_dim)
        self.gru = nn.GRUCell(input_dim, hidden_dim)
        
    def forward(self, inputs, memory_state):
        context_vector, attention_weights = self.attention(inputs.unsqueeze(1))
        updated_memory_state = self.gru(inputs, memory_state)
        return updated_memory_state, context_vector, attention_weights
    
    

# Some example usage
input_dim = 300
hidden_dim = 512


# We are initialize LLM with memory
llm_with_memory = LLMWithMemory(input_dim, hidden_dim)

# We are define an example with forward pass
batch_size = 32
seq_len = 10
inputs = torch.randn(batch_size, input_dim)
initial_memory_state = torch.zeros(batch_size, hidden_dim)
updated_memory_state, context_vector, attention_weights=llm_with_memory(inputs, initial_memory_state)


# We are print the shapes of outputs
print("Updated Memory State:", updated_memory_state.shape)
print("Context Vector:", context_vector.shape)
print("Attention Weights:", attention_weights.shape)
