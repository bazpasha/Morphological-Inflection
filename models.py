import torch
import torch.nn as nn
import torch.nn.functional as F


class WordEncoder(nn.Module):
    def __init__(self, n_tokens, embedding_dim, hidden_size):
        super(WordEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(num_embeddings=n_tokens, embedding_dim=embedding_dim)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
        )
    
    def forward(self, words):
        embedded = self.embedding(words)
        output, hidden = self.gru(embedded, hidden_start)
        return output, hidden.squeeze()
    

class TagsEncoder(nn.Module):
    def __init__(self, n_tags, hidden_size, output_size):
        super(TagsEncoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_tags, hidden_size),
            nn.LeakyReLU(0.05),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.05),
            nn.Linear(hidden_size, output_size),
            nn.LeakyReLU(0.05),
        )
    
    def forward(self, tags):
        return self.model(tags)


class SimpleDecoder(nn.Module):
    def __init__(self, n_tokens, embedding_dim, hidden_size):
        super(SimpleDecoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=n_tokens, embedding_dim=embedding_dim)
        self.cell = nn.GRUCell(input_size=embedding_dim, hidden_size=hidden_size)
        self.linear = nn.Linear(hidden_size, n_tokens)
    
    def forward(self, x, hidden):
        embedded = self.embedding(x)
        hidden = self.cell(embedded, hidden)
        output = self.linear(hidden)
        return output, hidden


class AttentionDecoder(nn.Module):
    def __init__(self, n_tokens, embedding_dim, hidden_size, encoder_hidden_size, values_size):
        super(AttentionDecoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=n_tokens, embedding_dim=embedding_dim)
        self.cell = nn.GRUCell(input_size=embedding_dim, hidden_size=hidden_size)
        self.linear = nn.Linear(hidden_size + encoder_hidden_size, n_tokens)
        
        self.encoder_attention_matrix = nn.Linear(encoder_hidden_size, values_size)
        self.hidden_attention_matrix = nn.Linear(hidden_size, values_size)
        self.attention_bias = nn.Parameter(torch.rand(values_size))
        self.values = nn.Parameter(torch.rand(values_size))
    
    def forward(self, x, hidden, encoder_output):
        embedded = self.embedding(x)
        new_hidden = self.cell(embedded, hidden)
        before_tanh = (
            self.encoder_attention_matrix(encoder_output) +
            self.hidden_attention_matrix(hidden).unsqueeze(1) +
            self.attention_bias.view(1, 1, -1)
        )
        attention_logit = (
            self.values.view(1, 1, -1) * F.tanh(before_tanh)
        ).sum(-1)
        attention_weights = F.softmax(attention_logit, -1)
        context_vector = (encoder_output * attention_weights.unsqueeze(2)).sum(1)
        
        output = self.linear(torch.cat((new_hidden, context_vector), -1))
        
        return output, new_hidden


class PointerDecoder(nn.Module):
    def __init__(self, n_tokens, embedding_dim, hidden_size, encoder_hidden_size, values_size, cuda=False):
        super(AttentionDecoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=n_tokens, embedding_dim=embedding_dim)
        self.cell = nn.GRUCell(input_size=embedding_dim, hidden_size=hidden_size)
        self.linear = nn.Linear(hidden_size + encoder_hidden_size, n_tokens)
        
        self.encoder_attention_matrix = nn.Linear(encoder_hidden_size, values_size)
        self.hidden_attention_matrix = nn.Linear(hidden_size, values_size)
        self.attention_bias = nn.Parameter(torch.rand(values_size))
        self.values = nn.Parameter(torch.rand(values_size))

        self.p_gen_context = nn.Parameter(torch.rand(encoder_hidden_size))
        self.p_gen_hidden = nn.Parameter(torch.rand(hidden_size))
        self.p_gen_input = nn.Parameter(torch.rand(embedding_dim))
        self.p_gen_bias = nn.Parameter(torch.rand(1))

        self.use_cuda = cuda
    
    def forward(self, x, hidden, encoder_output, lemmas):
        embedded = self.embedding(x)
        new_hidden = self.cell(embedded, hidden)
        before_tanh = (
            self.encoder_attention_matrix(encoder_output) +
            self.hidden_attention_matrix(hidden).unsqueeze(1) +
            self.attention_bias.view(1, 1, -1)
        )
        attention_logit = (
            self.values.view(1, 1, -1) * F.tanh(before_tanh)
        ).sum(-1)
        attention_weights = F.softmax(attention_logit, -1)
        context_vector = (encoder_output * attention_weights.unsqueeze(2)).sum(1)
        
        output = self.linear(torch.cat((new_hidden, context_vector), -1))
        output = F.softmax(output, -1)

        p_gen_logits = (
            (self.p_gen_context * context_vector).sum(-1) +
            (self.p_gen_hidden * new_hidden).sum(-1) +
            (self.p_gen_input * embedded).sum(-1) +
            self.p_gen_bias
        )
        p_gen = nn.Sigmoid()(p_gen_logits)
        
        p_gen_matrix = torch.zeros(batch_size, self.n_tokens)
        if self.use_cuda:
            p_gen_matrix = p_gen_matrix.cuda()
        p_gen_matrix.scatter_add_(1, lemmas, attention_weights)

        probas = p_gen.unsqueeze(1) * output + (1 - p_gen).unsqueeze(1) * p_gen_matrix
        probas = torch.clamp(probas, min=1e-8)
        
        return probas, new_hidden
