import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np

class PositionalEncoder(nn.Module):
    def __init__(self, embedding_size, max_seq_len):
        super(PositionalEncoder, self).__init__()
        self.embedding_size = embedding_size
        
        pos_encoder = torch.zeros(max_seq_len, embedding_size)
        for pos in range(max_seq_len):
            for i in range(0, embedding_size, 2):
                pos_encoder[pos, i] = math.sin(pos / (10000 ** ((2 * i)/embedding_size)))
                pos_encoder[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/embedding_size)))
                
        pos_encoder = pos_encoder.unsqueeze(0)
        self.register_buffer('pos_encoder', pos_encoder)
 
    def forward(self, x):
        x = x * math.sqrt(self.embedding_size)
        seq_len = x.size(1)
        x = x + self.pos_encoder[:, :seq_len]
        return x
    
class Attention(nn.Module):
    def __init__(self, layer_size, heads = 6, dropout=0.1):
        super(Attention, self).__init__()
        self.layer_size = layer_size
        self.heads = heads
        self.head_size = layer_size // heads
        
        self.get_query = nn.Linear(layer_size, layer_size, bias=False)
        self.get_key = nn.Linear(layer_size, layer_size, bias=False)
        self.get_value = nn.Linear(layer_size, layer_size, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.combine_final = nn.Linear(layer_size, layer_size)
        
    def attention(self, query, key, value, mask):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_size)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        
        return torch.matmul(scores, value)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
                
        query = self.get_query(query).view(batch_size, -1, self.heads, self.head_size).transpose(1,2)
        key = self.get_key(key).view(batch_size, -1, self.heads, self.head_size).transpose(1,2)
        value = self.get_value(value).view(batch_size, -1, self.heads, self.head_size).transpose(1,2)
        
        scores = self.attention(query, key, value, mask)
        concat = scores.transpose(1,2).contiguous().view(batch_size, -1, self.layer_size)
        return self.combine_final(concat)
    
class FeedForward(nn.Module):
    def __init__(self, input_size, hidden_size=2048, dropout=0.1):
        super(FeedForward, self).__init__()
        
        self.linear_1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(hidden_size, input_size)
        
    def forward(self, x):
        return self.linear_2(
            self.dropout(
                F.relu(
                    self.linear_1(x)
                )
            )
        )

class Normalize(nn.Module):
    def __init__(self, layer_size, eps = 1e-6):
        super(Normalize, self).__init__()
    
        self.size = layer_size
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
        
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm
    
class EncoderLayer(nn.Module):
    def __init__(self, layer_size, heads, dropout=0.1):
        super(EncoderLayer, self).__init__()
        
        self.attention = Attention(layer_size, heads)
        self.normalization_1 = Normalize(layer_size)
        self.normalization_2 = Normalize(layer_size)
        self.feed_forward = FeedForward(layer_size)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        x_temp = self.normalization_1(x)
        x = x + self.dropout_1(self.attention(x_temp, x_temp, x_temp, mask))
        x_temp = self.normalization_2(x)
        x = x + self.dropout_2(self.feed_forward(x_temp))
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, layer_size, heads, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.normalization_1 = Normalize(layer_size)
        self.normalization_2 = Normalize(layer_size)
        self.normalization_3 = Normalize(layer_size)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        
        self.attention_1 = Attention(layer_size, heads)
        self.attention_2 = Attention(layer_size, heads)
        self.feed_forward = FeedForward(layer_size)
        
    def forward(self, x, encoded, source_mask, target_mask):
        x_temp = self.normalization_1(x)
        x = x + self.dropout_1(self.attention_1(x_temp, x_temp, x_temp, target_mask))
        x_temp = self.normalization_2(x)
        x = x + self.dropout_2(self.attention_2(x_temp, encoded, encoded, source_mask))
        x_temp = self.normalization_3(x)
        x = x + self.dropout_3(self.feed_forward(x_temp))
        return x
    
class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, max_seq_len, blocks, heads):
        super(Encoder, self).__init__()
        
        self.blocks = blocks
        self.embedder = nn.Embedding(vocab_size, embedding_size)
        self.positional_encoder = PositionalEncoder(embedding_size, max_seq_len)
        self.encoder_layers = nn.ModuleList()
        for i in range(blocks):
            self.encoder_layers.append(EncoderLayer(embedding_size, heads))
        self.normalization = Normalize(embedding_size)
        
    def forward(self, x, mask):
        x = self.embedder(x)
        x = self.positional_encoder(x)
        for i in range(self.blocks):
            x = self.encoder_layers[i](x, mask)
        return self.normalization(x)
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, max_seq_len, blocks, heads):
        super(Decoder, self).__init__()
        
        self.blocks = blocks
        self.embedder = nn.Embedding(vocab_size, embedding_size)
        self.positional_encoder = PositionalEncoder(embedding_size, max_seq_len)
        self.decoder_layers = nn.ModuleList()
        for i in range(blocks):
            self.decoder_layers.append(DecoderLayer(embedding_size, heads))
        self.normalization = Normalize(embedding_size)
        
    def forward(self, x, encoded, source_mask, target_mask):
        x = self.embedder(x)
        x = self.positional_encoder(x)
        for i in range(self.blocks):
            x = self.decoder_layers[i](x, encoded, source_mask, target_mask)
        return self.normalization(x)
    
class Transformer(nn.Module):
    def __init__(self, vocab_size, embedding_size, max_seq_len, blocks, heads):
        super(Transformer, self).__init__()
        self.encoder = Encoder(vocab_size, embedding_size, max_seq_len, blocks, heads)
        self.decoder = Decoder(vocab_size, embedding_size, max_seq_len, blocks, heads)
        self.out = nn.Linear(embedding_size, vocab_size)
        
    def forward(self, src, trg, source_mask, target_mask):
        encoded = self.encoder(src, source_mask)
        decoded = self.decoder(trg, encoded, source_mask, target_mask)
        output = self.out(decoded)
        return output
    
def gen_input_mask(x, padding_token=0):
    input_msk = (x != padding_token).unsqueeze(1)
    return input_msk

def gen_target_mask(y, padding_token=0):
    target_msk = (y != padding_token).unsqueeze(1)
    target_size = y.size(1)
    
    nopeak_mask = np.triu(np.ones((1, target_size, target_size)), k=1).astype('uint8')
    nopeak_mask = torch.from_numpy(nopeak_mask) == 0

    return target_msk & nopeak_mask