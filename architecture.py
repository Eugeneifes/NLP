import torch
import torch.nn as nn

class TransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, feed_forward_dim, dropout_rate):
        super(TransformerLayer, self).__init__()
        
        # Multi-head self-attention layer
        self.attention = MultiHeadAttention(d_model, num_heads, dropout_rate)
        
        # Layer normalization for attention output
        self.attention_norm = nn.LayerNorm(d_model)
        
        # Feed-forward layer
        self.feed_forward = FeedForward(d_model, feed_forward_dim, dropout_rate)
        
        # Layer normalization for feed-forward output
        self.feed_forward_norm = nn.LayerNorm(d_model)
        
        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, inputs, attention_mask):
        # Perform self-attention
        attention_output = self.attention(inputs, attention_mask)
        
        # Apply residual connection and layer normalization to attention output
        attention_output = self.dropout(attention_output)
        attention_output = self.attention_norm(inputs + attention_output)
        
        # Apply feed-forward layer
        feed_forward_output = self.feed_forward(attention_output)
        
        # Apply residual connection and layer normalization to feed-forward output
        feed_forward_output = self.dropout(feed_forward_output)
        transformer_output = self.feed_forward_norm(attention_output + feed_forward_output)
        
        return transformer_output

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout_rate):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Linear layers for Q, K, and V projections
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        # Linear layer for final attention output
        self.output_linear = nn.Linear(d_model, d_model)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, inputs, attention_mask):
        batch_size, seq_len, d_model = inputs.size()
        
        # Project inputs to Q, K, and V
        q = self.q_linear(inputs)
        k = self.k_linear(inputs)
        v = self.v_linear(inputs)
        
        # Reshape Q, K, and V to multiple heads
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Calculate scaled dot-product attention
        scaled_attention_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
        # Apply attention mask
        scaled_attention_scores = scaled_attention_scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Apply softmax to obtain attention weights
        attention_weights = torch.softmax(scaled_attention_scores, dim=-1)
        
        # Apply dropout
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights to V
        attention_output = torch
