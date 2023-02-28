from transformers import logging
logging.set_verbosity_error()
import warnings
warnings.filterwarnings("ignore") 
import torch
from torch import nn
from torch.nn import functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=768, n_heads=12):
        """
        Args:
            embed_dim: dimension of embeding vector output
            n_heads: number of self attention heads
        """
        super(MultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim    #768 dim
        self.n_heads = n_heads   #12
        self.single_head_dim = int(self.embed_dim / self.n_heads)   #768/12 = 64  . each key,query, value will be of 768d
       
        #key,query and value matrixes    #64 x 64   
        self.query_matrix = nn.Linear(self.embed_dim , self.embed_dim ,bias=False)  # single key matrix for all 12 keys #768x768
        self.key_matrix = nn.Linear(self.embed_dim  , self.embed_dim, bias=False)
        self.value_matrix = nn.Linear(self.embed_dim ,self.embed_dim , bias=False)
        self.out = nn.Linear(self.embed_dim ,self.embed_dim) 

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.n_heads, self.single_head_dim).transpose(1, 2).contiguous()

    def forward(self,hidden_states,attention_mask=None, early_div = False):    #batch_size x sequence_length x embedding_dim    # 4 x 858 x 768
        
        """
        Args:
           attention_mask: attention_mask for decoder
        
        Returns:
           output vector from multihead attention
        """
        squroot = math.sqrt(self.single_head_dim )

        bsz, tgt_len, _ = hidden_states.size()
        query_states = self.query_matrix(hidden_states)
        if early_div:
            query_states /= squroot
            # print(f"We do early division \n" , flush = True)
        key_states = self._shape(self.key_matrix(hidden_states), -1, bsz)
        value_states = self._shape(self.value_matrix(hidden_states), -1, bsz)

        proj_shape = (bsz * self.n_heads, -1, self.single_head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        del query_states
        del key_states

        if not early_div:
            attn_weights /= squroot # / sqrt(64)
            # print(f"We do late division \n" , flush = True)

        
        # fill those positions of attn_weights matrix as (-1e20) where attention_mask positions are 0
        if attention_mask is not None:
            attention_mask = attention_mask.expand(-1 , -1 , attention_mask.shape[-1] , -1) # (batch_size , 1 , 1 , seq_length) ->(batch_size , 1 , seq_length , seq_length) 
            attn_weights = attn_weights.view(bsz, self.n_heads, tgt_len, src_len) + attention_mask
            # print(f"attn_weights shape after adding mask is {attn_weights.shape} \n" , flush = True)
            attn_weights = attn_weights.view(bsz * self.n_heads, tgt_len, src_len)
        
        del attention_mask


        #applying softmax
        scores = F.softmax(attn_weights, dim=-1)
 
        #mutiply with value matrix
        scores = torch.matmul(scores, value_states)  ##(4x12x 858x 858) x (4 x 12 x 858 x 64) = (4 x 12 x 858 x 64) 

        del value_states
        
        #concatenated output
        concat = scores.transpose(1,2).contiguous().view(bsz, src_len, self.single_head_dim*self.n_heads)  # (4x12x858x64) -> (4x858x12x64)  -> (4,858,768)
        
        output = self.out(concat) #(4,858,768) -> (4,858,768)
       
        return output


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, n_heads=12 , dropout = 0.2):
        super(TransformerBlock, self).__init__()
        
        """
        Args:
           embed_dim: dimension of the embedding
           expansion_factor: factor which determines output dimension of linear layer
           n_heads: number of attention heads
        
        """
        self.dropout = dropout
        self.attention = MultiHeadAttention(embed_dim, n_heads)
        
        self.dropout1 = nn.Dropout(self.dropout)
        self.norm1 = nn.LayerNorm(embed_dim) # in videoMAE its a nn.Layer()

        self.feed_forward = nn.Sequential(
                          nn.Dropout(self.dropout),
                          nn.Linear(embed_dim, expansion_factor*embed_dim),
                          nn.GELU(),
                          nn.Linear(expansion_factor*embed_dim, embed_dim),
        )

        self.dropout2 = nn.Dropout(self.dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self,hidden_states , attention_mask = None , early_div = False):
        
        """
        Args:
           hidden_states: the hidden states input, which is key,query,value
           norm2_out: output of transformer block
        
        """
        
        
        attention_residual_out = hidden_states +  self.attention(hidden_states, attention_mask , early_div) #32x10x512
        del attention_mask
        
        norm1_out = self.norm1(self.dropout1(attention_residual_out)) #32x10x512

        feed_fwd_out = self.feed_forward(norm1_out) #32x10x512 -> #32x10x2048 -> 32x10x512
        feed_fwd_residual_out = feed_fwd_out + norm1_out #32x10x512
        del feed_fwd_out
        del norm1_out
        norm2_out = self.norm2(self.dropout2(feed_fwd_residual_out)) #32x10x512
        

        return norm2_out

class TransformerEncoder(nn.Module):
    """
    Args:
        seq_len : length of input sequence
        embed_dim: dimension of embedding
        num_layers: number of encoder layers
        expansion_factor: factor which determines number of linear layers in feed forward layer
        n_heads: number of heads in multihead attention
        
    Returns:
        out: output of the encoder
    """
    def __init__(self, embed_dim, num_layers=2, expansion_factor=4, n_heads=12 , dropout = 0.2, early_div = False):
        super(TransformerEncoder, self).__init__()
        
        self.early_div = early_div
        self.layers = nn.ModuleList([TransformerBlock(embed_dim, expansion_factor, n_heads , dropout) for i in range(num_layers)])
    
    def forward(self, x , attention_mask = None):
        for layer in self.layers:
            x = layer(x, attention_mask , self.early_div)
            torch.cuda.empty_cache()
        return x  #32x10x768