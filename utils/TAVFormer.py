from transformers import logging
logging.set_verbosity_error()
import warnings
warnings.filterwarnings("ignore") 
import torch
from torch import nn
from torch.nn import functional as F
import math

import collections.abc
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Set, Tuple, Union
import numpy as np
from transformers import AutoConfig

# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid):
    """Sinusoid position encoding table"""

    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)
class VideoMAEEmbeddings(nn.Module):
    """
    Construct the patch and position embeddings.

    """

    def __init__(self, config):
        super().__init__()

        self.patch_embeddings = VideoMAEPatchEmbeddings(config)
        self.num_patches = self.patch_embeddings.num_patches
        # fixed sin-cos embedding
        self.position_embeddings = get_sinusoid_encoding_table(self.num_patches, config.hidden_size)
        self.config = config

    def forward(self, pixel_values, bool_masked_pos):
        # create patch embeddings
        embeddings = self.patch_embeddings(pixel_values)

        # add position embeddings
        embeddings = embeddings + self.position_embeddings.type_as(embeddings).to(embeddings.device).clone().detach()

        # only keep visible patches
        # ~bool_masked_pos means visible
        if bool_masked_pos is not None:
            batch_size, _, num_channels = embeddings.shape
            embeddings = embeddings[~bool_masked_pos]
            embeddings = embeddings.reshape(batch_size, -1, num_channels)
            

        return embeddings


class VideoMAEPatchEmbeddings(nn.Module):
    """
    Video to Patch Embedding. This module turns a batch of videos of shape (batch_size, num_frames, num_channels,
    height, width) into a tensor of shape (batch_size, seq_len, hidden_size) to be consumed by a Transformer encoder.

    The seq_len (the number of patches) equals (number of frames // tubelet_size) * (height // patch_size) * (width //
    patch_size).

    """

    def __init__(self, config):
        super().__init__()

        image_size = config.image_size
        patch_size = config.patch_size
        num_channels = config.num_channels
        hidden_size = config.hidden_size
        num_frames = config.num_frames
        tubelet_size = config.tubelet_size

        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        self.image_size = image_size
        self.patch_size = patch_size
        self.tubelet_size = int(tubelet_size)
        num_patches = (
            (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0]) * (num_frames // self.tubelet_size)
        )
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.projection = nn.Conv3d(
            in_channels=num_channels,
            out_channels=hidden_size,
            kernel_size=(self.tubelet_size, patch_size[0], patch_size[1]),
            stride=(self.tubelet_size, patch_size[0], patch_size[1]),
        )

    def forward(self, pixel_values):
        batch_size, num_frames, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        if height != self.image_size[0] or width != self.image_size[1]:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
            )
        # permute to (batch_size, num_channels, num_frames, height, width)
        pixel_values = pixel_values.permute(0, 2, 1, 3, 4)
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return embeddings

# Copied from transformers.models.vit.modeling_vit.ViTEncoder with ViT->VideoMAE
class VideoMAEEncoder(nn.Module):
    def __init__(self, config, num_layers: int) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([VideoMAELayer(config) for _ in range(num_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        
        hidden_states,
        attention_mask = None,
        head_mask = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) :
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    layer_head_mask,
                )
            else:
                layer_outputs = layer_module(hidden_states , attention_mask , layer_head_mask, output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
    
        return hidden_states
        # return BaseModelOutput(
        #     last_hidden_state=hidden_states,
        #     hidden_states=all_hidden_states,
        #     attentions=all_self_attentions,
        # )

class VideoMAELayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config) -> None:
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = VideoMAEAttention(config)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.intermediate = VideoMAEIntermediate(config)
        self.output = VideoMAEOutput(config)

    def forward(
        self,
        hidden_states ,
        attention_mask = None, 
        head_mask = None,
        output_attentions: bool = False,
    ) :
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in VideoMAE, layernorm is applied before self-attention
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in VideoMAE, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs
    
class VideoMAEAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.attention = VideoMAESelfAttention(config)
        self.output = VideoMAESelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads) -> None:
        if len(heads) == 0:
            return
        # heads, index = find_pruneable_heads_and_indices(
        #     heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        # )

        # # Prune linear layers
        # self.attention.query = prune_linear_layer(self.attention.query, index)
        # self.attention.key = prune_linear_layer(self.attention.key, index)
        # self.attention.value = prune_linear_layer(self.attention.value, index)
        # self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask = None,
        head_mask = None,
        output_attentions: bool = False,
    ) :
        self_outputs = self.attention(hidden_states , attention_mask , head_mask, output_attentions)

        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs
    
class VideoMAESelfAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=False)

        if config.qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(self.all_head_size))
            self.v_bias = nn.Parameter(torch.zeros(self.all_head_size))
        else:
            self.q_bias = None
            self.v_bias = None

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None,  head_mask = None, output_attentions: bool = False):


        k_bias = torch.zeros_like(self.v_bias, requires_grad=False) if self.q_bias is not None else None
        keys = nn.functional.linear(input=hidden_states, weight=self.key.weight, bias=k_bias)
        values = nn.functional.linear(input=hidden_states, weight=self.value.weight, bias=self.v_bias)
        queries = nn.functional.linear(input=hidden_states, weight=self.query.weight, bias=self.q_bias)

        key_layer = self.transpose_for_scores(keys)
        value_layer = self.transpose_for_scores(values) 
        query_layer = self.transpose_for_scores(queries)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        
        if attention_mask is not None:
            # Is that value 1 or 12?
            attention_mask = attention_mask.expand(-1 , 1 , attention_mask.shape[-1] , -1) # (batch_size , 1 , 1 , seq_length) ->(batch_size , 12 , seq_length , seq_length) 
            attention_probs = attention_probs + attention_mask


            #   attn_weights = attn_weights.view(bsz, self.n_heads, tgt_len, src_len) + attention_mask
            # # print(f"attn_weights shape after adding mask is {attn_weights.shape} \n" , flush = True)
            # attn_weights = attn_weights.view(bsz * self.n_heads, tgt_len, src_len)


        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs

class VideoMAEIntermediate(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = nn.GELU() #TODO: NEED TO CHECK THIS
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:

        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states

class VideoMAESelfOutput(nn.Module):
    """
    The residual connection is defined in VideoMAELayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:

        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states
    
class VideoMAEOutput(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states + input_tensor

        return hidden_states

class Head(nn.Module):
    
    def __init__(self , num_layers = 12 , learn_PosEmbeddings = True) -> None:
        super().__init__()
        self.config = AutoConfig.from_pretrained("MCG-NJU/videomae-base")
        self.embeddings = VideoMAEEmbeddings(self.config)
        self.encoder = VideoMAEEncoder(self.config, num_layers).apply(self.randomize_model)
        self.rand_norm = nn.LayerNorm(768)
        self.embedding = nn.Embedding(3 , 768)
        self.embedding.weight.requires_grad = learn_PosEmbeddings # Now the embedding is static
        
        self.iter = 1

    
    def randomize_model(self,model):
        for module_ in model.named_modules(): 
            if isinstance(module_[1],(torch.nn.Linear, torch.nn.Embedding)):
                module_[1].weight = torch.nn.init.xavier_uniform_(module_[1].weight)
            elif isinstance(module_[1], torch.nn.LayerNorm):
                module_[1].bias.data.zero_()
                module_[1].weight = torch.nn.init.constant_(nn.Parameter(torch.empty(768)) , 1)
                # module_[1].weight.data.fill_(1.0)
            if isinstance(module_[1], torch.nn.Linear) and module_[1].bias is not None:
                module_[1].bias.data.zero_()
        return model
    def masking(self , m , mask):
        batch_size , seq_len , _ = m.shape 
        pos = torch.zeros((batch_size , seq_len ) , device=torch.device('cuda') , dtype=torch.int)        
        if mask is not None:
            mask = (1.0 - mask[:, None, None, :]) * torch.finfo(torch.float16).min
        return pos , mask
    def forward(self , pixel_values , bool_masked_pos ,  ta , ta_mask , ta_embed):
        """
        ta: only text and audio at first since video will come from the embeddings inside
        ta_mask: only text and audio masks
        """
        embedding_output = self.embeddings(pixel_values, bool_masked_pos)
        
        del pixel_values
        del bool_masked_pos

        tav = torch.concat((ta ,embedding_output) , dim=1)
        
        visual_pos , visual_mask = self.masking(embedding_output , torch.ones(embedding_output.shape[0:2]).type(torch.float16).to("cuda")   )
        if self.iter == 1:
            print(f"the shape of ta_mask before encoder is {ta_mask.shape}\n shape of video is {visual_mask.shape}", flush = True)
            print(f"shape of video is {embedding_output.shape}\nthe shape of tav before encoder is {tav.shape}\n", flush = True)
            self.iter += 1
        del embedding_output
        del ta
        
        tav_embed = torch.concat((ta_embed , visual_pos+2) , dim = 1)
        
        del ta_embed
        del visual_pos
        
        tav_mask = torch.concat((ta_mask , visual_mask) ,  dim = -1)
        
        del ta_mask
        del visual_mask
        
        tav = tav + self.embedding(tav_embed)
        
        del tav_embed
        
        tav = self.encoder(tav , tav_mask)
        
        del tav_mask
        
        tav = self.rand_norm(torch.mean(tav, dim=1))
        
        return tav
        
        
        
        