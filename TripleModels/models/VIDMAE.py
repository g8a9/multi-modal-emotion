import torch
from torch import nn
import math

# Copied from transformers.models.vit.modeling_vit.ViTEncoder with ViT->VideoMAE
class VideoMAEEncoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([VideoMAELayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
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
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

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
        head_mask = None,
        output_attentions: bool = False,
    ) :
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in VideoMAE, layernorm is applied before self-attention
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
        head_mask = None,
        output_attentions: bool = False,
    ) :
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)

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
        self, hidden_states, head_mask = None, output_attentions: bool = False):

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
        
        print(f"shape of attention probs is {attention_probs.shape}" , flush = True)

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
