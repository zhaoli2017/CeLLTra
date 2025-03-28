import math
from pathlib import Path
import sys, os

path_root = Path(os.path.abspath(__file__)).parents[1]
sys.path.append(str(path_root))

from transformers.modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.modeling_outputs import BaseModelOutput, MaskedLMOutput
from transformers.activations import ACT2FN

from models.GeneTransformer.configuration_get import GEXTConfig


import torch
from torch import nn

class GEXTEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        #self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size)) if use_mask_token else None
        #self.gene_embeddings = nn.Parameter(torch.rand(config.num_genes, config.hidden_size))
        #self.batchnorm = nn.BatchNorm1d(config.num_genes)
        self.gene_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.landmark_genes = config.landmark_genes
        #self.gene_embeddings.weight.data.copy_(torch.rand((config.vocab_size, config.hidden_size))+1.0)
        #self.group_embeddings = GeneGroupEmbeddings()

    def forward(self, gene_expression, gene_input_ids, bool_masked_pos, group_mtx, mask_id=1):
        #gene_expression (bs * n_gene)
        #gene_input_ids (bs * n_gene)
        #bool_masked_pos (bs * n_gene)
        #group_mtx (bs * n_group * n_gene)

        # n_gene_group = torch.repeat_interleave(torch.sum(group_mtx, 1, keepdim=True), self.gene_embeddings.shape[1], dim=1)
        # embeddings = (torch.matmul((gene_expression.unsqueeze(1) * group_mtx), self.gene_embeddings) / n_gene_group)

        ## todo add a constant
        #gene_expression = self.batchnorm(gene_expression)
        #gene_expression = gene_expression * (1.0 - bool_masked_pos) + bool_masked_pos
        gene_expression = gene_expression[:, :self.landmark_genes]
        gene_input_ids = gene_input_ids[:, :self.landmark_genes]
        gene_embeddings = self.gene_embeddings(gene_input_ids) ## (bs, n_genes, hidden_size)
        gex_embeddings = gene_expression.unsqueeze(2) * gene_embeddings
        return gex_embeddings

class GEXTSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(self.query(hidden_states))

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

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs

class GEXTSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states

class GEXTAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = GEXTSelfAttention(config)
        self.output = GEXTSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)

        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

class GEXTIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states

class GEXTOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states + input_tensor

        return hidden_states

class GEXTLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = GEXTAttention(config)
        self.intermediate = GEXTIntermediate(config)
        self.output = GEXTOutput(config)

        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),
            head_mask,
            output_attentions=output_attentions,
        )

        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        # first residual connection
        hidden_states = attention_output + hidden_states

        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs

class GEXTEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([GEXTLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(self,
                hidden_states,
                head_mask=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states, )

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
            all_hidden_states = all_hidden_states + (hidden_states, )

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

class GEXTPooler(nn.Module):
    def __init__(self, config):
        super().__init__()


    def forward(self):
        pass

class GEXTPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GEXTConfig
    base_model_prefix = 'gext'
    main_input_name = 'gene_expression' ## values or changes of gene expression

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)


class GEXTModel(GEXTPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = GEXTEmbeddings(config)
        self.encoder = GEXTEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = GEXTPooler(config) if add_pooling_layer else None

        # Initialize weights
        self.post_init()

    def forward(self,
                gene_expression=None,
                gene_input_ids=None,
                bool_masked_pos=None,
                group_mtx=None,
                mask_id=1,
                head_mask=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if gene_expression is None:
            raise ValueError("You have to specify gene expression values or changes")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            gene_expression, gene_input_ids, bool_masked_pos, group_mtx, mask_id
        )

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

class GEXTForMaskedGEXModeling(GEXTPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.gext = GEXTModel(config, add_pooling_layer=False)
        self.decoder = nn.Linear(config.hidden_size, config.num_genes-config.landmark_genes)
        #self.l2loss = nn.MSELoss(reduction='none')
        self.landmark_genes = config.landmark_genes
        self.post_init()

    def forward(self,
                gene_expression=None,
                gene_input_ids=None,
                bool_masked_pos=None,
                group_mtx=None,
                mask_id=1,
                head_mask=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                ):
        outputs = self.gext(gene_expression=gene_expression,
                       gene_input_ids=gene_input_ids,
                       bool_masked_pos=bool_masked_pos,
                       group_mtx=group_mtx,
                       output_hidden_states=output_hidden_states,
                       output_attentions=output_attentions,
                       return_dict=return_dict,
                       mask_id = mask_id,
                       head_mask = head_mask)

        sequence_output = outputs[0]

        # Reconstruct gene expression values
        reconstructed_gex_values = self.decoder(sequence_output)
        reconstructed_gex_values = torch.mean(reconstructed_gex_values, dim=1)

        masked_gex_loss = None
        if bool_masked_pos is not None:
            reconstruction_loss = nn.functional.l1_loss(gene_expression[:, self.landmark_genes:], reconstructed_gex_values)
            masked_gex_loss = reconstruction_loss

        if not return_dict:
            output = (reconstructed_gex_values,) + outputs[2:]
            return ((masked_gex_loss,) + output) if masked_gex_loss is not None else output

        return MaskedLMOutput(
            loss=masked_gex_loss,
            logits=reconstructed_gex_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

if __name__ == '__main__':

    config = GEXTConfig()
    model = GEXTForMaskedGEXModeling(config)
    print(model)

    import numpy as np
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    print(f'num_parameters: {params}')

    # from torchviz import make_dot

    bs = 8
    gene_expression = torch.rand(bs, config.num_genes)
    gene_input_ids = torch.repeat_interleave(torch.LongTensor([list(range(2, config.vocab_size))]), bs, dim=0)

    import numpy as np
    mask_l = []

    mask_ratio = 0.1
    mask_count = int(np.ceil(config.num_genes * mask_ratio))
    for _ in range(8):
        mask_idx = np.random.permutation(config.num_genes)[: mask_count]
        mask = np.zeros(config.num_genes, dtype=int)
        mask[mask_idx] = 1
        mask_l.append(mask)
    bool_masked_pos = torch.LongTensor(np.array(mask_l))
    group_mtx = torch.randint(0, high=2, size=(8, 400, config.num_genes)).type(torch.FloatTensor)

    output = model(gene_expression = gene_expression,
                   gene_input_ids = gene_input_ids,
                   bool_masked_pos = bool_masked_pos,
                   group_mtx = group_mtx,
                   output_hidden_states=True,
                   output_attentions=True)
    print('Loss:', output[0])

    # make_dot(output[1], params=dict(list(model.named_parameters()))).render("gext_torchviz", format="png")

