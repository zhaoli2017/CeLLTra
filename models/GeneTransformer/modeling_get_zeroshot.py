from cmath import log
import math
from pathlib import Path
import sys, os

from torch.nn import CrossEntropyLoss

path_root = Path(os.path.abspath(__file__)).parents[1]
sys.path.append(str(path_root))

from transformers.modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.modeling_outputs import BaseModelOutput, MaskedLMOutput, SequenceClassifierOutput, \
    BaseModelOutputWithPooling

from transformers.activations import ACT2FN

from models.GeneTransformer.configuration_get import GEXTConfig


import torch
from torch import nn

import numpy as np

class GEXTEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        #self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size)) if use_mask_token else None
        #self.gene_embeddings = nn.Parameter(torch.rand(config.num_genes, config.hidden_size))
        #self.batchnorm = nn.BatchNorm1d(config.num_genes)
        self.cls_token = nn.Parameter(torch.zeros(1,1,config.hidden_size))
        self.gene_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config

        # self.gene_embeddings_random = nn.Embedding(config.vocab_size, config.hidden_size)
        # print('load external embeddings!!!')
        # all_embeddings_uci_array = np.load('/home/zli17/work/projects/geneTransformer/io/output_uci_dgex/Guocai_bioalbert.npy')
        # self.gene_embeddings.weight = nn.Parameter(torch.Tensor(all_embeddings_uci_array))
        # self.gene_embeddings.weight.requires_grad = False

        #self.gene_embeddings.weight.data.copy_(torch.rand((config.vocab_size, config.hidden_size))+1.0)
        #self.group_embeddings = GeneGroupEmbeddings()

    def forward(self, gene_expression, gene_input_ids, bool_masked_pos, group_mtx, mask_id=1):
        #gene_expression (bs * n_gene)
        #gene_input_ids (bs * n_gene)
        #bool_masked_pos (bs * n_gene) or None
        #group_mtx (bs * n_group * n_gene)

        # n_gene_group = torch.repeat_interleave(torch.sum(group_mtx, 1, keepdim=True), self.gene_embeddings.shape[1], dim=1)
        # embeddings = (torch.matmul((gene_expression.unsqueeze(1) * group_mtx), self.gene_embeddings) / n_gene_group)

        ## todo add a constant
        #gene_expression = self.batchnorm(gene_expression)
        #gene_expression = gene_expression * (1.0 - bool_masked_pos) + bool_masked_pos
        batch_size = gene_expression.shape[0]
        if bool_masked_pos is not None:
            gene_expression = gene_expression * (1.0 - bool_masked_pos) ### remove the masked genes
            gene_input_ids = gene_input_ids * (1-bool_masked_pos) + bool_masked_pos * mask_id

        gene_embeddings = self.gene_embeddings(gene_input_ids)  ## (bs, n_genes, hidden_size)
        gex_embeddings = gene_expression.unsqueeze(2) * gene_embeddings
        n_gene_group = torch.sum(group_mtx, 2, keepdim=True)
        embeddings = torch.matmul(gex_embeddings.transpose(2,1), group_mtx.transpose(2,1)).transpose(2,1)
        embeddings = embeddings/n_gene_group
        # add the [CLS] token to the embedded grouped genes
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        return embeddings

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
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:,0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)

        return pooled_output

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
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

class GEXTForMaskedGEXModeling(GEXTPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.gext = GEXTModel(config, add_pooling_layer=False)
        self.decoder = nn.Linear(config.hidden_size, config.num_genes)
        #self.l2loss = nn.MSELoss(reduction='none')
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
        sequence_output = sequence_output[:, 1:]
        # Reconstruct gene expression values
        reconstructed_gex_values = self.decoder(sequence_output)
        #reconstructed_gex_values = (reconstructed_gex_values - 30) * group_mtx
        reconstructed_gex_values = reconstructed_gex_values * group_mtx
        #reconstructed_gex_values = torch.mean(reconstructed_gex_values, dim=1)
        reconstructed_gex_values = torch.sum(reconstructed_gex_values, dim=1) / torch.sum(group_mtx, dim=1)

        masked_gex_loss = None
        if bool_masked_pos is not None:
            reconstruction_loss = nn.functional.l1_loss(gene_expression, reconstructed_gex_values, reduction="none")
            #reconstruction_loss = self.l2loss(gene_expression, reconstructed_gex_values)
            masked_gex_loss = (reconstruction_loss * bool_masked_pos).sum() / (bool_masked_pos.sum() + 1e-5)

        if not return_dict:
            output = (reconstructed_gex_values,) + outputs[2:]
            return ((masked_gex_loss,) + output) if masked_gex_loss is not None else output

        return MaskedLMOutput(
            loss=masked_gex_loss,
            logits=reconstructed_gex_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class GEXTForCellClassification(GEXTPreTrainedModel):
    def __init__(self, config: GEXTConfig):
        super().__init__(config)

        self.nun_labels = config.num_labels
        self.gext = GEXTModel(config, add_pooling_layer=True)

        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.post_init()

    def forward(self,
                gene_expression=None,
                label=None,
                gene_input_ids=None,
                bool_masked_pos=None,
                group_mtx=None,
                mask_id=1,
                head_mask=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=True,
                ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.gext(gene_expression=gene_expression,
                            gene_input_ids=gene_input_ids,
                            bool_masked_pos=bool_masked_pos,
                            group_mtx=group_mtx,
                            output_hidden_states=output_hidden_states,
                            output_attentions=output_attentions,
                            return_dict=return_dict,
                            mask_id=mask_id,
                            head_mask=head_mask)

        sequence_output = outputs[0]

        logits = self.classifier(sequence_output[:, 0, :])

        if label is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), label.view(-1))
        else:
            raise Exception('Please provide labels for CellClassification task.')

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output)

        return SequenceClassifierOutput(loss=loss,
                                        logits=logits,
                                        hidden_states=outputs.hidden_states,
                                        attentions=outputs.attentions)

def zeroshot_loss(output, target):
    loss = torch.sum((output - target)**2, dim=1, keepdim=True)
    loss = torch.mean(loss)
    return loss

# class GEXTForGeneClassification_ZeroShot(GEXTPreTrainedModel):
#     def __init__(self, config: GEXTConfig):
#         super().__init__(config)

#         self.num_labels = config.num_labels
#         self.gext = GEXTModel(config, add_pooling_layer=True)

#         self.cl_emb = torch.Tensor(np.load('/home/zli17/now/io/tabula_sapiens/output-zeroshot/onClass_emb.npy')).cuda()
#         self.cl_emb_dim = self.cl_emb.shape[1]
#         self.cl_num = self.cl_emb.shape[0]
#         hidden_size = 512
#         self.classifier_zs = nn.Sequential(
#             nn.Linear(config.hidden_size + self.cl_emb_dim, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, 1),
#             nn.Sigmoid()
#         )

#         self.post_init()

#     def forward(self,
#                 gene_expression=None,
#                 label=None,
#                 gene_input_ids=None,
#                 bool_masked_pos=None,
#                 group_mtx=None,
#                 cl_emb_mtx=None,
#                 mask_id=1,
#                 head_mask=None,
#                 output_attentions=None,
#                 output_hidden_states=None,
#                 return_dict=True,
#                 ):
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         outputs = self.gext(gene_expression=gene_expression,
#                             gene_input_ids=gene_input_ids,
#                             bool_masked_pos=bool_masked_pos,
#                             group_mtx=group_mtx,
#                             output_hidden_states=output_hidden_states,
#                             output_attentions=output_attentions,
#                             return_dict=return_dict,
#                             mask_id=mask_id,
#                             head_mask=head_mask)

#         sequence_output = outputs[0]

#         x = sequence_output[:, 0, :]
#         x_expand = torch.cat((x.repeat(self.cl_num, 1, 1).transpose(1, 0), cl_emb_mtx), 2)
#         logits = self.classifier_zs(x_expand).squeeze()

#         if label is not None:
#             loss_fct = nn.MSELoss(reduction='sum')
#             loss_fct = zeroshot_loss
#             label_onehot = nn.functional.one_hot(label, num_classes=self.cl_num)
#             loss = loss_fct(logits, label_onehot.float())
#         else:
#             raise Exception('Please provide labels for CellClassification task.')

#         if not return_dict:
#             output = (logits,) + outputs[1:]
#             return ((loss,) + output)

#         return SequenceClassifierOutput(loss=loss,
#                                         logits=logits,
#                                         hidden_states=outputs.hidden_states,
#                                         attentions=outputs.attentions)

class GEXTForGeneClassification_ZeroShot(GEXTPreTrainedModel):
    def __init__(self, config: GEXTConfig):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.gext = GEXTModel(config, add_pooling_layer=True)

        self.cl_emb = torch.Tensor(np.load('/home/zli17/now/io/tabula_sapiens/output-zeroshot/onClass_emb.npy')).cuda()
        self.cl_emb_dim = self.cl_emb.shape[1]
        self.cl_num = self.cl_emb.shape[0]
        hidden_size = 512
        self.classifier_zs = nn.Sequential(
            nn.Linear(config.hidden_size + self.cl_emb_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            #nn.Softmax(dim=1)
        )
        
        # self.classifier_zs = nn.Sequential(nn.Linear(config.hidden_size, self.cl_num)
        #                                    , nn.Sigmoid())
        # self.classifier_zs = nn.Sequential(nn.Linear(config.hidden_size, self.cl_emb_dim), nn.LeakyReLU())
        # self.softmax = nn.Softmax(dim=1)
        # self.sigmoid = nn.Sigmoid()
        self.post_init()

    def forward(self,
                gene_expression=None,
                label=None,
                gene_input_ids=None,
                bool_masked_pos=None,
                group_mtx=None,
                cl_emb_mtx=None,
                mask_id=1,
                head_mask=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=True,
                ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.gext(gene_expression=gene_expression,
                            gene_input_ids=gene_input_ids,
                            bool_masked_pos=bool_masked_pos,
                            group_mtx=group_mtx,
                            output_hidden_states=output_hidden_states,
                            output_attentions=output_attentions,
                            return_dict=return_dict,
                            mask_id=mask_id,
                            head_mask=head_mask)

        sequence_output = outputs[0]

        x = sequence_output[:, 0, :]
        x_expand = torch.cat((x.repeat(self.cl_num, 1, 1).transpose(1, 0), cl_emb_mtx), 2)
        logits = self.classifier_zs(x_expand).squeeze()
        # logits = self.classifier_zs(sequence_output[:, 0, :])
        # logits = torch.matmul(logits, cl_emb_mtx[0].transpose(1, 0))
        # logits = self.softmax(logits)
        #logits = self.sigmoid(logits)
        if label is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, label)
            
            # loss_fct = zeroshot_loss
            # label_onehot = nn.functional.one_hot(label, num_classes=self.cl_num)
            # loss = loss_fct(logits, label_onehot.float())
        else:
            raise Exception('Please provide labels for CellClassification task.')

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output)

        return SequenceClassifierOutput(loss=loss,
                                        logits=logits,
                                        hidden_states=outputs.hidden_states,
                                        attentions=outputs.attentions)

if __name__ == '__main__':

    config = GEXTConfig()
    config.update(
        {
            "hidden_size": 200,
            "num_hidden_layers": 8,
            "num_attention_heads": 10,
            "intermediate_size": 1024,
            "num_genes": 2000,
            "vocab_size": 2000 + 2,
            "num_labels": 177
        }
    )

    model_cls = GEXTForGeneClassification_ZeroShot(config)

    print(model_cls)
    bs = 8
    gene_expression = torch.rand(bs, config.num_genes)
    gene_input_ids = torch.repeat_interleave(torch.LongTensor([list(range(2, config.vocab_size))]), bs, dim=0)
    labels = torch.randint(low=0, high=2331, size=(bs,))

    bool_masked_pos = None

    group_mtx = torch.randint(0, high=2, size=(8, 400, config.num_genes)).type(torch.FloatTensor)

    output = model_cls(gene_expression = gene_expression,
                   gene_input_ids = gene_input_ids,
                   label=labels,
                   bool_masked_pos = bool_masked_pos,
                   group_mtx = group_mtx,
                   output_hidden_states=True,
                   output_attentions=True)

    print('Loss:', output[0])
