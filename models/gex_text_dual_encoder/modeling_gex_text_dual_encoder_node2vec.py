from pathlib import Path
import sys, os
path_root = Path(os.path.abspath(__file__)).parents[2]
sys.path.append(str(path_root))

from typing import Optional, Tuple, Any

import torch
from torch import nn

from dataclasses import dataclass
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers import AutoModel
from transformers import AutoConfig
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, SequenceClassifierOutput

from models.gex_text_dual_encoder.configuration_gex_text_dual_encoder import GEXTextDualEncoderConfig
from models.GeneTransformer.modeling_get import GEXTModel
from models.GeneTransformer.configuration_get import GEXTConfig

logger = logging.get_logger(__name__)

@dataclass
class DualModelOutput(ModelOutput):
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for image-text similarity.
        logits_per_cell:(`torch.FloatTensor` of shape `(cell_batch_size, text_batch_size)`):
            The scaled dot product scores between `gex_embeds` and `text_embeds`. This represents the image-text
            similarity scores.
        logits_per_text:(`torch.FloatTensor` of shape `(text_batch_size, gex_batch_size)`):
            The scaled dot product scores between `text_embeds` and `gex_embeds`. This represents the text-image
            similarity scores.
        text_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`CLIPTextModel`].
        gex_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The image embeddings obtained by applying the projection layer to the pooled output of [`CLIPVisionModel`].
        text_model_output(`BaseModelOutputWithPooling`):
            The output of the [`TextModel`].
        gex_model_output(`BaseModelOutputWithPooling`):
            The output of the [`GexModel`].
    """

    loss: Optional[torch.FloatTensor] = None
    logits_per_cell: torch.FloatTensor = None
    logits_per_text: torch.FloatTensor = None
    text_embeds: torch.FloatTensor = None
    gex_embeds: torch.FloatTensor = None
    #text_model_output: BaseModelOutputWithPooling = None
    #gex_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "gex_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


# Copied from transformers.models.clip.modeling_clip.contrastive_loss
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

# Copied from transformers.models.clip.modeling_clip.clip_loss
def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    cell_loss = contrastive_loss(similarity.T)
    return (caption_loss + cell_loss) / 2.0



class GEXTextDualEncoderModel(PreTrainedModel):
    config_class = GEXTextDualEncoderConfig
    base_model_prefix = "gex_text_dual_encoder"
    
    def __init__(self, config: Optional[GEXTextDualEncoderConfig] = None
                 , gex_model: Optional[PreTrainedModel] = None
                 , text_model: Optional[PreTrainedModel] = None):
        
        if config is None and (gex_model is not None or text_model is not None):
            raise ValueError("Either a configuration or an vision and a text model has to be provided")
        if config is None:
            config = GEXTextDualEncoderConfig.from_gex_text_configs(gex_model.config
                                                                    , text_model.config)
        else:
            if not isinstance(config, self.config_class):
                raise ValueError(f"config: {config} has to be of type {self.config_class}")
            
        super().__init__(config)
        
        if gex_model is None:
            if isinstance(config.gex_config, GEXTConfig):
                gex_model = GEXTModel(config.gex_config)
        
        if text_model is None:
            text_model = AutoModel.from_config(config.text_config)
            
        self.gex_model = gex_model
        self.text_model = text_model
        
        self.gex_model.config = self.config.gex_config
        self.text_model.config = self.config.text_config
        
        self.gex_embed_dim = config.gex_config.hidden_size
        self.text_embed_dim = config.text_config.hidden_size
        self.projection_dim = config.projection_dim
        
        self.gex_projection = nn.Linear(self.gex_embed_dim, self.projection_dim)
        #self.text_projection = nn.Linear(self.text_embed_dim + 832, self.projection_dim)
        self.text_projection = nn.Linear(self.text_embed_dim + 64, self.projection_dim)
        #self.text_projection = nn.Linear(self.text_embed_dim + 1536, self.projection_dim) # bert emb + openai_api emb
        # self.gex_projection = nn.Sequential(
        #     nn.Linear(self.gex_embed_dim, self.projection_dim+100),
        #     nn.Tanh(),
        #     nn.Linear(self.projection_dim+100, self.projection_dim)
        # )
        # self.text_projection = nn.Sequential(
        #     nn.Linear(self.text_embed_dim, self.projection_dim+100),
        #     nn.Tanh(),
        #     nn.Linear(self.projection_dim+100, self.projection_dim)
        # )
        self.logit_scale = nn.Parameter(torch.ones([]) * self.config.logit_scale_init_value)
    
    def forward(self,
        input_ids=None,
        gene_expression=None,
        gene_input_ids=None,
        bool_masked_pos=None,
        group_mtx=None,
        label=None,
        mask_id=1,
        cl_node2vec = None,
        head_mask=None,
        attention_mask=None,
        position_ids=None,
        return_loss=None,
        token_type_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None):

        return_dict = return_dict if return_dict is not None else self.config.return_dict
        gex_outputs = self.gex_model(gene_expression=gene_expression,
                            gene_input_ids=gene_input_ids,
                            bool_masked_pos=bool_masked_pos,
                            group_mtx=group_mtx,
                            output_hidden_states=output_hidden_states,
                            output_attentions=output_attentions,
                            return_dict=return_dict,
                            mask_id=mask_id,
                            head_mask=head_mask)

        if input_ids.shape[1] == 1:
            input_ids = input_ids.squeeze(1)
            attention_mask = attention_mask.squeeze(1)
        else:
            input_ids = input_ids[0, :, :]
            attention_mask = attention_mask[0, :, :]
            cl_node2vec = cl_node2vec[0, :, :]

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        gex_embeds = gex_outputs[0]
        gex_embeds = self.gex_projection(gex_embeds[:, 0, :])
        
        text_embeds = text_outputs[1]  # pooler_output
        #text_embeds = self.text_projection(text_embeds)
        text_embeds = self.text_projection(torch.cat([text_embeds, cl_node2vec], dim=1))
        
        # normalized features
        gex_embeds = gex_embeds / gex_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, gex_embeds.t()) * logit_scale
        logits_per_cell = logits_per_text.T
        
        loss = None
        if return_loss:
            if text_embeds.shape[0] == gex_embeds.shape[0]:
                loss = clip_loss(logits_per_text)
            else:
                loss =  torch.tensor(0.0, device=text_embeds.device)

        if not return_dict:
            output = (logits_per_cell, logits_per_text, text_embeds, gex_embeds, text_outputs, gex_outputs)
            return ((loss,) + output) if loss is not None else output
        
        return DualModelOutput(
            loss=loss,
            logits_per_cell=logits_per_cell,
            #logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            gex_embeds=gex_embeds,
            #text_model_output=text_outputs,
            #gex_model_output=gex_outputs,
        )
        
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # At the moment fast initialization is not supported
        # for composite models
        kwargs["_fast_init"] = False
        return super().from_pretrained(*args, **kwargs)

    @classmethod
    def from_gex_text_pretrained(
        cls,
        gex_model_name_or_path: str = None,
        text_model_name_or_path: str = None,
        *model_args,
        **kwargs,
    ) -> PreTrainedModel:
        kwargs_gex = {
            argument[len("gex_") :]: value for argument, value in kwargs.items() if argument.startswith("gex_")
        }

        kwargs_text = {
            argument[len("text_") :]: value for argument, value in kwargs.items() if argument.startswith("text_")
        }

        # remove gex, text kwargs from kwargs
        for key in kwargs_gex.keys():
            del kwargs["gex_" + key]
        for key in kwargs_text.keys():
            del kwargs["text_" + key]
            
        gex_model = kwargs_gex.pop("model", None)
        if gex_model is None:

            if "config" not in kwargs_gex:
                from models.GeneTransformer.configuration_get import GEXTConfig
                gex_config = GEXTConfig()

            if gex_model_name_or_path is None:
                gex_config.update(
                    {
                        "num_genes": kwargs_gex['num_genes'],
                        "vocab_size": kwargs_gex['vocab_size'] ,
                    }
                )
                #kwargs_gex["config"] = gex_config
                gex_model = GEXTModel(gex_config)
                #raise ValueError(
                #    "If `gex_model` is not defined as an argument, a `gex_model_name_or_path` has to be defined"
                #)
            else:

                kwargs_gex["config"] = gex_config
                AutoModel.register(GEXTConfig, GEXTModel)
                gex_model = AutoModel.from_pretrained(gex_model_name_or_path, *model_args, **kwargs_gex)

        text_model = kwargs_text.pop("model", None)
        if text_model is None:
            if text_model_name_or_path is None:
                raise ValueError(
                    "If `text_model` is not defined as an argument, a `text_model_name_or_path` has to be defined"
                )

            if "config" not in kwargs_text:
                text_config = AutoConfig.from_pretrained(text_model_name_or_path)
                kwargs_text["config"] = text_config

            text_model = AutoModel.from_pretrained(text_model_name_or_path, *model_args, **kwargs_text)
            
        # instantiate config with corresponding kwargs
        config = GEXTextDualEncoderConfig.from_gex_text_configs(gex_model.config, text_model.config, **kwargs)
        
        # init model
        model = cls(config=config, gex_model=gex_model, text_model=text_model)
        
        # the projection layers are always newly initialized when loading the model
        # using pre-trained gex and text model.
        logger.warning(
            "The projection layer and logit scale weights `['gex_projection.weight', 'text_projection.weight',"
            " 'logit_scale']` are newly initialized. You should probably TRAIN this model on a down-stream task to be"
            " able to use it for predictions and inference."
        )

        return model

if __name__ == "__main__":
    model = GEXTextDualEncoderModel.from_gex_text_pretrained(
        '/home/zli17/work/projects/geneTransformer_sc/io/tabula_sapiens/output-zeroshot/models/no_zs_celoss_concat/checkpoint-4500'
        , 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
    )
    
    print(model)
    
    bs = 8
    num_genes = 27985
    vocab_size = num_genes + 2
    gene_expression = torch.rand(bs, num_genes)
    gene_input_ids = torch.repeat_interleave(torch.LongTensor([list(range(2, vocab_size))]), bs, dim=0)
    labels = torch.randint(low=0, high=2331, size=(bs,))

    bool_masked_pos = None

    group_mtx = torch.randint(0, high=2, size=(8, 400, num_genes)).type(torch.FloatTensor)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract')
    texts = ['This is a test', 'This is another test', 'This is a third test', 'This is a fourth test', 'This is a fifth test', 'This is a sixth test', 'This is a seventh test', 'This is an eighth test']
    input_ids = []
    attention_mask = []
    for captions in texts:
        text_inputs = tokenizer(captions, max_length=512, padding="max_length", truncation=True)
        input_ids.append(text_inputs.input_ids)
        attention_mask.append(text_inputs.attention_mask)
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                gene_expression = gene_expression,
                gene_input_ids = gene_input_ids,
                bool_masked_pos = bool_masked_pos,
                group_mtx = group_mtx,
                output_hidden_states=True,
                output_attentions=True,
                return_loss=True)
    
    print(output)
    