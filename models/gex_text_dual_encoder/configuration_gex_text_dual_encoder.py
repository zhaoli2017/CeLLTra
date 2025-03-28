from pathlib import Path
import sys, os
path_root = Path(os.path.abspath(__file__)).parents[1]
sys.path.append(str(path_root))

import copy

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers import AutoConfig
from models.GeneTransformer.configuration_get import GEXTConfig

class GEXTextDualEncoderConfig(PretrainedConfig):
    model_type = "gex_text_dual_encoder"
    is_composition = True
    
    def __init__(self, projection_dim=512, logit_scale_init_value=2.6592, **kwargs):
        super().__init__(**kwargs)
        
        if 'gex_config' not in kwargs:
            raise ValueError("`gex_config` must be specified.")
        
        if 'text_config' not in kwargs:
            raise ValueError("`text_config` must be specified.")
        
        gex_config = kwargs.pop('gex_config')
        text_config = kwargs.pop('text_config')
        
        gex_model_type = gex_config.pop("model_type")
        text_model_type = text_config.pop("model_type")
        
        if gex_model_type == 'gext':
            self.gex_config = GEXTConfig(**gex_config)
        else:
            raise ValueError(f"Unknown gex model type: {gex_model_type}")
        
        self.text_config = AutoConfig.for_model(text_model_type, **text_config)
        
        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value
    
    @classmethod
    def from_gex_text_configs(cls, gex_config, text_config, **kwargs):
        return cls(gex_config=gex_config.to_dict(), text_config=text_config.to_dict(), **kwargs)
    
    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].
        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["gex_config"] = self.gex_config.to_dict()
        output["text_config"] = self.text_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output