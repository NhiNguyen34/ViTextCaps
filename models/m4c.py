import torch
from torch import nn
from transformers import BertGenerationDecoder, BertGenerationEncoder

from models.modules.object_encoder import ObjectEncoder
from models.modules.ocr_encoder import OCREncoder
from utils import get_padding_mask

from typing import Optional, Tuple

class M4C(BertGenerationDecoder):
    def __init__(self, config, pretrained_config):
        pretrained_config = pretrained_config.update({
            "is_decoder": True,
            "add_cross_attention": True
        })
        super().__init__(pretrained_config)
        self.bert.from_pretrained(config.pretrained_name)

        pretrained_config = pretrained_config.update({
            "is_decoder": False,
            "add_cross_attention": False
        })
        self.encoder = BertGenerationEncoder(pretrained_config)
        self.encoder.from_pretrained(config.pretrained_name)
        self.encoder.set_input_embeddings(self.bert.get_input_embeddings())

        self.object_encoder = ObjectEncoder(
            config.object_in_dim,
            pretrained_config.d_model,
            pretrained_config.dropout
        )
        self.ocr_encoder = OCREncoder(
            config.ocr_in_dim,
            pretrained_config.d_model,
            pretrained_config.dropout
        )
        self.masked_vision_value = config.masked_vision_value

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            object_features: Optional[torch.Tensor] = None,
            object_boxes: Optional[torch.Tensor] = None,
            ocr_boxes: Optional[torch.Tensor] = None,
            ocr_tokens: Optional[torch.Tensor] = None,
            ocr_det_features: Optional[torch.Tensor] = None,
            ocr_rec_features: Optional[torch.Tensor] = None,
            caption_tokens: Optional[torch.Tensor] = None,
            caption_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None
    ):
        obj_feats = self.object_encoder(
            object_boxes,
            object_features
        )
        obj_mask = get_padding_mask(obj_feats, self.masked_vision_value)

        ocr_feats = self.ocr_encoder(
            ocr_boxes,
            ocr_tokens,
            ocr_rec_features,
            ocr_det_features
        )
        ocr_mask = get_padding_mask(ocr_feats, self.masked_vision_value)

        input_embs = torch.cat([obj_feats, ocr_feats], dim=1)
        input_mask = torch.cat([obj_mask, ocr_mask], dim=1)

        encoder_outputs = self.encoder(
            inputs_embeds=input_embs,
            attention_mask=input_mask
        ).hidden_states

        return super().forward(
            input_ids=caption_tokens,
            attention_mask=caption_mask,
            encoder_hidden_states=encoder_outputs,
            encoder_attention_mask=input_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=input_embs,
            #encoder_hidden_states=encoder_hidden_states,
            #encoder_attention_mask=encoder_attention_mask,
            labels=caption_tokens,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids, 
            "attention_mask": attention_mask, 
            "past_key_values": past_key_values,
            **model_kwargs
        }
