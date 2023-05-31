import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from .bert_model import BertConfig, BertModel, BertCrossLayer
from . import swin_transformer as swin
from . import vit_model as vit
from .vit_model import resize_pos_embed
from . import heads, objectives, meter_utils
from .clip_model import build_model, adapt_position_encoding
from .swin_helpers import swin_adapt_position_encoding
from transformers import RobertaConfig, RobertaModel

class MTTransformer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.prepare_data_per_node = False
        self.is_clip= ('CLIP' in config["vit"])
        self.is_swin= ('swin' in config["vit"])
        self.is_vit= ('vit' in config["vit"])
        self.jump_val_first_for_irtr_itm_irc = True

        if 'roberta' in config['tokenizer']:
            bert_config = RobertaConfig(
                vocab_size=config["vocab_size"],
                hidden_size=config["hidden_size"],
                num_hidden_layers=config["num_layers"],
                num_attention_heads=config["num_heads"],
                intermediate_size=config["hidden_size"] * config["mlp_ratio"],
                max_position_embeddings=config["max_text_len"],
                hidden_dropout_prob=config["drop_rate"],
                attention_probs_dropout_prob=config["drop_rate"],
            )
        else:
            bert_config = BertConfig(
                vocab_size=config["vocab_size"],
                hidden_size=config["hidden_size"],
                num_hidden_layers=config["num_layers"],
                num_attention_heads=config["num_heads"],
                intermediate_size=config["hidden_size"] * config["mlp_ratio"],
                max_position_embeddings=config["max_text_len"],
                hidden_dropout_prob=config["drop_rate"],
                attention_probs_dropout_prob=config["drop_rate"],
            )

        resolution_after = config['image_size']
        self.cross_modal_text_transform = nn.Linear(config['input_text_embed_size'], config['hidden_size'])
        self.cross_modal_image_transform = nn.Linear(config['input_image_embed_size'], config['hidden_size'])
        self.cross_modal_text_transform.apply(objectives.init_weights)
        self.cross_modal_image_transform.apply(objectives.init_weights)

        self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])
        self.token_type_embeddings.apply(objectives.init_weights)

        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                if self.is_clip:
                    build_model(config["vit"], resolution_after=resolution_after, model_type=config["model_type"], vit_layernorm_shared=config["vit_layernorm_shared"], vit_remove_last=config["vit_remove_last"])
                elif self.is_swin:
                    getattr(swin, config["vit"])(pretrained=True, config=config,)
                else:
                    getattr(vit, config["vit"])(pretrained=True, img_size=resolution_after, model_type=config["model_type"],)
                if 'roberta' in config['tokenizer']:
                    RobertaModel.from_pretrained(config['tokenizer'])
                else:
                    BertModel.from_pretrained(config['tokenizer'])

            torch.distributed.barrier()

        if self.is_clip:
            self.vit_model = build_model(config["vit"], resolution_after=resolution_after, model_type=config["model_type"], vit_layernorm_shared=config["vit_layernorm_shared"], vit_remove_last=config["vit_remove_last"])
        elif self.is_swin:
            self.vit_model = getattr(swin, config["vit"])(
                pretrained=True, config=config,
            )
            self.avgpool = nn.AdaptiveAvgPool1d(1)
        else:
            self.vit_model = getattr(vit, config["vit"])(pretrained=True, img_size=resolution_after, model_type=config["model_type"],)

        if 'roberta' in config['tokenizer']:
            self.text_transformer = RobertaModel.from_pretrained(config['tokenizer'])
        else:
            self.text_transformer = BertModel.from_pretrained(config['tokenizer'])

        if not config["vit_layernorm_shared"] and config["vit_layernorm_init_from_vit"]:
            for ln in self.vit_model.visual.cross_modal_ln_separate:
                ln.weight.data = self.vit_model.visual.ln_post.weight.data
                ln.bias.data = self.vit_model.visual.ln_post.bias.data

        self.cross_modal_image_layers = nn.ModuleList([BertCrossLayer(bert_config) for _ in range(config['num_layers'])])
        self.cross_modal_image_layers.apply(objectives.init_weights)
        self.cross_modal_text_layers = nn.ModuleList([BertCrossLayer(bert_config) for _ in range(config['num_layers'])])
        self.cross_modal_text_layers.apply(objectives.init_weights)

        # Class token => Linear => Tanh
        self.cross_modal_image_pooler = heads.Pooler(config["hidden_size"])
        self.cross_modal_image_pooler.apply(objectives.init_weights)
        self.cross_modal_text_pooler = heads.Pooler(config["hidden_size"])
        self.cross_modal_text_pooler.apply(objectives.init_weights)

        # Temperature for image text contrastive learning
        self.temperature = nn.Parameter(torch.ones([]) * config['temperature']) 

        if config["loss_names"]["mlm"] > 0:
            # MLM Head weights don't tie with BERT Embedding weights. Train from scratch.
            self.mlm_score = heads.MLMHead(bert_config)
            self.mlm_score.apply(objectives.init_weights)

        if config["loss_names"]["itm"] > 0 or config["loss_names"]["itm_itc"] > 0 or config["loss_names"]["irtr_itm_itc"] > 0:
            self.itm_score = heads.ITMHead(config["hidden_size"] * 2)
            self.itm_score.apply(objectives.init_weights)

        if config["loss_names"]["itc"] > 0 or config["loss_names"]["itm_itc"] > 0 or config["loss_names"]["irtr_itm_itc"] > 0:
            self.cross_modal_itc_text_head = heads.ITCHead(config['hidden_size'], config['contrastive_hidden_size'])
            self.cross_modal_itc_text_head.apply(objectives.init_weights)
            self.cross_modal_itc_image_head = heads.ITCHead(config['hidden_size'], config['contrastive_hidden_size'])
            self.cross_modal_itc_image_head.apply(objectives.init_weights)

        hs = config["hidden_size"]

        # ===================== Initialize MT Components ===================== #
        self.num_layers_text = len(self.text_transformer.encoder.layer)
        if self.is_clip:
            self.num_layers_image = len(self.vit_model.visual.transformer.resblocks)
        elif self.is_vit:
            self.num_layers_image = len(self.vit_model.blocks)
        self.layer_embeddings_text = nn.Embedding(self.num_layers_text, config["hidden_size"])
        self.layer_embeddings_text.apply(objectives.init_weights)
        self.layer_embeddings_image = nn.Embedding(self.num_layers_image, config["hidden_size"])
        self.layer_embeddings_image.apply(objectives.init_weights)

        if config["manager_type"] in ['SAUE', 'AAUE']:
            self.cross_modal_text_manager_tower = nn.ModuleList([heads.Manager(config, config['managed_layers_text'], i) for i in range(config['num_layers'])])
            self.cross_modal_image_manager_tower = nn.ModuleList([heads.Manager(config, config['managed_layers_image'], i) for i in range(config['num_layers'])])
        else:
            raise NotImplementedError(f"manager_type {config['manager_type']} is not implemented")

        self.cross_modal_text_manager_tower.apply(objectives.init_weights)
        self.cross_modal_image_manager_tower.apply(objectives.init_weights)

        # ===================== Load Pretrained METER Weights ===================== 

        if (config["load_path"] != "" and not config["test_only"]):
            ckpt = torch.load(config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]

            if self.is_clip:
                state_dict = adapt_position_encoding(state_dict, after=resolution_after, patch_size=config['patch_size'])
            elif self.is_swin:
                state_dict = swin_adapt_position_encoding(state_dict, after=resolution_after, before=config['resolution_before'])
            else:
                state_dict['vit_model.pos_embed'] = resize_pos_embed(state_dict['vit_model.pos_embed'], self.vit_model.pos_embed, getattr(self.vit_model, 'num_tokens', 1), self.vit_model.patch_embed.grid_size)
            
            self.load_state_dict(state_dict, strict=False)

        # ===================== Downstream ===================== #
        
        hscale = config["head_hidden_scale"]
        if config["loss_names"]["vqa"] > 0:
            vs = config["vqav2_label_size"]
            if config["task_head_layers"] == 1:
                self.vqa_classifier = nn.Sequential(
                    nn.Linear(hs * 2, vs),
                )
            elif config["task_head_layers"] == 2:
                self.vqa_classifier = nn.Sequential(
                    nn.Linear(hs * 2, hs * 2 * hscale),
                    nn.LayerNorm(hs * 2 * hscale),
                    nn.GELU(),
                    nn.Linear(hs * 2 * hscale, vs),
                )
            self.vqa_classifier.apply(objectives.init_weights)

        if config["loss_names"]["nlvr2"] > 0:
            
            if config["task_head_layers"] == 1:
                self.nlvr2_classifier = nn.Sequential(
                    nn.Linear(hs * 4, 2),
                )
            elif config["task_head_layers"] == 2:
                self.nlvr2_classifier = nn.Sequential(
                    nn.Linear(hs * 4, int(hs * 2 * hscale)),
                    nn.LayerNorm(int(hs * 2 * hscale)),
                    nn.GELU(),
                    nn.Linear(int(hs * 2 * hscale), 2),
                )
            self.nlvr2_classifier.apply(objectives.init_weights)
            if config["nlvr2_drop_rate"] > 0:
                self.nlvr2_classifier_dropout = nn.Dropout(config['nlvr2_drop_rate'])
            emb_data = self.token_type_embeddings.weight.data
            self.token_type_embeddings = nn.Embedding(3, hs)
            self.token_type_embeddings.apply(objectives.init_weights)
            self.token_type_embeddings.weight.data[0, :] = emb_data[0, :]
            self.token_type_embeddings.weight.data[1, :] = emb_data[1, :]
            self.token_type_embeddings.weight.data[2, :] = emb_data[1, :]

        if config["loss_names"]["snli"] > 0:
            if config["task_head_layers"] == 1:
                self.snli_classifier = nn.Sequential(
                    nn.Linear(hs * 2, 3),
                )
            elif config["task_head_layers"] == 2:
                self.snli_classifier = nn.Sequential(
                    nn.Linear(hs * 2, hs * 2 * hscale),
                    nn.LayerNorm(hs * 2 * hscale),
                    nn.GELU(),
                    nn.Linear(hs * 2 * hscale, 3),
                )
            self.snli_classifier.apply(objectives.init_weights)

        if config["loss_names"]["irtr"] > 0:
            self.rank_output = nn.Linear(hs * 2, 1)
            self.rank_output.weight.data = self.itm_score.fc.weight.data[1:]
            self.rank_output.bias.data = self.itm_score.fc.bias.data[1:]
            for p in self.itm_score.parameters():
                p.requires_grad = False

        # ===================== load downstream (test_only) ======================

        if config["load_path"] != "" and config["test_only"]:
            ckpt = torch.load(config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            if self.is_clip:
                state_dict = adapt_position_encoding(state_dict, after=resolution_after, patch_size=config['patch_size'])
            elif self.is_swin:
                state_dict = swin_adapt_position_encoding(state_dict, after=resolution_after, before=config['resolution_before'])
            else:
                state_dict['vit_model.pos_embed'] = resize_pos_embed(state_dict['vit_model.pos_embed'], self.vit_model.pos_embed, getattr(self.vit_model, 'num_tokens', 1), self.vit_model.patch_embed.grid_size)
            self.load_state_dict(state_dict, strict=False)

        meter_utils.set_metrics(self)
        self.current_tasks = list()

    def get_cls_feats(self, text_feats, image_feats):
        cls_feats_text = self.cross_modal_text_pooler(text_feats)
        if self.is_clip:
            cls_feats_image = self.cross_modal_image_pooler(image_feats)
        elif self.is_swin:
            avg_image_feats = self.avgpool(image_feats.transpose(1, 2)).view(image_feats.size(0), 1, -1)
            cls_feats_image = self.cross_modal_image_pooler(avg_image_feats)
        else:
            cls_feats_image = self.cross_modal_image_pooler(image_feats)
        return torch.cat([cls_feats_text, cls_feats_image], dim=-1)

    def get_uni_modal_features(self, batch, fusion_features=False, itc=False):
        img = batch["image"][0]
        text_ids = batch[f"text_ids"]
        text_masks = batch[f"text_masks"]

        text_embeds = self.text_transformer.embeddings(input_ids=text_ids)
        input_shape = text_masks.size()
        extend_text_masks = self.text_transformer.get_extended_attention_mask(text_masks, input_shape, self.device)
        
        text_embedss, image_embedss = [], []
        split_index_text = self.num_layers_text - self.hparams.config['managed_layers_text']
        for index, layer in enumerate(self.text_transformer.encoder.layer):
            text_embeds = layer(text_embeds, extend_text_masks)[0]
            if index >= split_index_text:
                text_embedss.append(text_embeds)
        text_embedss = torch.stack(text_embedss, dim=0) # [N, B, L, D]

        split_index_image = self.num_layers_image - self.hparams.config['managed_layers_image']
        if self.is_clip:
            image_embeds = self.vit_model.visual.forward_pre(img.type(self.vit_model.dtype))
            for index, block in enumerate(self.vit_model.visual.transformer.resblocks):
                image_embeds = block(image_embeds)
                if index >= split_index_image:
                    image_embedss.append(image_embeds)
                    image_embedss[-1] = self.vit_model.visual.forward_post(image_embedss[-1].type(self.vit_model.dtype))
        else:
            image_embeds = self.vit_model.forward_pre(img)
            for index, block in enumerate(self.vit_model.blocks):
                image_embeds = block(image_embeds)
                if index >= split_index_image:
                    image_embedss.append(image_embeds)
                    image_embedss[-1] = self.vit_model.forward_post(image_embedss[-1])
        image_embedss = torch.stack(image_embedss, dim=0) # [N, B, L, D]
        
        if itc:
            unimodal_feats_text = F.normalize(self.cross_modal_itc_text_head(text_embedss[-1][:, 0, :]), dim=-1, p=2)
            unimodal_feats_image = F.normalize(self.cross_modal_itc_image_head(image_embedss[-1][:, 0, :]), dim=-1, p=2)
            if not fusion_features:
                ret = {
                    'unimodal_feats_text': unimodal_feats_text,
                    'unimodal_feats_image': unimodal_feats_image,
                }
                return ret

        # cross_modal transform
        text_embedss = self.cross_modal_text_transform(text_embedss)
        image_embedss = self.cross_modal_image_transform(image_embedss)
        
        if not itc:
            ret = {
                "text_embedss": text_embedss,
                "image_embedss": image_embedss,
                "text_ids": text_ids,
            }
        else:
            if fusion_features:
                ret = {
                    'unimodal_feats_text': unimodal_feats_text,
                    'unimodal_feats_image': unimodal_feats_image,
                    "text_embedss": text_embedss,
                    "image_embedss": image_embedss,
                    "text_ids": text_ids,
                }
        return ret

    def infer_text(
        self,
        batch,
        mask_text=False,
        itc=False,
    ):
        
        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_masks = batch[f"text_masks"]

        text_embeds = self.text_transformer.embeddings(input_ids=text_ids)
        input_shape = text_masks.size()
        extend_text_masks = self.text_transformer.get_extended_attention_mask(text_masks, input_shape, self.device)
        
        text_embedss = []
        split_index_text = self.num_layers_text - self.hparams.config['managed_layers_text']
        for index, layer in enumerate(self.text_transformer.encoder.layer):
            text_embeds = layer(text_embeds, extend_text_masks)[0]
            if index >= split_index_text:
                text_embedss.append(text_embeds)
        text_embedss = torch.stack(text_embedss, dim=0) # [N, B, L, D]

        if itc:
            unimodal_feats_text = F.normalize(self.cross_modal_itc_text_head(text_embedss[-1][:, 0, :]), dim=-1, p=2)

        
        text_embedss = self.cross_modal_text_transform(text_embedss)
        if itc:
            return text_embedss, extend_text_masks, unimodal_feats_text
        else:
            return text_embedss, extend_text_masks

    def infer_image(
        self,
        img,
        itc=False,
    ):
        image_embedss = []
        split_index_image = self.num_layers_image - self.hparams.config['managed_layers_image']
        
        if self.is_clip:
            image_embeds = self.vit_model.visual.forward_pre(img.type(self.vit_model.dtype))
            for index, block in enumerate(self.vit_model.visual.transformer.resblocks):
                image_embeds = block(image_embeds)
                if index >= split_index_image:
                    image_embedss.append(image_embeds)
                    image_embedss[-1] = self.vit_model.visual.forward_post(image_embedss[-1].type(self.vit_model.dtype))
        else:
            image_embeds = self.vit_model.forward_pre(img)
            for index, block in enumerate(self.vit_model.blocks):
                image_embeds = block(image_embeds)
                if index >= split_index_image:
                    image_embedss.append(image_embeds)
                    image_embedss[-1] = self.vit_model.forward_post(image_embedss[-1])
        image_embedss = torch.stack(image_embedss, dim=0) # [N, B, L, D]

        if itc:
            unimodal_feats_image = F.normalize(self.cross_modal_itc_image_head(image_embedss[-1][:, 0, :]), dim=-1, p=2)

        image_embedss = self.cross_modal_image_transform(image_embedss)
        if itc:
            return image_embedss, unimodal_feats_image
        else:
            return image_embedss

    def infer_fusion(
        self, 
        image_embedss, 
        text_embedss, 
        extend_text_masks, 
        image_token_type_idx=1,
        irtr_len_image=0,
        irtr_len_text=0,
    ):
        split_index_text = self.num_layers_text - self.hparams.config['managed_layers_text']
        split_index_image = self.num_layers_image - self.hparams.config['managed_layers_image']

        if irtr_len_image == irtr_len_text == 0:
            text_embedss = text_embedss.transpose(0, 1)
            image_embedss = image_embedss.transpose(0, 1)
        elif irtr_len_image > 0:
            text_embedss = text_embedss.transpose(0, 1)
        elif irtr_len_text > 0:
            image_embedss = image_embedss.transpose(0, 1)

        text_token_type_embeddings = self.token_type_embeddings(torch.zeros(1).long().to(self.device)).expand_as(text_embedss)
        text_layer_index = list(range(self.num_layers_text))[split_index_text:]
        text_token_layer_embeddings = self.layer_embeddings_text.weight[text_layer_index] # [N, D]
        if irtr_len_text > 0: # [N, L, D]
            text_token_layer_embeddings = text_token_layer_embeddings.unsqueeze(1).expand(-1, text_embedss.size(1), -1)
        else: # [B, N, L, D]
            text_token_layer_embeddings = text_token_layer_embeddings.unsqueeze(1).unsqueeze(0).expand(text_embedss.size(0), -1, text_embedss.size(2), -1)
        text_embedss = text_embedss + text_token_type_embeddings + text_token_layer_embeddings

        image_token_type_embeddings = self.token_type_embeddings(torch.zeros(1).long().to(self.device).fill_(image_token_type_idx)).expand_as(image_embedss)
        image_layer_index = list(range(self.num_layers_image))[split_index_image:]
        image_token_layer_embeddings = self.layer_embeddings_image.weight[image_layer_index] # [N, D]
        if irtr_len_image > 0: # [N, L, D]
            image_token_layer_embeddings = image_token_layer_embeddings.unsqueeze(1).expand(-1, image_embedss.size(1), -1)
        else: # [B, N, L, D]
            image_token_layer_embeddings = image_token_layer_embeddings.unsqueeze(1).unsqueeze(0).expand(image_embedss.size(0), -1, image_embedss.size(2), -1)
        image_embedss = image_embedss + image_token_type_embeddings + image_token_layer_embeddings

        if irtr_len_text > 0:
            _N, _L, _D = text_embedss.size()
            text_embedss = text_embedss.unsqueeze(0).expand(irtr_len_text, _N, _L, _D).contiguous()
        
        if irtr_len_image > 0:
            _N, _L, _D = image_embedss.size()
            image_embedss = image_embedss.unsqueeze(0).expand(irtr_len_image, _N, _L, _D).contiguous()

        image_masks = torch.ones((image_embedss.size(0), image_embedss.size(2)), dtype=torch.long, device=self.device)
        extend_image_masks = self.text_transformer.get_extended_attention_mask(image_masks, image_masks.size(), self.device)

        for manager_layer_index in range(self.hparams.config["num_layers"]):
            text_manager_tower = self.cross_modal_text_manager_tower[manager_layer_index]
            image_manager_tower = self.cross_modal_image_manager_tower[manager_layer_index]

            if manager_layer_index == 0:
                x1_ = text_manager_tower(text_embedss, 0, extend_text_masks, is_training=self.trainer.training) 
                y1_ = image_manager_tower(image_embedss, 0, extend_image_masks, is_training=self.trainer.training)                    
            else:
                x1_ = text_manager_tower(text_embedss, x1, extend_text_masks, extra_query=y1, is_training=self.trainer.training) 
                y1_ = image_manager_tower(image_embedss, y1, extend_image_masks, extra_query=x1, is_training=self.trainer.training)  

            x1 = self.cross_modal_text_layers[manager_layer_index](x1_, y1_, extend_text_masks, extend_image_masks)[0]
            y1 = self.cross_modal_image_layers[manager_layer_index](y1_, x1_, extend_image_masks, extend_text_masks)[0]      

        text_feats, image_feats = x1, y1
        cls_feats = self.get_cls_feats(text_feats, image_feats)

        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
        }
        
        return ret    

    def infer(
        self,
        batch,
        mask_text=False,
        mask_image=False,
        image_token_type_idx=1,
        img=None,
        irtr_len_image=0,
    ):
        if img is None:
            if f"image_{image_token_type_idx - 1}" in batch:
                imgkey = f"image_{image_token_type_idx - 1}"
            else:
                imgkey = "image"
            img = batch[imgkey][0]
        
        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_masks = batch[f"text_masks"]

        text_embeds = self.text_transformer.embeddings(input_ids=text_ids)
        input_shape = text_masks.size()
        extend_text_masks = self.text_transformer.get_extended_attention_mask(text_masks, input_shape, self.device)

        text_embedss, image_embedss = [], []
        split_index_text = self.num_layers_text - self.hparams.config['managed_layers_text']
        for index, layer in enumerate(self.text_transformer.encoder.layer):
            text_embeds = layer(text_embeds, extend_text_masks)[0]
            if index >= split_index_text:
                text_embedss.append(text_embeds)
        text_embedss = torch.stack(text_embedss, dim=1) # [B, N, L, D]
        
        split_index_image = self.num_layers_image - self.hparams.config['managed_layers_image']
        if self.is_clip:
            image_embeds = self.vit_model.visual.forward_pre(img.type(self.vit_model.dtype))
            for index, block in enumerate(self.vit_model.visual.transformer.resblocks):
                image_embeds = block(image_embeds)
                if index >= split_index_image:
                    image_embedss.append(image_embeds)
                    image_embedss[-1] = self.vit_model.visual.forward_post(image_embedss[-1].type(self.vit_model.dtype))
        else:
            image_embeds = self.vit_model.forward_pre(img)
            for index, block in enumerate(self.vit_model.blocks):
                image_embeds = block(image_embeds)
                if index >= split_index_image:
                    image_embedss.append(image_embeds)
                    image_embedss[-1] = self.vit_model.forward_post(image_embedss[-1])
        image_embedss = torch.stack(image_embedss, dim=1) # [B, N, L, D]
        
        if self.hparams.config["num_layers"] == 0:
            cls_feats = self.get_cls_feats(text_embedss[:, -1], image_embedss[:, -1])

            ret = {
                "text_feats": text_embedss[:, -1],
                "image_feats": image_embedss[:, -1],
                "cls_feats": cls_feats,
                "text_ids": text_ids,
            }
        
            return ret

        text_embedss = self.cross_modal_text_transform(text_embedss)
        image_embedss = self.cross_modal_image_transform(image_embedss)

        text_token_type_embeddings = self.token_type_embeddings(torch.zeros(1).long().to(self.device)).expand_as(text_embedss)
        text_layer_index = list(range(self.num_layers_text))[split_index_text:]
        text_token_layer_embeddings = self.layer_embeddings_text.weight[text_layer_index] # [N, D]
        text_token_layer_embeddings = text_token_layer_embeddings.unsqueeze(1).unsqueeze(0).expand(text_embedss.size(0), -1, text_embedss.size(2), -1)
        text_embedss = text_embedss + text_token_type_embeddings + text_token_layer_embeddings

        image_token_type_embeddings = self.token_type_embeddings(torch.zeros(1).long().to(self.device).fill_(image_token_type_idx)).expand_as(image_embedss)
        image_layer_index = list(range(self.num_layers_image))[split_index_image:]
        image_token_layer_embeddings = self.layer_embeddings_image.weight[image_layer_index] # [N, D]
        image_token_layer_embeddings = image_token_layer_embeddings.unsqueeze(1).unsqueeze(0).expand(image_embedss.size(0), -1, image_embedss.size(2), -1)
        image_embedss = image_embedss + image_token_type_embeddings + image_token_layer_embeddings
        
        if irtr_len_image > 0:
            _B, _N, _L, _D = image_embedss.size()
            image_embedss = image_embedss.unsqueeze(1).expand(_B, irtr_len_image, _N, _L, _D).contiguous().view(-1, _N, _L, _D)

        image_masks = torch.ones((image_embedss.size(0), image_embedss.size(2)), dtype=torch.long, device=self.device)
        extend_image_masks = self.text_transformer.get_extended_attention_mask(image_masks, image_masks.size(), self.device)

        for manager_layer_index in range(self.hparams.config["num_layers"]):
            text_manager_tower = self.cross_modal_text_manager_tower[manager_layer_index]
            image_manager_tower = self.cross_modal_image_manager_tower[manager_layer_index]

            if manager_layer_index == 0:
                x1_ = text_manager_tower(text_embedss, 0, extend_text_masks, is_training=self.trainer.training) 
                y1_ = image_manager_tower(image_embedss, 0, extend_image_masks, is_training=self.trainer.training)                    
            else:
                x1_ = text_manager_tower(text_embedss, x1, extend_text_masks, extra_query=y1, is_training=self.trainer.training) 
                y1_ = image_manager_tower(image_embedss, y1, extend_image_masks, extra_query=x1, is_training=self.trainer.training)  

            x1 = self.cross_modal_text_layers[manager_layer_index](x1_, y1_, extend_text_masks, extend_image_masks)[0]
            y1 = self.cross_modal_image_layers[manager_layer_index](y1_, x1_, extend_image_masks, extend_text_masks)[0]      
        
        text_feats, image_feats = x1, y1
        cls_feats = self.get_cls_feats(text_feats, image_feats)

        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "text_ids": text_ids,
        }
        
        return ret

    def forward(self, batch, split):
        ret = dict()
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        # Masked Language Modeling
        if "mlm" in self.current_tasks:
            ret.update(objectives.compute_mlm(self, batch, split))

        # Image Text Matching
        if "itm" in self.current_tasks:
            ret.update(objectives.compute_itm(self, batch, split))
        
        if "itc" in self.current_tasks:
            ret.update(objectives.compute_itc(self, batch, split))
        
        if "itm_itc" in self.current_tasks:
            ret.update(objectives.compute_itm_itc(self, batch, split, pretrain=True))

        if "irtr_itm_itc" in self.current_tasks:
            ret.update(objectives.compute_itm_itc(self, batch, split, pretrain=False))

        # Visual Question Answering
        if "vqa" in self.current_tasks:
            ret.update(objectives.compute_vqa(self, batch, split))

        # Natural Language for Visual Reasoning 2
        if "nlvr2" in self.current_tasks:
            ret.update(objectives.compute_nlvr2(self, batch, split))

        # SNLI Visual Entailment
        if "snli" in self.current_tasks:
            ret.update(objectives.compute_snli(self, batch, split))

        # Image Retrieval and Text Retrieval
        if "irtr" in self.current_tasks:
            ret.update(objectives.compute_irtr(self, batch, split))

        return ret

    def training_step(self, batch, batch_idx):
        meter_utils.set_task(self)
        output = self(batch, 'train')
        total_loss = sum([v for k, v in output.items() if "loss" in k])

        return total_loss

    def training_epoch_end(self, outs):
        meter_utils.epoch_wrapup(self, 'train')

    def validation_step(self, batch, batch_idx):
        meter_utils.set_task(self)
        output = self(batch, 'val')

    def validation_epoch_end(self, outs):
        if self.jump_val_first_for_irtr_itm_irc and "irtr_itm_itc" in self.hparams.config["group_name"]:
            old_get_recall_metric = self.hparams.config["get_recall_metric"]
            self.hparams.config["get_recall_metric"] = False
            meter_utils.epoch_wrapup(self, 'val')
            self.hparams.config["get_recall_metric"] = old_get_recall_metric
            self.jump_val_first_for_irtr_itm_irc = False
        else:
            meter_utils.epoch_wrapup(self, 'val')

    def test_step(self, batch, batch_idx):
        meter_utils.set_task(self)
        output = self(batch, 'test')
        ret = dict()

        if self.hparams.config["loss_names"]["vqa"] > 0:
            ret.update(objectives.vqa_test_step(self, batch, output))

        return ret

    def test_epoch_end(self, outs):
        model_name = self.hparams.config["load_path"].split("/")[-2]
        checkpoint_name = self.hparams.config["load_path"].split("/")[-1][:-5]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            objectives.vqa_test_wrapup(outs, f"{model_name}_{checkpoint_name}", self.hparams.config["log_dir"])
        meter_utils.epoch_wrapup(self, 'test')

    def configure_optimizers(self):
        # Optimizer: AdamW; Scheduler: linear_schedule_with_warmup
        # Parameters for cross-modal and each task head will be multiply by lr_mult_cross_modal or lr_mult_head
        # New task heads need to enroll here.
        return meter_utils.set_schedule(self)
