import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from tqdm import tqdm
import functools

from torch.utils.data.distributed import DistributedSampler
from einops import rearrange
import torch.distributed as dist

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

# pre-train
def compute_mlm(pl_module, batch, split):
    infer = pl_module.infer(batch, mask_text=True, mask_image=False)
    mlm_logits = pl_module.mlm_score(infer["text_feats"])
    mlm_labels = batch[f"text_labels_mlm"]

    mlm_loss = F.cross_entropy(
        mlm_logits.view(-1, pl_module.hparams.config["vocab_size"]),
        mlm_labels.view(-1),
        ignore_index=-100,
    )

    ret = {
        "mlm_loss": mlm_loss,
        "mlm_logits": mlm_logits,
        "mlm_labels": mlm_labels,
        "mlm_ids": infer["text_ids"],
    }

    loss_name = 'mlm'

    loss = getattr(pl_module, f"{split}_{loss_name}_loss")(ret["mlm_loss"])
    acc = getattr(pl_module, f"{split}_{loss_name}_accuracy")(
        ret["mlm_logits"], ret["mlm_labels"]
    )
    pl_module.log(f"{split}/{loss_name}/loss", loss)
    pl_module.log(f"{split}/{loss_name}/accuracy", acc)

    return ret

def compute_itm(pl_module, batch, split):
    pos_len = len(batch["text"]) // 2
    neg_len = len(batch["text"]) - pos_len
    itm_labels = torch.cat([torch.ones(pos_len), torch.zeros(neg_len)]).to(
        pl_module.device
    )
    itm_labels = itm_labels[torch.randperm(itm_labels.size(0))]

    itm_images = [
        torch.stack(
            [
                ti if itm_labels[i] == 1 else fi
                for i, (ti, fi) in enumerate(zip(bti, bfi))
            ]
        )
        for bti, bfi in zip(batch["image"], batch["false_image_0"])
    ]

    batch = {k: v for k, v in batch.items()}
    batch["image"] = itm_images

    infer = pl_module.infer(batch)

    itm_logits = pl_module.itm_score(infer["cls_feats"])
    itm_loss = F.cross_entropy(itm_logits, itm_labels.long())

    ret = {
        "itm_loss": itm_loss,
        "itm_logits": itm_logits,
        "itm_labels": itm_labels,
    }
    
    loss_name = 'itm'

    loss = getattr(pl_module, f"{split}_{loss_name}_loss")(ret["itm_loss"])
    acc = getattr(pl_module, f"{split}_{loss_name}_accuracy")(
        ret["itm_logits"], ret["itm_labels"]
    )
    pl_module.log(f"{split}/{loss_name}/loss", loss)
    pl_module.log(f"{split}/{loss_name}/accuracy", acc)

    return ret

def compute_itc(pl_module, batch, split):
    assert batch["image"][0].size(0) == len(batch["text"])
    bs, rank = len(batch["text"]), torch.distributed.get_rank()
    
    with torch.no_grad():
        pl_module.temperature.clamp_(0.001, 0.5)

    infer = pl_module.get_uni_modal_features(batch, itc=True)
    unimodal_feats_text = infer['unimodal_feats_text']
    unimodal_feats_image = infer['unimodal_feats_image']

    gather_unimodal_feats_text = pl_module.all_gather(unimodal_feats_text, sync_grads=True)
    gather_unimodal_feats_image = pl_module.all_gather(unimodal_feats_image, sync_grads=True)
    
    gather_unimodal_feats_text = gather_unimodal_feats_text.view((-1,) + (gather_unimodal_feats_text.shape)[2:])
    gather_unimodal_feats_image = gather_unimodal_feats_image.view((-1,) + (gather_unimodal_feats_image.shape)[2:])
    
    logit_scale = torch.log(1 / pl_module.temperature).exp()
    itc_logits_i2t = logit_scale * unimodal_feats_image @ gather_unimodal_feats_text.t()
    itc_logits_t2i = logit_scale * unimodal_feats_text @ gather_unimodal_feats_image.t()

    itc_labels = torch.arange(bs).to(pl_module.device)
    itc_labels = itc_labels + bs * rank
    i2t_loss = F.cross_entropy(itc_logits_i2t, itc_labels)
    t2i_loss = F.cross_entropy(itc_logits_t2i, itc_labels)
    itc_loss = (i2t_loss + t2i_loss) / 2

    ret = {
        "itc_loss": itc_loss,
    }
    
    loss_name = 'itc'

    if pl_module.hparams.config["num_layers"] == 0:
        loss_name = 'irtr_itm'

    loss = getattr(pl_module, f"{split}_{loss_name}_loss")(ret["itc_loss"])
    pl_module.log(f"{split}/{loss_name}/loss", loss)

    return ret

def compute_itm_itc(pl_module, batch, split, pretrain=False):
    # REMEMBER: No need to draw false images for image text matching in data preprocessing.
    assert batch["image"][0].size(0) == len(batch["text"])
    bs, rank = len(batch["text"]), torch.distributed.get_rank()
    
    # forward the positive image-text pair
    with torch.no_grad():
        pl_module.temperature.clamp_(0.001, 0.5)

    infer = pl_module.get_uni_modal_features(batch, fusion_features=True, itc=True)
    unimodal_feats_text = infer['unimodal_feats_text']
    unimodal_feats_image = infer['unimodal_feats_image']

    gather_unimodal_feats_text = pl_module.all_gather(unimodal_feats_text, sync_grads=True)
    gather_unimodal_feats_image = pl_module.all_gather(unimodal_feats_image, sync_grads=True)

    gather_unimodal_feats_text = gather_unimodal_feats_text.view((-1,) + (gather_unimodal_feats_text.shape)[2:])
    gather_unimodal_feats_image = gather_unimodal_feats_image.view((-1,) + (gather_unimodal_feats_image.shape)[2:])
    
    logit_scale = torch.log(1 / pl_module.temperature).exp()
    itc_logits_i2t = logit_scale * unimodal_feats_image @ gather_unimodal_feats_text.t()
    itc_logits_t2i = logit_scale * unimodal_feats_text @ gather_unimodal_feats_image.t()

    if pretrain:
        itc_labels = torch.arange(bs).to(pl_module.device)
        itc_labels = itc_labels + bs * rank
    else:
        idx = torch.LongTensor(batch["img_index"]).view(-1, 1).to(pl_module.device)
        idx_all = pl_module.all_gather(idx).view(-1, 1)
        assert idx_all.size(0) == gather_unimodal_feats_image.size(0)
        idx_all = torch.eq(idx_all, idx_all.t()).to(pl_module.device)
        idx_all = idx_all[bs * rank:bs * (rank+1)]
        pos_idx = idx_all.float()
        assert pos_idx.size(0) == len(idx)
        itc_labels = pos_idx / pos_idx.sum(1, keepdim=True)

    i2t_loss = F.cross_entropy(itc_logits_i2t, itc_labels)
    t2i_loss = F.cross_entropy(itc_logits_t2i, itc_labels)
    itc_loss = (i2t_loss + t2i_loss) / 2

    if pretrain:    
        loss_name = 'itc'
    else:
        loss_name = 'irtr_itc'
    
    loss = getattr(pl_module, f"{split}_{loss_name}_loss")(itc_loss)
    pl_module.log(f"{split}/{loss_name}/loss", loss)

    # sample hard negative images for image text matching from image text contrastive logits
    if pl_module.hparams.config["gather_global_negative"]:
        # select a negative image for each text
        with torch.no_grad():
            weights_i2t = F.softmax(itc_logits_i2t, dim=-1)
            weights_t2i = F.softmax(itc_logits_t2i, dim=-1)
            if pretrain:
                weights_i2t[:, bs * rank:bs * (rank+1)].fill_diagonal_(0)
                weights_t2i[:, bs * rank:bs * (rank+1)].fill_diagonal_(0)
            else:
                weights_i2t.masked_fill_(idx_all, 0)
                weights_t2i.masked_fill_(idx_all, 0)
        
        if pl_module.hparams.config["gather_all_inputs"]:
            gather_image_ids = pl_module.all_gather(batch["image"][0])
            gather_image_ids = gather_image_ids.view((-1,) + (gather_image_ids.shape)[2:])
        else:
            global_image_embedss = pl_module.all_gather(infer['image_embedss'].transpose(0, 1), sync_grads=True).view(-1, infer['image_embedss'].size(0), infer['image_embedss'].size(2), infer['image_embedss'].size(3))
        
        neg_idxes = []
        for b in range(bs):
            try:
                neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            except:
                neg_idx = torch.multinomial(weights_t2i[b] + 1e-5, 1).item()
            neg_idxes.append(neg_idx)

        if pl_module.hparams.config["gather_all_inputs"]:
            image_ids_neg = []
            for neg_idx in neg_idxes:
                image_ids_neg.append(gather_image_ids[neg_idx])
            image_ids_neg = torch.stack(image_ids_neg, dim=0)
        else:
            image_embeds_neg = []
            for neg_idx in neg_idxes:
                image_embeds_neg.append(global_image_embedss[neg_idx])
            image_embeds_neg = torch.stack(image_embeds_neg, dim=1)   
        # del global_image_embedss

        # select a negative text for each image
        if pl_module.hparams.config["gather_all_inputs"]:
            gather_text_ids = pl_module.all_gather(batch["text_ids"])
            gather_text_masks = pl_module.all_gather(batch["text_masks"])
            gather_text_ids = gather_text_ids.view((-1,) + (gather_text_ids.shape)[2:])
            gather_text_masks = gather_text_masks.view((-1,) + (gather_text_masks.shape)[2:])
        else:
            global_text_embedss = pl_module.all_gather(infer['text_embedss'].transpose(0, 1), sync_grads=True).view(-1, infer['text_embedss'].size(0), infer['text_embedss'].size(2), infer['text_embedss'].size(3))
            global_text_masks = pl_module.all_gather(batch["text_masks"]).view(-1, batch["text_masks"].size(1))

        neg_idxes = []
        for b in range(bs):
            try:
                neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            except:
                neg_idx = torch.multinomial(weights_i2t[b] + 1e-5, 1).item()
            neg_idxes.append(neg_idx)

        if pl_module.hparams.config["gather_all_inputs"]:
            text_ids_neg, text_masks_neg = [], []
            for neg_idx in neg_idxes:
                text_ids_neg.append(gather_text_ids[neg_idx])
                text_masks_neg.append(gather_text_masks[neg_idx])
            text_ids_neg = torch.stack(text_ids_neg, dim=0)
            text_masks_neg = torch.stack(text_masks_neg, dim=0)
        else:
            text_embeds_neg, text_masks_neg = [], []
            for neg_idx in neg_idxes:
                text_embeds_neg.append(global_text_embedss[neg_idx])
                text_masks_neg.append(global_text_masks[neg_idx])
            text_embeds_neg = torch.stack(text_embeds_neg, dim=1)
            text_masks_neg = torch.stack(text_masks_neg, dim=0) 
        # del global_text_embedss, global_text_masks
    else:
        # select a negative image for each text
        with torch.no_grad():
            weights_i2t = F.softmax(itc_logits_i2t[:, bs * rank:bs * (rank+1)], dim=-1)
            weights_t2i = F.softmax(itc_logits_t2i[:, bs * rank:bs * (rank+1)], dim=-1)
            if pretrain:
                weights_i2t.fill_diagonal_(0)
                weights_t2i.fill_diagonal_(0)
            else:
                mask = torch.eq(idx, idx.t()).to(pl_module.device)
                weights_i2t.masked_fill_(mask, 0)
                weights_t2i.masked_fill_(mask, 0)
        
        neg_idxes = []
        for b in range(bs):
            try:
                neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            except:
                neg_idx = torch.multinomial(weights_t2i[b] + 1e-5, 1).item()
            neg_idxes.append(neg_idx)
        
        if pl_module.hparams.config["gather_all_inputs"]:
            image_ids_neg = []
            for neg_idx in neg_idxes:
                image_ids_neg.append(batch["image"][0][neg_idx])
            image_ids_neg = torch.stack(image_ids_neg, dim=0)
        else:
            image_embeds_neg = []
            for neg_idx in neg_idxes:
                image_embeds_neg.append(infer['image_embedss'][:, neg_idx])
            image_embeds_neg = torch.stack(image_embeds_neg, dim=1)

        # select a negative text for each image
        neg_idxes = []
        for b in range(bs):
            try:
                neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            except:
                neg_idx = torch.multinomial(weights_i2t[b] + 1e-5, 1).item()
            neg_idxes.append(neg_idx)
        
        if pl_module.hparams.config["gather_all_inputs"]:
            text_ids_neg, text_masks_neg = [], []
            for neg_idx in neg_idxes:
                text_ids_neg.append(batch["text_ids"][neg_idx])
                text_masks_neg.append(batch["text_masks"][neg_idx])
            text_ids_neg = torch.stack(text_ids_neg, dim=0)
            text_masks_neg = torch.stack(text_masks_neg, dim=0)
        else:
            text_embeds_neg, text_masks_neg = [], []
            for neg_idx in neg_idxes:
                text_embeds_neg.append(infer['text_embedss'][:, neg_idx])
                text_masks_neg.append(batch["text_masks"][neg_idx])
            text_embeds_neg = torch.stack(text_embeds_neg, dim=1)
            text_masks_neg = torch.stack(text_masks_neg, dim=0)
    
    # pack the image-text pairs for fusion, which is 3 x batch_size 
    if pl_module.hparams.config["gather_all_inputs"]:
        text_ids = torch.cat([batch["text_ids"], batch["text_ids"], text_ids_neg], dim=0)
        text_masks = torch.cat([batch["text_masks"], batch["text_masks"], text_masks_neg], dim=0)
        image_ids = torch.cat([batch["image"][0], image_ids_neg, batch["image"][0]], dim=0)
        infer = pl_module.infer({ "text_ids": text_ids, "text_masks": text_masks, "image": [image_ids]}, )
        cls_feats = infer["cls_feats"]
    else:
        text_embedss = torch.cat([infer['text_embedss'], infer['text_embedss'], text_embeds_neg], dim=1)     
        text_masks = torch.cat([batch["text_masks"], batch["text_masks"], text_masks_neg], dim=0)
        extend_text_masks = pl_module.text_transformer.get_extended_attention_mask(text_masks, text_masks.size(), pl_module.device)  
        image_embedss = torch.cat([infer['image_embedss'], image_embeds_neg, infer['image_embedss']], dim=1)
        infer = pl_module.infer_fusion(image_embedss, text_embedss, extend_text_masks)
        cls_feats = infer["cls_feats"]

    itm_labels = torch.cat([
        torch.ones(bs, dtype=torch.long),
        torch.zeros(2 * bs, dtype=torch.long)]
    ).to(pl_module.device)

    itm_logits = pl_module.itm_score(cls_feats)
    itm_loss = F.cross_entropy(itm_logits, itm_labels)

    ret = {
        "itc_loss": itc_loss,
        "itm_loss": itm_loss,
        "itm_logits": itm_logits,
        "itm_labels": itm_labels,
    }

    if pretrain:    
        loss_name = 'itm'
    else:
        loss_name = 'irtr_itm'
    loss = getattr(pl_module, f"{split}_{loss_name}_loss")(ret["itm_loss"])
    acc = getattr(pl_module, f"{split}_{loss_name}_accuracy")(
        ret["itm_logits"], ret["itm_labels"]
    )
    pl_module.log(f"{split}/{loss_name}/loss", loss)
    pl_module.log(f"{split}/{loss_name}/accuracy", acc)
    return ret

def compute_itm_itc_meter(pl_module, batch, split, pretrain=False):
    # REMEMBER: No need to draw false images for image text matching in data preprocessing.
    assert batch["image"][0].size(0) == len(batch["text"])
    bs, rank = len(batch["text"]), torch.distributed.get_rank()
    
    # forward the positive image-text pair
    with torch.no_grad():
        pl_module.temperature.clamp_(0.001, 0.5)

    infer = pl_module.get_uni_modal_features(batch, fusion_features=True, itc=True)
    unimodal_feats_text = infer['unimodal_feats_text']
    unimodal_feats_image = infer['unimodal_feats_image']
    
    gather_unimodal_feats_text = pl_module.all_gather(unimodal_feats_text, sync_grads=True)
    gather_unimodal_feats_image = pl_module.all_gather(unimodal_feats_image, sync_grads=True)
   
    gather_unimodal_feats_text = gather_unimodal_feats_text.view((-1,) + (gather_unimodal_feats_text.shape)[2:])
    gather_unimodal_feats_image = gather_unimodal_feats_image.view((-1,) + (gather_unimodal_feats_image.shape)[2:])
    
    logit_scale = torch.log(1 / pl_module.temperature).exp()
    itc_logits_i2t = logit_scale * unimodal_feats_image @ gather_unimodal_feats_text.t()
    itc_logits_t2i = logit_scale * unimodal_feats_text @ gather_unimodal_feats_image.t()

    if pretrain:
        itc_labels = torch.arange(bs).to(pl_module.device)
        itc_labels = itc_labels + bs * rank
    else:
        idx = torch.LongTensor(batch["img_index"]).view(-1, 1).to(pl_module.device)
        idx_all = pl_module.all_gather(idx).view(-1, 1)
        assert idx_all.size(0) == gather_unimodal_feats_image.size(0)
        idx_all = torch.eq(idx_all, idx_all.t()).to(pl_module.device)
        idx_all = idx_all[bs * rank:bs * (rank+1)]
        pos_idx = idx_all.float()
        assert pos_idx.size(0) == len(idx)
        itc_labels = pos_idx / pos_idx.sum(1, keepdim=True)

    i2t_loss = F.cross_entropy(itc_logits_i2t, itc_labels)
    t2i_loss = F.cross_entropy(itc_logits_t2i, itc_labels)
    itc_loss = (i2t_loss + t2i_loss) / 2

    if pretrain:    
        loss_name = 'itc'
    else:
        loss_name = 'irtr_itc'
    
    loss = getattr(pl_module, f"{split}_{loss_name}_loss")(itc_loss)
    pl_module.log(f"{split}/{loss_name}/loss", loss)

    # sample hard negative images for image text matching from image text contrastive logits
    if pl_module.hparams.config["gather_global_negative"]:
        # select a negative image for each text
        with torch.no_grad():
            weights_i2t = F.softmax(itc_logits_i2t, dim=-1)
            weights_t2i = F.softmax(itc_logits_t2i, dim=-1)
            if pretrain:
                weights_i2t[:, bs * rank:bs * (rank+1)].fill_diagonal_(0)
                weights_t2i[:, bs * rank:bs * (rank+1)].fill_diagonal_(0)
            else:
                weights_i2t.masked_fill_(idx_all, 0)
                weights_t2i.masked_fill_(idx_all, 0)
        
        if pl_module.hparams.config["gather_all_inputs"]:
            gather_image_ids = pl_module.all_gather(batch["image"][0])
            gather_image_ids = gather_image_ids.view((-1,) + (gather_image_ids.shape)[2:])
        else:
            global_image_embeds = pl_module.all_gather(infer['image_embeds'], sync_grads=True).view(-1, infer['image_embeds'].size(1), infer['image_embeds'].size(2))
        
        neg_idxes = []
        for b in range(bs):
            try:
                neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            except:
                neg_idx = torch.multinomial(weights_t2i[b] + 1e-5, 1).item()
            neg_idxes.append(neg_idx)

        if pl_module.hparams.config["gather_all_inputs"]:
            image_ids_neg = []
            for neg_idx in neg_idxes:
                image_ids_neg.append(gather_image_ids[neg_idx])
            image_ids_neg = torch.stack(image_ids_neg, dim=0)
        else:
            image_embeds_neg = []
            for neg_idx in neg_idxes:
                image_embeds_neg.append(global_image_embeds[neg_idx])
            image_embeds_neg = torch.stack(image_embeds_neg, dim=0)   
        # del global_image_embeds

        # select a negative text for each image
        if pl_module.hparams.config["gather_all_inputs"]:
            gather_text_ids = pl_module.all_gather(batch["text_ids"])
            gather_text_masks = pl_module.all_gather(batch["text_masks"])
            gather_text_ids = gather_text_ids.view((-1,) + (gather_text_ids.shape)[2:])
            gather_text_masks = gather_text_masks.view((-1,) + (gather_text_masks.shape)[2:])
        else:
            global_text_embeds = pl_module.all_gather(infer['text_embeds'], sync_grads=True).view(-1, infer['text_embeds'].size(1), infer['text_embeds'].size(2))
            global_text_masks = pl_module.all_gather(batch["text_masks"]).view(-1, batch["text_masks"].size(1))

        neg_idxes = []
        for b in range(bs):
            try:
                neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            except:
                neg_idx = torch.multinomial(weights_i2t[b] + 1e-5, 1).item()
            neg_idxes.append(neg_idx)

        if pl_module.hparams.config["gather_all_inputs"]:
            text_ids_neg, text_masks_neg = [], []
            for neg_idx in neg_idxes:
                text_ids_neg.append(gather_text_ids[neg_idx])
                text_masks_neg.append(gather_text_masks[neg_idx])
            text_ids_neg = torch.stack(text_ids_neg, dim=0)
            text_masks_neg = torch.stack(text_masks_neg, dim=0)
        else:
            text_embeds_neg, text_masks_neg = [], []
            for neg_idx in neg_idxes:
                text_embeds_neg.append(global_text_embeds[neg_idx])
                text_masks_neg.append(global_text_masks[neg_idx])
            text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
            text_masks_neg = torch.stack(text_masks_neg, dim=0)         
        # del global_text_embeds, global_text_masks
    else:
        # select a negative image for each text
        with torch.no_grad():
            weights_i2t = F.softmax(itc_logits_i2t[:, bs * rank:bs * (rank+1)], dim=-1)
            weights_t2i = F.softmax(itc_logits_t2i[:, bs * rank:bs * (rank+1)], dim=-1)
            if pretrain:
                weights_i2t.fill_diagonal_(0)
                weights_t2i.fill_diagonal_(0)
            else:
                mask = torch.eq(idx, idx.t()).to(pl_module.device)
                weights_i2t.masked_fill_(mask, 0)
                weights_t2i.masked_fill_(mask, 0)
        
        image_embeds_neg = []
        for b in range(bs):
            try:
                neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            except:
                neg_idx = torch.multinomial(weights_t2i[b] + 1e-5, 1).item()
            image_embeds_neg.append(infer['image_embeds'][neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)   

        # select a negative text for each image
        text_embeds_neg = []
        text_masks_neg = []
        for b in range(bs):
            try:
                neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            except:
                neg_idx = torch.multinomial(weights_i2t[b] + 1e-5, 1).item()
            text_embeds_neg.append(infer['text_embeds'][neg_idx])
            text_masks_neg.append(batch["text_masks"][neg_idx])
        text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
        text_masks_neg = torch.stack(text_masks_neg, dim=0)
    
    # pack the image-text pairs for fusion, which is 3 x batch_size 
    text_embeds = torch.cat([infer['text_embeds'], infer['text_embeds'], text_embeds_neg], dim=0)
    text_masks = torch.cat([batch["text_masks"], batch["text_masks"], text_masks_neg], dim=0)
    extend_text_masks = pl_module.text_transformer.get_extended_attention_mask(text_masks, text_masks.size(), pl_module.device)  

    image_embeds = torch.cat([infer['image_embeds'], image_embeds_neg, infer['image_embeds']], dim=0)

    # fusion 
    cls_feats = pl_module.infer_fusion(image_embeds, text_embeds, extend_text_masks)['cls_feats']

    itm_labels = torch.cat([
        torch.ones(bs, dtype=torch.long),
        torch.zeros(2 * bs, dtype=torch.long)]
    ).to(pl_module.device)

    itm_logits = pl_module.itm_score(cls_feats)
    itm_loss = F.cross_entropy(itm_logits, itm_labels)

    ret = {
        "itc_loss": itc_loss,
        "itm_loss": itm_loss,
        "itm_logits": itm_logits,
        "itm_labels": itm_labels,
    }

    if pretrain:    
        loss_name = 'itm'
    else:
        loss_name = 'irtr_itm'
    loss = getattr(pl_module, f"{split}_{loss_name}_loss")(ret["itm_loss"])
    acc = getattr(pl_module, f"{split}_{loss_name}_accuracy")(
        ret["itm_logits"], ret["itm_labels"]
    )
    pl_module.log(f"{split}/{loss_name}/loss", loss)
    pl_module.log(f"{split}/{loss_name}/accuracy", acc)
    return ret

# fine-tune
def compute_snli(pl_module, batch, split):
    infer = pl_module.infer(batch) 
    snli_logits = pl_module.snli_classifier(infer["cls_feats"])

    snli_labels = batch["labels"]
    snli_labels = torch.tensor(snli_labels).to(pl_module.device).long()
    snli_loss = F.cross_entropy(snli_logits, snli_labels.view(-1))

    ret = {
        "snli_loss": snli_loss,
        "snli_logits": snli_logits,
        "snli_labels": snli_labels,
    }

    loss_name = 'snli'

    if split == "train":
        loss = getattr(pl_module, f"{split}_{loss_name}_loss")(ret["snli_loss"])
        acc = getattr(pl_module, f"{split}_{loss_name}_accuracy")(
            ret["snli_logits"], ret["snli_labels"]
        )
        pl_module.log(f"{split}/{loss_name}/loss", loss)
        pl_module.log(f"{split}/{loss_name}/accuracy", acc)
    else:
        val_batches = [i for i, n in enumerate(batch["table_name"]) if "dev" in n]
        test_batches = [i for i, n in enumerate(batch["table_name"]) if "test" in n]

        if val_batches:
            val_loss = getattr(pl_module, f"val_{loss_name}_loss")(
                F.cross_entropy(
                    ret["snli_logits"][val_batches], ret["snli_labels"][val_batches]
                )
            )
            val_acc = getattr(pl_module, f"val_{loss_name}_accuracy")(
                ret["snli_logits"][val_batches], ret["snli_labels"][val_batches]
            )
            pl_module.log(f"val/snli/loss", val_loss)
            pl_module.log(f"val/snli/accuracy", val_acc)

        if test_batches:
            test_loss = getattr(pl_module, f"test_{loss_name}_loss")(
                F.cross_entropy(
                    ret["snli_logits"][test_batches], ret["snli_labels"][test_batches]
                )
            )
            test_acc = getattr(pl_module, f"test_{loss_name}_accuracy")(
                ret["snli_logits"][test_batches], ret["snli_labels"][test_batches]
            )
            pl_module.log(f"test/snli/loss", test_loss)
            pl_module.log(f"test/snli/accuracy", test_acc)

    return ret

def compute_vqa(pl_module, batch, split):
    infer = pl_module.infer(batch)
    
    vqa_logits = pl_module.vqa_classifier(infer["cls_feats"])
    vqa_targets = torch.zeros(
        len(vqa_logits), pl_module.hparams.config["vqav2_label_size"]
    ).to(pl_module.device)

    vqa_labels = batch["vqa_labels"]
    vqa_scores = batch["vqa_scores"]

    for i, (_label, _score) in enumerate(zip(vqa_labels, vqa_scores)):
        for l, s in zip(_label, _score):
            vqa_targets[i, l] = s

    vqa_loss = (
        F.binary_cross_entropy_with_logits(vqa_logits, vqa_targets)
        * vqa_targets.shape[1]
    )  # https://github.com/jnhwkim/ban-vqa/blob/master/train.py#L19

    ret = {
        "vqa_loss": vqa_loss,
        "vqa_logits": vqa_logits,
        "vqa_targets": vqa_targets,
        "vqa_labels": vqa_labels,
        "vqa_scores": vqa_scores,
    }

    loss_name = 'vqa'

    loss = getattr(pl_module, f"{split}_{loss_name}_loss")(ret["vqa_loss"])
    score = getattr(pl_module, f"{split}_{loss_name}_score")(
        ret["vqa_logits"], ret["vqa_targets"]
    )
    pl_module.log(f"{split}/{loss_name}/loss", loss)
    pl_module.log(f"{split}/{loss_name}/score", score)

    return ret

def compute_nlvr2(pl_module, batch, split):
    infer1 = pl_module.infer(batch, image_token_type_idx=1)
    infer2 = pl_module.infer(batch, image_token_type_idx=2)
    
    cls_feats = torch.cat([infer1["cls_feats"], infer2["cls_feats"]], dim=-1)
    if pl_module.hparams.config["nlvr2_drop_rate"] > 0:
        cls_feats = pl_module.nlvr2_classifier_dropout(cls_feats)
    nlvr2_logits = pl_module.nlvr2_classifier(cls_feats)

    nlvr2_labels = batch["answers"]
    nlvr2_labels = torch.tensor(nlvr2_labels).to(pl_module.device).long()
    nlvr2_loss = F.cross_entropy(nlvr2_logits, nlvr2_labels.view(-1))

    ret = {
        "nlvr2_loss": nlvr2_loss,
        "nlvr2_logits": nlvr2_logits,
        "nlvr2_labels": nlvr2_labels,
    }

    loss_name = 'nlvr2'

    if split == "train":
        loss = getattr(pl_module, f"{split}_{loss_name}_loss")(ret["nlvr2_loss"])
        acc = getattr(pl_module, f"{split}_{loss_name}_accuracy")(
            ret["nlvr2_logits"], ret["nlvr2_labels"]
        )
        pl_module.log(f"{split}/{loss_name}/loss", loss)
        pl_module.log(f"{split}/{loss_name}/accuracy", acc)
    else:
        val_batches = [i for i, n in enumerate(batch["table_name"]) if "dev" in n]
        test_batches = [i for i, n in enumerate(batch["table_name"]) if "test" in n]

        if val_batches:
            val_loss = getattr(pl_module, f"val_{loss_name}_loss")(
                F.cross_entropy(
                    ret["nlvr2_logits"][val_batches], ret["nlvr2_labels"][val_batches]
                )
            )
            val_acc = getattr(pl_module, f"val_{loss_name}_accuracy")(
                ret["nlvr2_logits"][val_batches], ret["nlvr2_labels"][val_batches]
            )
            pl_module.log(f"val/nlvr2/loss", val_loss)
            pl_module.log(f"val/nlvr2/accuracy", val_acc)
        if test_batches:
            test_loss = getattr(pl_module, f"test_{loss_name}_loss")(
                F.cross_entropy(
                    ret["nlvr2_logits"][test_batches], ret["nlvr2_labels"][test_batches]
                )
            )
            test_acc = getattr(pl_module, f"test_{loss_name}_accuracy")(
                ret["nlvr2_logits"][test_batches], ret["nlvr2_labels"][test_batches]
            )
            pl_module.log(f"test/nlvr2/loss", test_loss)
            pl_module.log(f"test/nlvr2/accuracy", test_acc)

    return ret

def compute_irtr(pl_module, batch, split):
    _bs, _c, _h, _w = batch["image"][0].shape
    false_len = pl_module.hparams.config["draw_false_text"]
    text_ids = torch.stack(
        [batch[f"false_text_{i}_ids"] for i in range(false_len)], dim=1
    )
    text_masks = torch.stack(
        [batch[f"false_text_{i}_masks"] for i in range(false_len)], dim=1
    )

    text_ids = torch.stack([batch["text_ids"], text_ids], dim=1)
    text_masks = torch.stack([batch["text_masks"], text_masks], dim=1)

    infer = pl_module.infer(
        {
            "image": batch["image"],
            "text_ids": rearrange(text_ids, "bs fs tl -> (bs fs) tl"),
            "text_masks": rearrange(text_masks, "bs fs tl -> (bs fs) tl"),
        },
        irtr_len_image=false_len+1,
    )
    score = pl_module.rank_output(infer["cls_feats"])[:, 0]
    score = rearrange(score, "(bs fs) -> bs fs", bs=_bs, fs=false_len + 1)
    answer = torch.zeros(_bs).to(score).long()
    irtr_loss = F.cross_entropy(score, answer)

    ret = {
        "irtr_loss": irtr_loss,
    }

    loss_name = 'irtr'

    loss = getattr(pl_module, f"{split}_{loss_name}_loss")(ret["irtr_loss"])
    pl_module.log(f"{split}/{loss_name}/loss", loss)

    return ret

## calculate recall for irtr task
@torch.no_grad()
def compute_irtr_recall(pl_module, split):
    print("[Evaluation] load irtr dataset for text features caching")
    text_dset = pl_module.trainer.datamodule.dms[0].make_no_false_dset(split)
    text_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    text_dist_sampler = DistributedSampler(text_dset, shuffle=False)
    text_loader = torch.utils.data.DataLoader(
        text_dset,
        batch_size=pl_module.hparams.config["per_gpu_eval_batchsize_text"],
        num_workers=pl_module.hparams.config["num_workers"],
        sampler=text_dist_sampler,
        pin_memory=True,
        collate_fn=functools.partial(
            text_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
        shuffle=False,
    )
    print("[Evaluation] load irtr dataset for image features caching")
    image_dset = pl_module.trainer.datamodule.dms[0].make_no_false_dset(split, image_only=True)
    image_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    image_dist_sampler = DistributedSampler(image_dset, shuffle=False)
    image_loader = torch.utils.data.DataLoader(
        image_dset,
        batch_size=pl_module.hparams.config["per_gpu_eval_batchsize_image"],
        num_workers=pl_module.hparams.config["num_workers"],
        sampler=image_dist_sampler,
        pin_memory=True,
        collate_fn=functools.partial(
            image_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
        shuffle=False,
    )
    print("[Evaluation] start to cache the text features")
    text_embedss_cache, extend_text_masks_cache, tiids = list(), list(), list()
    for _b in tqdm(text_loader, desc="text prefetch loop"):
        text_embedss, extend_text_masks = pl_module.infer_text(
            {
                "text_ids": _b["text_ids"].to(pl_module.device),
                "text_masks": _b["text_masks"].to(pl_module.device),
            },
        )
        text_embedss_cache.append(text_embedss)
        extend_text_masks_cache.append(extend_text_masks)
        tiids += _b["img_index"]

    text_embedss_cache = torch.cat(text_embedss_cache, dim=1)
    extend_text_masks_cache = torch.cat(extend_text_masks_cache, dim=0)
    tiids = torch.LongTensor(tiids)

    # gather all text features
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    text_embedss_cache = pl_module.all_gather(text_embedss_cache.transpose(0, 1)).to(pl_module.device).view(-1, text_embedss_cache.size(0), text_embedss_cache.size(2), text_embedss_cache.size(3)).transpose(0, 1)
    extend_text_masks_cache = pl_module.all_gather(extend_text_masks_cache).to(pl_module.device).view(-1, extend_text_masks_cache.size(1), extend_text_masks_cache.size(2), extend_text_masks_cache.size(3))
    tiids = pl_module.all_gather(tiids).to(pl_module.device).view(-1)
    
    print("[Evaluation] start to cache the image features")
    image_embedss_cache, iids_cache = list(), list()
    for _b in tqdm(image_loader, desc="image prefetch loop"):
        image_embedss = pl_module.infer_image(img=_b["image"][0].to(pl_module.device))
        image_embedss_cache.append(image_embedss)
        iids_cache += _b["img_index"]
    image_embedss_cache = torch.cat(image_embedss_cache, dim=1)
    
    image_index, rank_scores, rank_iids = 0, list(), list()
    
    text_chunk_size = pl_module.hparams.config["per_gpu_eval_batchsize_fusion_text"]
    if text_embedss_cache.size(1) % text_chunk_size == 0:
        text_chunk_num = text_embedss_cache.size(1) // text_chunk_size
    else:
        text_chunk_num = text_embedss_cache.size(1) // text_chunk_size + 1

    print("[Evaluation] start to compute the irtr recall")
    for _iid in tqdm(iids_cache, desc="rank loop"):
        image_embedss = image_embedss_cache[:, image_index]
        image_index += 1
        
        img_batch_score = list()
        for _i in range(text_chunk_num):
            text_embedss = text_embedss_cache[:, _i*text_chunk_size:(_i+1)*text_chunk_size]
            extend_text_masks = extend_text_masks_cache[_i*text_chunk_size:(_i+1)*text_chunk_size]
            if pl_module.hparams.config["amp_flag"]:
                with torch.cuda.amp.autocast():
                    score = pl_module.rank_output(
                        pl_module.infer_fusion(
                            image_embedss, 
                            text_embedss, 
                            extend_text_masks, 
                            irtr_len_image=text_embedss.size(1),
                        )["cls_feats"]
                    )[:, 0]            
            else:
                score = pl_module.rank_output(
                    pl_module.infer_fusion(
                        image_embedss, 
                        text_embedss, 
                        extend_text_masks, 
                        irtr_len_image=text_embedss.size(1),
                    )["cls_feats"]
                )[:, 0]
            img_batch_score.append(score)

        img_batch_score = torch.cat(img_batch_score)
        rank_scores.append(img_batch_score)
        rank_iids.append(_iid)
    rank_iids = torch.LongTensor(rank_iids)
    rank_scores = torch.cat(rank_scores, dim=0)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    iids = pl_module.all_gather(rank_iids).to(pl_module.device).view(-1)
    scores = pl_module.all_gather(rank_scores).to(pl_module.device).view(len(iids), -1)

    topk10 = scores.topk(10, dim=1)
    topk5 = scores.topk(5, dim=1)
    topk1 = scores.topk(1, dim=1)
    topk10_iids = tiids[topk10.indices]
    topk5_iids = tiids[topk5.indices]
    topk1_iids = tiids[topk1.indices]

    tr_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
    tr_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
    tr_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()

    topk10 = scores.topk(10, dim=0)
    topk5 = scores.topk(5, dim=0)
    topk1 = scores.topk(1, dim=0)
    topk10_iids = iids[topk10.indices]
    topk5_iids = iids[topk5.indices]
    topk1_iids = iids[topk1.indices]

    ir_r10 = (tiids.unsqueeze(0) == topk10_iids).float().max(dim=0)[0].mean()
    ir_r5 = (tiids.unsqueeze(0) == topk5_iids).float().max(dim=0)[0].mean()
    ir_r1 = (tiids.unsqueeze(0) == topk1_iids).float().max(dim=0)[0].mean()

    tr_mean = (tr_r1 + tr_r5 + tr_r10) / 3
    ir_mean = (ir_r1 + ir_r5 + ir_r10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    return (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10, ir_mean, tr_mean, r_mean)

@torch.no_grad()
def compute_irtr_itm_itc_recall(pl_module, split):
    print("[Evaluation] load irtr dataset for text features caching")
    torch.cuda.empty_cache()
    text_dset = pl_module.trainer.datamodule.dms[0].make_no_false_dset(split)
    text_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    text_dist_sampler = DistributedSampler(text_dset, shuffle=False)
    text_loader = torch.utils.data.DataLoader(
        text_dset,
        batch_size=pl_module.hparams.config["per_gpu_eval_batchsize_text"],
        num_workers=pl_module.hparams.config["num_workers"],
        sampler=text_dist_sampler,
        pin_memory=True,
        collate_fn=functools.partial(
            text_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
        shuffle=False,
    )
    
    print("[Evaluation] load irtr dataset for image features caching")
    image_dset = pl_module.trainer.datamodule.dms[0].make_no_false_dset(split, image_only=True)
    image_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    image_dist_sampler = DistributedSampler(image_dset, shuffle=False)
    image_loader = torch.utils.data.DataLoader(
        image_dset,
        batch_size=pl_module.hparams.config["per_gpu_eval_batchsize_image"],
        num_workers=pl_module.hparams.config["num_workers"],
        sampler=image_dist_sampler,
        pin_memory=True,
        collate_fn=functools.partial(
            image_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
        shuffle=False,
    )
    
    print("[Evaluation] start to cache the text features")
    text_embedss_cache, extend_text_masks_cache, unimodal_feats_text_cache, tiids = list(), list(), list(), list()

    if pl_module.hparams.config["amp_flag"]:
        with torch.cuda.amp.autocast():
            for _b in tqdm(text_loader, desc="text prefetch loop"):
                text_embedss, extend_text_masks, unimodal_feats_text = pl_module.infer_text(
                    {
                        "text_ids": _b["text_ids"].to(pl_module.device),
                        "text_masks": _b["text_masks"].to(pl_module.device),
                    }, 
                    itc=True,
                )
                text_embedss_cache.append(text_embedss)
                unimodal_feats_text_cache.append(unimodal_feats_text)
                extend_text_masks_cache.append(extend_text_masks)
                tiids += _b["img_index"]
    else:
        for _b in tqdm(text_loader, desc="text prefetch loop"):
            text_embedss, extend_text_masks, unimodal_feats_text = pl_module.infer_text(
                {
                    "text_ids": _b["text_ids"].to(pl_module.device),
                    "text_masks": _b["text_masks"].to(pl_module.device),
                }, 
                itc=True,
            )
            text_embedss_cache.append(text_embedss)
            unimodal_feats_text_cache.append(unimodal_feats_text)
            extend_text_masks_cache.append(extend_text_masks)
            tiids += _b["img_index"]

    text_embedss_cache = torch.cat(text_embedss_cache, dim=1)
    unimodal_feats_text_cache = torch.cat(unimodal_feats_text_cache, dim=0)
    extend_text_masks_cache = torch.cat(extend_text_masks_cache, dim=0)
    tiids = torch.LongTensor(tiids)

    print("[Evaluation] gather all texts")
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    text_embedss_cache = pl_module.all_gather(text_embedss_cache.transpose(0, 1)).to(pl_module.device).view(-1, text_embedss_cache.size(0), text_embedss_cache.size(2), text_embedss_cache.size(3)).transpose(0, 1)
    unimodal_feats_text_cache = pl_module.all_gather(unimodal_feats_text_cache).view(-1, unimodal_feats_text_cache.size(1)).to(pl_module.device)
    extend_text_masks_cache = pl_module.all_gather(extend_text_masks_cache).to(pl_module.device).view(-1, extend_text_masks_cache.size(1), extend_text_masks_cache.size(2), extend_text_masks_cache.size(3))
    tiids = pl_module.all_gather(tiids).to(pl_module.device).view(-1)
    
    print("[Evaluation] start to cache the image features")
    image_embedss_cache, unimodal_feats_image_cache, iids_cache = list(), list(), list()
    if pl_module.hparams.config["gather_all_image_inputs"]:
        img_input_cache = list()
    if pl_module.hparams.config["amp_flag"]:
        with torch.cuda.amp.autocast():
            for _b in tqdm(image_loader, desc="image prefetch loop"):
                img_input = _b["image"][0].to(pl_module.device)
                image_embedss, unimodal_feats_image = pl_module.infer_image(img=img_input, itc=True)
                image_embedss_cache.append(image_embedss)
                if pl_module.hparams.config["gather_all_image_inputs"]:
                    img_input_cache.append(img_input)
                unimodal_feats_image_cache.append(unimodal_feats_image)
                iids_cache += _b["img_index"]
    else:
        for _b in tqdm(image_loader, desc="image prefetch loop"):
            img_input = _b["image"][0].to(pl_module.device)
            image_embedss, unimodal_feats_image = pl_module.infer_image(img=img_input, itc=True)
            image_embedss_cache.append(image_embedss)
            if pl_module.hparams.config["gather_all_image_inputs"]:
                img_input_cache.append(img_input)
            unimodal_feats_image_cache.append(unimodal_feats_image)
            iids_cache += _b["img_index"]
    image_embedss_cache = torch.cat(image_embedss_cache, dim=1)
    if pl_module.hparams.config["gather_all_image_inputs"]:
        img_input_cache = torch.cat(img_input_cache, dim=0)
    unimodal_feats_image_cache = torch.cat(unimodal_feats_image_cache, dim=0)

    # top-k contrastive scores
    print("[Evaluation] start to compute the irtr recall")

    print("[Evaluation] start image-to-text")

    sims_matrix = unimodal_feats_image_cache @ unimodal_feats_text_cache.t()
    _, topk_idx = sims_matrix.topk(k=pl_module.hparams.config['k_test'], dim=1)

    torch.cuda.empty_cache()

    image_index, rank_scores, rank_iids = 0, list(), list()
    for _iid in tqdm(iids_cache, desc="image-to-text rank loop"):
        topk_idx_i = topk_idx[image_index]
        image_embedss = image_embedss_cache[:, image_index]
        text_embedss = text_embedss_cache[:, topk_idx_i]
        extend_text_masks = extend_text_masks_cache[topk_idx_i]
        if pl_module.hparams.config["image_chunks"] >= 2:
            text_embedss = torch.chunk(text_embedss, pl_module.hparams.config["text_chunks"], dim=1)
            extend_text_masks = torch.chunk(extend_text_masks, pl_module.hparams.config["text_chunks"], dim=0)
            score_list, img_batch_score = [], None
            for text_embedss_, extend_text_masks_ in zip(text_embedss, extend_text_masks):
                if pl_module.hparams.config["amp_flag"]:
                    with torch.cuda.amp.autocast():
                        score = pl_module.itm_score(
                            pl_module.infer_fusion(
                                image_embedss, 
                                text_embedss_, 
                                extend_text_masks_, 
                                irtr_len_image=text_embedss_.size(1),
                            )["cls_feats"]
                        )[:, 1]
                        if img_batch_score is None:
                            img_batch_score = torch.full((sims_matrix.size(1),), -100.0, dtype=score.dtype, device=pl_module.device)
                        score_list.append(score)
                        
                else:
                    score = pl_module.itm_score(
                        pl_module.infer_fusion(
                            image_embedss, 
                            text_embedss_, 
                            extend_text_masks_, 
                            irtr_len_image=text_embedss_.size(1),
                        )["cls_feats"]
                    )[:, 1]
                    if img_batch_score is None:
                        img_batch_score = torch.full((sims_matrix.size(1),), -100.0, dtype=score.dtype, device=pl_module.device)
                    score_list.append(score)
                img_batch_score[topk_idx_i] = torch.cat(score_list, dim=0)
        else:
            if pl_module.hparams.config["amp_flag"]:
                with torch.cuda.amp.autocast():
                    score = pl_module.itm_score(
                        pl_module.infer_fusion(
                            image_embedss, 
                            text_embedss, 
                            extend_text_masks, 
                            irtr_len_image=text_embedss.size(1),
                        )["cls_feats"]
                    )[:, 1]
                    img_batch_score = torch.full((sims_matrix.size(1),), -100.0, dtype=score.dtype, device=pl_module.device)
                    img_batch_score[topk_idx_i] = score
            else:
                score = pl_module.itm_score(
                    pl_module.infer_fusion(
                        image_embedss, 
                        text_embedss, 
                        extend_text_masks, 
                        irtr_len_image=text_embedss.size(1),
                    )["cls_feats"]
                )[:, 1]
                img_batch_score = torch.full((sims_matrix.size(1),), -100.0, dtype=score.dtype, device=pl_module.device)
                img_batch_score[topk_idx_i] = score
        rank_scores.append(img_batch_score)
        rank_iids.append(_iid)

        image_index += 1
    rank_iids = torch.LongTensor(rank_iids)
    rank_scores = torch.cat(rank_scores, dim=0)
    print("[Evaluation] start text-to-image")

    unimodal_feats_image_cache = pl_module.all_gather(unimodal_feats_image_cache).to(pl_module.device).view(-1, unimodal_feats_image_cache.size(1))

    sims_matrix = unimodal_feats_image_cache @ unimodal_feats_text_cache.t()
    _, topk_idx = sims_matrix.topk(k=pl_module.hparams.config['k_test'], dim=0)
    rank = torch.distributed.get_rank()
    del unimodal_feats_image_cache, unimodal_feats_text_cache
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print("[Evaluation] gather all images")
    # if out of memory, then let's gather all the image input and rerun the vision part, but slower 4~5 times

    if text_embedss_cache.size(1) % torch.distributed.get_world_size() == 0:
        step = text_embedss_cache.size(1) // torch.distributed.get_world_size()
    else:
        step = text_embedss_cache.size(1) // torch.distributed.get_world_size() + 1
    start = rank * step
    end = min(text_embedss_cache.size(1), (rank + 1) * step)
    text_embedss_cache = text_embedss_cache[:, start:end]
    extend_text_masks_cache = extend_text_masks_cache[start:end]
    # topk_idx = topk_idx[:, start:end]

    if pl_module.hparams.config["gather_all_image_inputs"]:
        if not pl_module.hparams.config["save_memory"]:
            img_input_cache = pl_module.all_gather(img_input_cache).to(pl_module.device).view(-1, img_input_cache.size(1), img_input_cache.size(2), img_input_cache.size(3))
        else:
            useful_num = topk_idx.tolist()
            print(len(useful_num), len(useful_num[0]))
            useful_num = [item for sublist in useful_num for item in sublist]
            useful_num = set(useful_num)
            print(len(useful_num))
            all_idx_matrix = torch.zeros(sims_matrix.size(0)).long().to(pl_module.device)
            for i in range(topk_idx.size(1)):
                all_idx_matrix[topk_idx[:, i]] = 1

            image_input_list, image_input_idx_list = [], []
            current_image_num = sims_matrix.size(0) // dist.get_world_size()
            for i in range(current_image_num):
                j = i + current_image_num * rank
                if all_idx_matrix[j] == 1:
                    image_input_list.append(img_input_cache[i])
                    image_input_idx_list.append(j)
            image_input_list = torch.stack(image_input_list, dim=0)
            image_input_idx_list = torch.LongTensor(image_input_idx_list)
            img_input_cache = image_input_list

            gather_img_input_cache = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gather_img_input_cache, img_input_cache)
            gather_img_input_cache = [i.to(pl_module.device) for i in gather_img_input_cache]
            gather_img_input_cache = torch.cat(gather_img_input_cache, dim=0)
            img_input_cache = gather_img_input_cache
            
            gather_image_input_idx_list = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gather_image_input_idx_list, image_input_idx_list)
            gather_image_input_idx_list = [i.to(pl_module.device) for i in gather_image_input_idx_list]
            gather_image_input_idx_list = torch.cat(gather_image_input_idx_list, dim=0)
            image_input_idx_list = gather_image_input_idx_list

            print(img_input_cache.shape, image_input_idx_list.shape)
            
            inverse_img_input_idx = torch.zeros(sims_matrix.size(0)).long().fill_(-1).to(pl_module.device)
            for i in range(image_input_idx_list.size(0)):
                inverse_img_input_idx[image_input_idx_list[i]] = i

    else:
        if not pl_module.hparams.config["save_memory"]:
            image_embedss_cache = pl_module.all_gather(image_embedss_cache.transpose(0, 1)).to(pl_module.device).view(-1, image_embedss_cache.size(0), image_embedss_cache.size(2), image_embedss_cache.size(3)).transpose(0, 1)
        else:
            useful_num = topk_idx.tolist()
            print(len(useful_num), len(useful_num[0]))
            useful_num = [item for sublist in useful_num for item in sublist]
            useful_num = set(useful_num)
            print(len(useful_num))

            all_idx_matrix = torch.zeros(sims_matrix.size(0)).long().to(pl_module.device)
            for i in range(topk_idx.size(1)):
                all_idx_matrix[topk_idx[:, i]] = 1
            # current_idx_matrix = torch.zeros(sims_matrix.size(0))
            # for i in range(end-start):
            #     current_idx_matrix[topk_idx[:, i]] = 1
            image_embedss_cache = image_embedss_cache.transpose(0, 1)
            image_embedss_list, image_embedss_idx_list = [], []
            current_image_num = sims_matrix.size(0) // dist.get_world_size()
            for i in range(current_image_num):
                j = i + current_image_num * rank
                if all_idx_matrix[j] == 1:
                    image_embedss_list.append(image_embedss_cache[i])
                    image_embedss_idx_list.append(j)
            image_embedss_list = torch.stack(image_embedss_list, dim=0)
            image_embedss_idx_list = torch.LongTensor(image_embedss_idx_list)
            image_embedss_cache = image_embedss_list

            gather_image_embedss_cache = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gather_image_embedss_cache, image_embedss_cache)
            gather_image_embedss_cache = [i.to(pl_module.device) for i in gather_image_embedss_cache]
            gather_image_embedss_cache = torch.cat(gather_image_embedss_cache, dim=0)
            image_embedss_cache = gather_image_embedss_cache
            
            gather_image_embedss_idx_list = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gather_image_embedss_idx_list, image_embedss_idx_list)
            gather_image_embedss_idx_list = [i.to(pl_module.device) for i in gather_image_embedss_idx_list]
            gather_image_embedss_idx_list = torch.cat(gather_image_embedss_idx_list, dim=0)
            image_embedss_idx_list = gather_image_embedss_idx_list

            print(image_embedss_cache.shape, image_embedss_idx_list.shape)
            image_embedss_cache = image_embedss_cache.transpose(0, 1)
            
            inverse_image_embedss_idx = torch.zeros(sims_matrix.size(0)).long().fill_(-1).to(pl_module.device)
            for i in range(image_embedss_idx_list.size(0)):
                inverse_image_embedss_idx[image_embedss_idx_list[i]] = i

    topk_idx = topk_idx[:, start:end]

    txt_rank_scores = list()
    for text_index in tqdm(range(end-start), desc="text-to-image rank loop"):
        topk_idx_i = topk_idx[:, text_index]
        if pl_module.hparams.config["gather_all_image_inputs"]:
            if pl_module.hparams.config["save_memory"]:
                img_input = img_input_cache[inverse_img_input_idx[topk_idx_i]]
            else:
                img_input = img_input_cache[topk_idx_i]
            if pl_module.hparams.config["amp_flag"]:
                with torch.cuda.amp.autocast():
                    image_embedss = pl_module.infer_image(img=img_input)
            else:
                image_embedss = pl_module.infer_image(img=img_input)
        else:
            if pl_module.hparams.config["save_memory"]:
                image_embedss = image_embedss_cache[:, inverse_image_embedss_idx[topk_idx_i]]
            else:
                image_embedss = image_embedss_cache[:, topk_idx_i]
        text_embedss = text_embedss_cache[:, text_index]
        extend_text_masks = extend_text_masks_cache[text_index].unsqueeze_(0).expand(image_embedss.size(1), extend_text_masks_cache.size(1), extend_text_masks_cache.size(2), extend_text_masks_cache.size(3))
        if pl_module.hparams.config["image_chunks"] >= 2:
            image_embedss = torch.chunk(image_embedss, pl_module.hparams.config["image_chunks"], dim=1)
            extend_text_masks = torch.chunk(extend_text_masks, pl_module.hparams.config["image_chunks"], dim=0)
            score_list, txt_batch_score = [], None
            for image_embedss_, extend_text_masks_ in zip(image_embedss, extend_text_masks):
                if pl_module.hparams.config["amp_flag"]:
                    with torch.cuda.amp.autocast():
                        score = pl_module.itm_score(
                            pl_module.infer_fusion(
                                image_embedss_, 
                                text_embedss, 
                                extend_text_masks_, 
                                irtr_len_text=image_embedss_.size(1),
                            )["cls_feats"]
                        )[:, 1]
                        if txt_batch_score is None:
                            txt_batch_score = torch.full((sims_matrix.size(0),), -100.0, dtype=score.dtype,device=pl_module.device)
                        score_list.append(score)
                else:
                    score = pl_module.itm_score(
                        pl_module.infer_fusion(
                            image_embedss_, 
                            text_embedss, 
                            extend_text_masks_, 
                            irtr_len_text=image_embedss_.size(1),
                        )["cls_feats"]
                    )[:, 1]
                    if txt_batch_score is None:
                        txt_batch_score = torch.full((sims_matrix.size(0),), -100.0, dtype=score.dtype,device=pl_module.device)
                    score_list.append(score)
            txt_batch_score[topk_idx_i] = torch.cat(score_list, dim=0)
        else:
            if pl_module.hparams.config["amp_flag"]:
                with torch.cuda.amp.autocast():
                    score = pl_module.itm_score(
                        pl_module.infer_fusion(
                            image_embedss, 
                            text_embedss, 
                            extend_text_masks, 
                            irtr_len_text=image_embedss.size(1),
                        )["cls_feats"]
                    )[:, 1]
                    txt_batch_score = torch.full((sims_matrix.size(0),), -100.0, dtype=score.dtype,device=pl_module.device)
                    txt_batch_score[topk_idx_i] = score
            else:
                score = pl_module.itm_score(
                    pl_module.infer_fusion(
                        image_embedss, 
                        text_embedss, 
                        extend_text_masks, 
                        irtr_len_text=image_embedss.size(1),
                    )["cls_feats"]
                )[:, 1]
                txt_batch_score = torch.full((sims_matrix.size(0),), -100.0, dtype=score.dtype,device=pl_module.device)
                txt_batch_score[topk_idx_i] = score
        txt_rank_scores.append(txt_batch_score)
    txt_rank_scores = torch.cat(txt_rank_scores, dim=0)
    
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    
    iids = pl_module.all_gather(rank_iids).to(pl_module.device).view(-1)
    img_scores = pl_module.all_gather(rank_scores).to(pl_module.device).view(len(iids), -1)
    txt_scores = pl_module.all_gather(txt_rank_scores).to(pl_module.device).view(-1, len(iids)).t()

    scores = torch.stack((img_scores, txt_scores), dim=-1)
    scores = torch.max(scores, dim=-1)[0]

    topk10 = scores.topk(10, dim=1)
    topk5 = scores.topk(5, dim=1)
    topk1 = scores.topk(1, dim=1)
    topk10_iids = tiids[topk10.indices]
    topk5_iids = tiids[topk5.indices]
    topk1_iids = tiids[topk1.indices]

    tr_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
    tr_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
    tr_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()

    topk10 = scores.topk(10, dim=0)
    topk5 = scores.topk(5, dim=0)
    topk1 = scores.topk(1, dim=0)
    topk10_iids = iids[topk10.indices]
    topk5_iids = iids[topk5.indices]
    topk1_iids = iids[topk1.indices]

    ir_r10 = (tiids.unsqueeze(0) == topk10_iids).float().max(dim=0)[0].mean()
    ir_r5 = (tiids.unsqueeze(0) == topk5_iids).float().max(dim=0)[0].mean()
    ir_r1 = (tiids.unsqueeze(0) == topk1_iids).float().max(dim=0)[0].mean()

    tr_mean = (tr_r1 + tr_r5 + tr_r10) / 3
    ir_mean = (ir_r1 + ir_r5 + ir_r10) / 3
    r_mean = (tr_mean + ir_mean) / 2
    del text_embedss_cache, extend_text_masks_cache, image_embedss_cache
    if pl_module.hparams.config["gather_all_image_inputs"]:
        del img_input_cache
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10, ir_mean, tr_mean, r_mean)

@torch.no_grad()
def compute_irtr_itm_itc_recall_meter(pl_module, split):
    print("[Evaluation] load irtr dataset for text features caching")
    torch.cuda.empty_cache()
    text_dset = pl_module.trainer.datamodule.dms[0].make_no_false_dset(split)
    text_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    text_dist_sampler = DistributedSampler(text_dset, shuffle=False)
    text_loader = torch.utils.data.DataLoader(
        text_dset,
        batch_size=pl_module.hparams.config["per_gpu_eval_batchsize_text"],
        num_workers=pl_module.hparams.config["num_workers"],
        sampler=text_dist_sampler,
        pin_memory=True,
        collate_fn=functools.partial(
            text_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
        shuffle=False,
    )
    
    print("[Evaluation] load irtr dataset for image features caching")
    image_dset = pl_module.trainer.datamodule.dms[0].make_no_false_dset(split, image_only=True)
    image_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    image_dist_sampler = DistributedSampler(image_dset, shuffle=False)
    image_loader = torch.utils.data.DataLoader(
        image_dset,
        batch_size=pl_module.hparams.config["per_gpu_eval_batchsize_image"],
        num_workers=pl_module.hparams.config["num_workers"],
        sampler=image_dist_sampler,
        pin_memory=True,
        collate_fn=functools.partial(
            image_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
        shuffle=False,
    )
    
    print("[Evaluation] start to cache the text features")
    text_embeds_cache, extend_text_masks_cache, unimodal_feats_text_cache, tiids = list(), list(), list(), list()
    
    if pl_module.hparams.config["amp_flag"]:
        with torch.cuda.amp.autocast():
            for _b in tqdm(text_loader, desc="text prefetch loop"):
                text_embeds, extend_text_masks, unimodal_feats_text = pl_module.infer_text(
                    {
                        "text_ids": _b["text_ids"].to(pl_module.device),
                        "text_masks": _b["text_masks"].to(pl_module.device),
                    }, 
                    itc=True,
                )
                text_embeds_cache.append(text_embeds)
                unimodal_feats_text_cache.append(unimodal_feats_text)
                extend_text_masks_cache.append(extend_text_masks)
                tiids += _b["img_index"]
    else:
        for _b in tqdm(text_loader, desc="text prefetch loop"):
            text_embeds, extend_text_masks, unimodal_feats_text = pl_module.infer_text(
                {
                    "text_ids": _b["text_ids"].to(pl_module.device),
                    "text_masks": _b["text_masks"].to(pl_module.device),
                }, 
                itc=True,
            )
            text_embeds_cache.append(text_embeds)
            unimodal_feats_text_cache.append(unimodal_feats_text)
            extend_text_masks_cache.append(extend_text_masks)
            tiids += _b["img_index"]

    text_embeds_cache = torch.cat(text_embeds_cache, dim=0)
    unimodal_feats_text_cache = torch.cat(unimodal_feats_text_cache, dim=0)
    extend_text_masks_cache = torch.cat(extend_text_masks_cache, dim=0)
    tiids = torch.LongTensor(tiids)

    print("[Evaluation] gather all texts")
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    text_embeds_cache = pl_module.all_gather(text_embeds_cache).to(pl_module.device).view(-1, text_embeds_cache.size(1), text_embeds_cache.size(2))
    unimodal_feats_text_cache = pl_module.all_gather(unimodal_feats_text_cache).view(-1, unimodal_feats_text_cache.size(1)).to(pl_module.device)
    extend_text_masks_cache = pl_module.all_gather(extend_text_masks_cache).to(pl_module.device).view(-1, extend_text_masks_cache.size(1), extend_text_masks_cache.size(2), extend_text_masks_cache.size(3))
    tiids = pl_module.all_gather(tiids).to(pl_module.device).view(-1)
    
    print("[Evaluation] start to cache the image features")
    image_embeds_cache, unimodal_feats_image_cache, iids_cache = list(), list(), list()
    if pl_module.hparams.config["gather_all_image_inputs"]:
        img_input_cache = list()
    if pl_module.hparams.config["amp_flag"]:
        with torch.cuda.amp.autocast():
            for _b in tqdm(image_loader, desc="image prefetch loop"):
                img_input = _b["image"][0].to(pl_module.device)
                image_embeds, unimodal_feats_image = pl_module.infer_image(img=img_input, itc=True)
                image_embeds_cache.append(image_embeds)
                if pl_module.hparams.config["gather_all_image_inputs"]:
                    img_input_cache.append(img_input)
                unimodal_feats_image_cache.append(unimodal_feats_image)
                iids_cache += _b["img_index"]
    else:
        for _b in tqdm(image_loader, desc="image prefetch loop"):
            img_input = _b["image"][0].to(pl_module.device)
            image_embeds, unimodal_feats_image = pl_module.infer_image(img=img_input, itc=True)
            image_embeds_cache.append(image_embeds)
            if pl_module.hparams.config["gather_all_image_inputs"]:
                img_input_cache.append(img_input)
            unimodal_feats_image_cache.append(unimodal_feats_image)
            iids_cache += _b["img_index"]
    image_embeds_cache = torch.cat(image_embeds_cache, dim=0)
    if pl_module.hparams.config["gather_all_image_inputs"]:
        img_input_cache = torch.cat(img_input_cache, dim=0)
    unimodal_feats_image_cache = torch.cat(unimodal_feats_image_cache, dim=0)

    # top-k contrastive scores
    print("[Evaluation] start to compute the irtr recall")

    print("[Evaluation] start image-to-text")

    sims_matrix = unimodal_feats_image_cache @ unimodal_feats_text_cache.t()
    _, topk_idx = sims_matrix.topk(k=pl_module.hparams.config['k_test'], dim=1)

    torch.cuda.empty_cache()

    image_index, rank_scores, rank_iids = 0, list(), list()
    for _iid in tqdm(iids_cache, desc="image-to-text rank loop"):
        topk_idx_i = topk_idx[image_index]
        image_embeds = image_embeds_cache[image_index]
        text_embeds = text_embeds_cache[topk_idx_i]
        extend_text_masks = extend_text_masks_cache[topk_idx_i]
        if pl_module.hparams.config["image_chunks"] >= 2:
            text_embeds = torch.chunk(text_embeds, pl_module.hparams.config["text_chunks"], dim=0)
            extend_text_masks = torch.chunk(extend_text_masks, pl_module.hparams.config["text_chunks"], dim=0)
            score_list, img_batch_score = [], None
            for text_embeds_, extend_text_masks_ in zip(text_embeds, extend_text_masks):
                if pl_module.hparams.config["amp_flag"]:
                    with torch.cuda.amp.autocast():
                        score = pl_module.itm_score(
                            pl_module.infer_fusion(
                                image_embeds, 
                                text_embeds_, 
                                extend_text_masks_, 
                                irtr_len_image=text_embeds_.size(0),
                            )["cls_feats"]
                        )[:, 1]
                        if img_batch_score is None:
                            img_batch_score = torch.full((sims_matrix.size(1),), -100.0, dtype=score.dtype, device=pl_module.device)
                        score_list.append(score)
                        
                else:
                    score = pl_module.itm_score(
                        pl_module.infer_fusion(
                            image_embeds, 
                            text_embeds_, 
                            extend_text_masks_, 
                            irtr_len_image=text_embeds_.size(0),
                        )["cls_feats"]
                    )[:, 1]
                    if img_batch_score is None:
                        img_batch_score = torch.full((sims_matrix.size(1),), -100.0, dtype=score.dtype, device=pl_module.device)
                    score_list.append(score)
                img_batch_score[topk_idx_i] = torch.cat(score_list, dim=0)
        else:
            if pl_module.hparams.config["amp_flag"]:
                with torch.cuda.amp.autocast():
                    score = pl_module.itm_score(
                        pl_module.infer_fusion(
                            image_embeds, 
                            text_embeds, 
                            extend_text_masks, 
                            irtr_len_image=text_embeds.size(0),
                        )["cls_feats"]
                    )[:, 1]
                    img_batch_score = torch.full((sims_matrix.size(1),), -100.0, dtype=score.dtype, device=pl_module.device)
                    img_batch_score[topk_idx_i] = score
            else:
                score = pl_module.itm_score(
                    pl_module.infer_fusion(
                        image_embeds, 
                        text_embeds, 
                        extend_text_masks, 
                        irtr_len_image=text_embeds.size(0),
                    )["cls_feats"]
                )[:, 1]
                img_batch_score = torch.full((sims_matrix.size(1),), -100.0, dtype=score.dtype, device=pl_module.device)
                img_batch_score[topk_idx_i] = score
        rank_scores.append(img_batch_score)
        rank_iids.append(_iid)

        image_index += 1
    rank_iids = torch.LongTensor(rank_iids)
    rank_scores = torch.cat(rank_scores, dim=0)
    print("[Evaluation] start text-to-image")

    unimodal_feats_image_cache = pl_module.all_gather(unimodal_feats_image_cache).to(pl_module.device).view(-1, unimodal_feats_image_cache.size(1))

    sims_matrix = unimodal_feats_image_cache @ unimodal_feats_text_cache.t()
    _, topk_idx = sims_matrix.topk(k=pl_module.hparams.config['k_test'], dim=0)
    rank = torch.distributed.get_rank()
    del unimodal_feats_image_cache, unimodal_feats_text_cache
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print("[Evaluation] gather all images")
    # if out of memory, then let's gather all the image input and rerun the vision part, but slower 4~5 times

    if text_embeds_cache.size(0) % torch.distributed.get_world_size() == 0:
        step = text_embeds_cache.size(0) // torch.distributed.get_world_size()
    else:
        step = text_embeds_cache.size(0) // torch.distributed.get_world_size() + 1
    start = rank * step
    end = min(text_embeds_cache.size(0), (rank + 1) * step)
    text_embeds_cache = text_embeds_cache[start:end]
    extend_text_masks_cache = extend_text_masks_cache[start:end]
    topk_idx = topk_idx[:, start:end]

    if pl_module.hparams.config["gather_all_image_inputs"]:
        img_input_cache = pl_module.all_gather(img_input_cache).to(pl_module.device).view(-1, img_input_cache.size(1), img_input_cache.size(2), img_input_cache.size(3))
    else:
        image_embeds_cache = pl_module.all_gather(image_embeds_cache).to(pl_module.device).view(-1, image_embeds_cache.size(1), image_embeds_cache.size(2))

    txt_rank_scores = list()
    for text_index in tqdm(range(end-start), desc="text-to-image rank loop"):
        topk_idx_i = topk_idx[:, text_index]
        if pl_module.hparams.config["gather_all_image_inputs"]:
            img_input = img_input_cache[topk_idx_i]
            if pl_module.hparams.config["amp_flag"]:
                with torch.cuda.amp.autocast():
                    image_embeds = pl_module.infer_image(img=img_input)
            else:
                image_embeds = pl_module.infer_image(img=img_input)
        else:
            image_embeds = image_embeds_cache[topk_idx_i]
        text_embeds = text_embeds_cache[text_index]
        extend_text_masks = extend_text_masks_cache[text_index].unsqueeze_(0).expand(image_embeds.size(0), extend_text_masks_cache.size(1), extend_text_masks_cache.size(2), extend_text_masks_cache.size(3))
        if pl_module.hparams.config["image_chunks"] >= 2:
            image_embeds = torch.chunk(image_embeds, pl_module.hparams.config["image_chunks"], dim=0)
            extend_text_masks = torch.chunk(extend_text_masks, pl_module.hparams.config["image_chunks"], dim=0)
            score_list, txt_batch_score = [], None
            for image_embeds_, extend_text_masks_ in zip(image_embeds, extend_text_masks):
                if pl_module.hparams.config["amp_flag"]:
                    with torch.cuda.amp.autocast():
                        score = pl_module.itm_score(
                            pl_module.infer_fusion(
                                image_embeds_, 
                                text_embeds, 
                                extend_text_masks_, 
                                irtr_len_text=image_embeds_.size(0),
                            )["cls_feats"]
                        )[:, 1]
                        if txt_batch_score is None:
                            txt_batch_score = torch.full((sims_matrix.size(0),), -100.0, dtype=score.dtype,device=pl_module.device)
                        score_list.append(score)
                else:
                    score = pl_module.itm_score(
                        pl_module.infer_fusion(
                            image_embeds_, 
                            text_embeds, 
                            extend_text_masks_, 
                            irtr_len_text=image_embeds_.size(0),
                        )["cls_feats"]
                    )[:, 1]
                    if txt_batch_score is None:
                        txt_batch_score = torch.full((sims_matrix.size(0),), -100.0, dtype=score.dtype,device=pl_module.device)
                    score_list.append(score)
            txt_batch_score[topk_idx_i] = torch.cat(score_list, dim=0)
        else:
            if pl_module.hparams.config["amp_flag"]:
                with torch.cuda.amp.autocast():
                    score = pl_module.itm_score(
                        pl_module.infer_fusion(
                            image_embeds, 
                            text_embeds, 
                            extend_text_masks, 
                            irtr_len_text=image_embeds.size(0),
                        )["cls_feats"]
                    )[:, 1]
                    txt_batch_score = torch.full((sims_matrix.size(0),), -100.0, dtype=score.dtype,device=pl_module.device)
                    txt_batch_score[topk_idx_i] = score
            else:
                score = pl_module.itm_score(
                    pl_module.infer_fusion(
                        image_embeds, 
                        text_embeds, 
                        extend_text_masks, 
                        irtr_len_text=image_embeds.size(0),
                    )["cls_feats"]
                )[:, 1]
                txt_batch_score = torch.full((sims_matrix.size(0),), -100.0, dtype=score.dtype,device=pl_module.device)
                txt_batch_score[topk_idx_i] = score
        txt_rank_scores.append(txt_batch_score)
    txt_rank_scores = torch.cat(txt_rank_scores, dim=0)
    
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    
    iids = pl_module.all_gather(rank_iids).to(pl_module.device).view(-1)
    img_scores = pl_module.all_gather(rank_scores).to(pl_module.device).view(len(iids), -1)
    txt_scores = pl_module.all_gather(txt_rank_scores).to(pl_module.device).view(-1, len(iids)).t()

    scores = torch.stack((img_scores, txt_scores), dim=-1)
    scores = torch.max(scores, dim=-1)[0]

    topk10 = scores.topk(10, dim=1)
    topk5 = scores.topk(5, dim=1)
    topk1 = scores.topk(1, dim=1)
    topk10_iids = tiids[topk10.indices]
    topk5_iids = tiids[topk5.indices]
    topk1_iids = tiids[topk1.indices]

    tr_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
    tr_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
    tr_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()

    topk10 = scores.topk(10, dim=0)
    topk5 = scores.topk(5, dim=0)
    topk1 = scores.topk(1, dim=0)
    topk10_iids = iids[topk10.indices]
    topk5_iids = iids[topk5.indices]
    topk1_iids = iids[topk1.indices]

    ir_r10 = (tiids.unsqueeze(0) == topk10_iids).float().max(dim=0)[0].mean()
    ir_r5 = (tiids.unsqueeze(0) == topk5_iids).float().max(dim=0)[0].mean()
    ir_r1 = (tiids.unsqueeze(0) == topk1_iids).float().max(dim=0)[0].mean()

    tr_mean = (tr_r1 + tr_r5 + tr_r10) / 3
    ir_mean = (ir_r1 + ir_r5 + ir_r10) / 3
    r_mean = (tr_mean + ir_mean) / 2
    del text_embeds_cache, extend_text_masks_cache, image_embeds_cache
    if pl_module.hparams.config["gather_all_image_inputs"]:
        del img_input_cache
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10, ir_mean, tr_mean, r_mean)

@torch.no_grad()
def compute_irtr_itc_recall(pl_module, split):
    print("[Evaluation] load irtr dataset for text features caching")
    torch.cuda.empty_cache()
    text_dset = pl_module.trainer.datamodule.dms[0].make_no_false_dset(split)
    text_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    text_dist_sampler = DistributedSampler(text_dset, shuffle=False)
    text_loader = torch.utils.data.DataLoader(
        text_dset,
        batch_size=pl_module.hparams.config["per_gpu_eval_batchsize_text"],
        num_workers=pl_module.hparams.config["num_workers"],
        sampler=text_dist_sampler,
        pin_memory=True,
        collate_fn=functools.partial(
            text_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
        shuffle=False,
    )
    
    print("[Evaluation] load irtr dataset for image features caching")
    image_dset = pl_module.trainer.datamodule.dms[0].make_no_false_dset(split, image_only=True)
    image_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    image_dist_sampler = DistributedSampler(image_dset, shuffle=False)
    image_loader = torch.utils.data.DataLoader(
        image_dset,
        batch_size=pl_module.hparams.config["per_gpu_eval_batchsize_image"],
        num_workers=pl_module.hparams.config["num_workers"],
        sampler=image_dist_sampler,
        pin_memory=True,
        collate_fn=functools.partial(
            image_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
        shuffle=False,
    )
    
    print("[Evaluation] start to cache the text features")
    unimodal_feats_text_cache, tiids = list(), list()
    
    if pl_module.hparams.config["amp_flag"]:
        with torch.cuda.amp.autocast():
            for _b in tqdm(text_loader, desc="text prefetch loop"):
                unimodal_feats_text = pl_module.infer_text(
                    {
                        "text_ids": _b["text_ids"].to(pl_module.device),
                        "text_masks": _b["text_masks"].to(pl_module.device),
                    }, 
                    itc=True,
                )[2]
                unimodal_feats_text_cache.append(unimodal_feats_text)
                tiids += _b["img_index"]
    else:
        for _b in tqdm(text_loader, desc="text prefetch loop"):
            unimodal_feats_text = pl_module.infer_text(
                {
                    "text_ids": _b["text_ids"].to(pl_module.device),
                    "text_masks": _b["text_masks"].to(pl_module.device),
                }, 
                itc=True,
            )[2]
            unimodal_feats_text_cache.append(unimodal_feats_text)
            tiids += _b["img_index"]

    unimodal_feats_text_cache = torch.cat(unimodal_feats_text_cache, dim=0)
    tiids = torch.LongTensor(tiids)

    print("[Evaluation] gather all texts")
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    unimodal_feats_text_cache = pl_module.all_gather(unimodal_feats_text_cache).view(-1, unimodal_feats_text_cache.size(1)).to(pl_module.device)
    tiids = pl_module.all_gather(tiids).to(pl_module.device).view(-1)
    
    print("[Evaluation] start to cache the image features")
    unimodal_feats_image_cache, iids_cache = list(), list()
    if pl_module.hparams.config["amp_flag"]:
        with torch.cuda.amp.autocast():
            for _b in tqdm(image_loader, desc="image prefetch loop"):
                img_input = _b["image"][0].to(pl_module.device)
                unimodal_feats_image = pl_module.infer_image(img=img_input, itc=True)[1]
                unimodal_feats_image_cache.append(unimodal_feats_image)
                iids_cache += _b["img_index"]
    else:
        for _b in tqdm(image_loader, desc="image prefetch loop"):
            img_input = _b["image"][0].to(pl_module.device)
            unimodal_feats_image = pl_module.infer_image(img=img_input, itc=True)[1]
            unimodal_feats_image_cache.append(unimodal_feats_image)
            iids_cache += _b["img_index"]
    unimodal_feats_image_cache = torch.cat(unimodal_feats_image_cache, dim=0)

    torch.cuda.empty_cache()
    print("[Evaluation] start to compute the itc recall")

    sims_matrix = unimodal_feats_image_cache @ unimodal_feats_text_cache.t()
    rank_iids = torch.LongTensor(iids_cache)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    
    sims_matrix = pl_module.all_gather(sims_matrix).view(-1, sims_matrix.size(1)).to(pl_module.device)
    iids = pl_module.all_gather(rank_iids).to(pl_module.device).view(-1)
    scores = sims_matrix
    
    topk10 = scores.topk(10, dim=1)
    topk5 = scores.topk(5, dim=1)
    topk1 = scores.topk(1, dim=1)
    topk10_iids = tiids[topk10.indices]
    topk5_iids = tiids[topk5.indices]
    topk1_iids = tiids[topk1.indices]

    tr_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
    tr_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
    tr_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()

    topk10 = scores.topk(10, dim=0)
    topk5 = scores.topk(5, dim=0)
    topk1 = scores.topk(1, dim=0)
    topk10_iids = iids[topk10.indices]
    topk5_iids = iids[topk5.indices]
    topk1_iids = iids[topk1.indices]

    ir_r10 = (tiids.unsqueeze(0) == topk10_iids).float().max(dim=0)[0].mean()
    ir_r5 = (tiids.unsqueeze(0) == topk5_iids).float().max(dim=0)[0].mean()
    ir_r1 = (tiids.unsqueeze(0) == topk1_iids).float().max(dim=0)[0].mean()

    tr_mean = (tr_r1 + tr_r5 + tr_r10) / 3
    ir_mean = (ir_r1 + ir_r5 + ir_r10) / 3
    r_mean = (tr_mean + ir_mean) / 2
    torch.cuda.empty_cache()
    return (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10, ir_mean, tr_mean, r_mean)

## save vqa test results to json file, then you can manually upload it to the evalai server
def vqa_test_step(pl_module, batch, output):
    try:
        id2answer = (
            pl_module.trainer.datamodule.dm_dicts["vqa_trainval"].id2answer
            if "vqa_trainval" in pl_module.trainer.datamodule.dm_dicts
            else pl_module.trainer.datamodule.dm_dicts["vqa"].id2answer
        )
    except:
        id2answer = (
            pl_module.trainer.datamodule.dm_dicts["gqa_test"].id2answer
            if "gqa_test" in pl_module.trainer.datamodule.dm_dicts
            else pl_module.trainer.datamodule.dm_dicts["gqa"].id2answer
        )
        vqa_logits = output["vqa_logits"]
        vqa_preds = vqa_logits.argmax(dim=-1)
        vqa_preds = [id2answer[pred.item()] for pred in vqa_preds]
        questions = batch["text"]
        qids = batch["qid"]
        return {"qids": qids, "preds": vqa_preds, "gqa": True}
    vqa_logits = output["vqa_logits"]
    vqa_preds = vqa_logits.argmax(dim=-1)
    vqa_preds = [id2answer[pred.item()] for pred in vqa_preds]
    questions = batch["text"]
    qids = batch["qid"]
    return {"qids": qids, "preds": vqa_preds, "gqa": False}

def vqa_test_wrapup(outs, model_name, log_dir):
    rank = torch.distributed.get_rank()
    qids, preds = list(), list()
    gqa = False
    for out in outs:
        qids += out["qids"]
        preds += out["preds"]
        gqa = out['gqa']

    rets = list()
    for qid, pred in zip(qids, preds):
        if gqa:
            rets.append({"questionId": qid, "prediction": pred})
        else:
            rets.append({"question_id": qid, "answer": pred})

    if torch.distributed.is_initialized():
        torch.distributed.barrier()   
    
    print(f'rank: {rank}, world_size: {dist.get_world_size()}, length of rets: {len(rets)}')
    gather_rets = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gather_rets, rets)
    print(f'rank: {rank}, length of gather_rets: {len(gather_rets)}')
    print(f'rank: {rank}, length of gather_rets[0]: {len(gather_rets[0])}')
    
    if rank == 0:
        jsons = list()
        for rets_ in gather_rets:
            jsons += rets_
        with open(f"{log_dir}/vqa_submit_{model_name}.json", "w") as fp:
            json.dump(jsons, fp, indent=4)
    
    if torch.distributed.is_initialized():
        torch.distributed.barrier()