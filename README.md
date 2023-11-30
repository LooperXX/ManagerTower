# ManagerTower

This repo is the official `Pytorch` implementation of the paper:

**ManagerTower: Aggregating the Insights of Uni-Modal Experts for Vision-Language Representation Learning**

[Xiao Xu](http://ir.hit.edu.cn/~xxu/), [Bei Li](https://libeineu.github.io/), [Chenfei Wu](https://chenfei-wu.github.io/), [Shao-Yen Tseng](https://www.shaoyen.me/), [Anahita Bhiwandiwalla](https://scholar.google.com/citations?user=N-Qoq1gAAAAJ&hl=en), [Shachar Rosenman](https://scholar.google.com/citations?user=-8JzBBEAAAAJ&hl=en), [Vasudev Lal](https://scholar.google.com/citations?user=Qbu4oKwAAAAJ&hl=en), [Wanxiang Che](http://ir.hit.edu.cn/~car/), [Nan Duan](https://nanduan.github.io/).

[ACL 2023 (Oral)](https://2023.aclweb.org/) | Association for Computational Linguistics

[Arxiv](https://arxiv.org/abs/2306.00103)

## Abstract

Two-Tower Vision-Language (VL) models have shown promising improvements on various downstream VL tasks. Although the most advanced work improves performance by building bridges between encoders, it suffers from ineffective layer-by-layer utilization of uni-modal representations and cannot flexibly exploit different levels of uni-modal semantic knowledge. In this work, we propose ManagerTower, a novel VL model architecture that gathers and combines the insights of pre-trained uni-modal experts at different levels. The managers introduced in each cross-modal layer can adaptively aggregate uni-modal semantic knowledge to facilitate more comprehensive cross-modal alignment and fusion. ManagerTower outperforms previous strong baselines both with and without Vision-Language Pre-training (VLP). With only 4M VLP data, ManagerTower achieves superior performances on various downstream VL tasks, especially 79.15% accuracy on VQAv2 Test-Std, 86.56% IR@1 and 95.64% TR@1 on Flickr30K. Code and checkpoints are available at https://github.com/LooperXX/ManagerTower.

## Architecture

![Architecture](images/framework.jpg)

## BridgeTower vs. ManagerTower

<div align=center>
<img src="images/comparison.jpg" alt="Comparison" style="width: 60%"/>
</div>

## Main Results

![Result](images/result.jpg)

## Visualization

![Visualization](images/visualization.jpg)

## Deployment

- Run `setup.sh` to set up the environment.
- [Optional] We use [wandb](https://wandb.ai/) to track experiments! Please remember to `wandb login` and paste your token before running the script.

## Dataset Preparation

- We follow [ViLT](https://github.com/dandelin/ViLT) and use pyarrow to serialize the datasets. See [here](https://github.com/dandelin/ViLT/blob/master/DATA.md) for details.
- For SNLI-VE dataset, we follow [here](https://github.com/necla-ml/SNLI-VE).
- For VG-QA dataset, except the image-text pairs in [VG](https://visualgenome.org/api/v0/api_home.html) got from [here](https://github.com/dandelin/ViLT/blob/master/DATA.md), [image meta data](https://visualgenome.org/static/data/dataset/image_data_v1.json.zip), [question answers data](https://visualgenome.org/static/data/dataset/question_answers.json.zip) and [coco split information](https://github.com/peteanderson80/bottom-up-attention/tree/master/data/genome/coco_splits) also need to be downloaded.
- The final file structure of datasets are shown in `setup.sh`.

## Checkpoints

We provide the following checkpoints for reproducing our results. You can download them from [here](https://huggingface.co/LooperXX/ManagerTower).

- [Pre-trained checkpoints on 4M data](https://huggingface.co/LooperXX/ManagerTower/blob/main/ManagerTower_pt_base.ckpt)
- Fine-tuned checkpoints for
  - [Visual Question Answering on VQAv2](https://huggingface.co/LooperXX/ManagerTower/blob/main/ManagerTower_ftfpt_base_vqav2.ckpt)
  - [Image-Text Retrieval on Flickr30k](https://huggingface.co/LooperXX/ManagerTower/blob/main/ManagerTower_ftfpt_base_flickr30k.ckpt)
  - [Visual Entailment on SNLI-VE](https://huggingface.co/LooperXX/ManagerTower/blob/main/ManagerTower_ftfpt_base_snlive.ckpt)
  - [Visual Reasoning on NLVR$^2$](https://huggingface.co/LooperXX/ManagerTower/blob/main/ManagerTower_ftfpt_base_nlvr2.ckpt)

## Pre-training on Image-Text Datasets

```bash
# Pre-train ManagerTower Base Model
bash scripts/pre_train.sh
```

## Fine-tuning on Downstream VL Tasks

- VQAv2 Evaluation needs to submit the `json` file in the `logs/` directory to [eval.ai](https://eval.ai/web/challenges/challenge-page/830/overview) evaluation server to get the test-dev and/or test-std scores.

```bash
# Base Model on VQAv2 without VLP
bash scripts/ftfs_base_vqa.sh

# Base Model on VQAv2 with VLP
bash scripts/ftfpt_base_vqa.sh

# Base Model on SNLI-VE with VLP
bash scripts/ftfpt_base_snlive.sh

# Base Model on NLVR^2 with VLP
bash scripts/ftfpt_base_nlvr2.sh

# Base Model on IRTR-Flickr30K with VLP (follow ALBEF to use ITC to sample hard negatives for ITM)
bash scripts/ftfpt_base_flickr.sh
```

## Citation

```
@article{xu2023managertower,
  title={ManagerTower: Aggregating the Insights of Uni-Modal Experts for Vision-Language Representation Learning},
  author={Xu, Xiao and Li, Bei and Wu, Chenfei and Tseng, Shao-Yen and Bhiwandiwalla, Anahita and Rosenman, Shachar and Lal, Vasudev and Che, Wanxiang and Duan, Nan},
  journal={arXiv preprint arXiv:2306.00103},
  year={2023}
}
```

## Acknowledgement

We are highly grateful for the public code of the following papers, our code is partly based on them:
- Main Code: [BridgeTower](https://github.com/microsoft/BridgeTower) (which is highly based on [ViLT](https://github.com/dandelin/ViLT) and [METER](https://github.com/zdou0830/METER))
- Others: [CLIP](https://github.com/openai/CLIP), [ALBEF](https://github.com/salesforce/ALBEF), [BLIP](https://github.com/salesforce/BLIP)
