# ManagerTower

This repo is the official `Pytorch` implementation of the paper:

**ManagerTower: Aggregating the Insights of Uni-Modal Experts for Vision-Language Representation Learning**

[Xiao Xu*](http://ir.hit.edu.cn/~xxu/), [Bei Li](https://libeineu.github.io/), [Chenfei Wu](https://chenfei-wu.github.io/), [Shao-Yen Tseng](https://www.shaoyen.me/), [Anahita Bhiwandiwalla](https://scholar.google.com/citations?user=N-Qoq1gAAAAJ&hl=en), [Shachar Rosenman](https://scholar.google.com/citations?user=-8JzBBEAAAAJ&hl=en), [Vasudev Lal](https://scholar.google.com/citations?user=Qbu4oKwAAAAJ&hl=en), [Wanxiang Che](http://ir.hit.edu.cn/~car/), and [Nan Duan](https://nanduan.github.io/).

[ACL 2023 (Oral)](https://2023.aclweb.org/) | Association for Computational Linguistics

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

- [Pre-trained checkpoints on 4M data](https://chenfei.blob.core.windows.net/data/G/ManagerTower/best_checkpoints/ManagerTower_pt_base.ckpt?sv=2020-10-02&st=2023-05-31T03%3A43%3A27Z&se=2123-06-01T03%3A43%3A00Z&sr=b&sp=r&sig=SD8V1dubKJ6jOko8iQD69Rc0RrwOf8Cc5KMUWRKWNb8%3D)
- Fine-tuned checkpoints for
  - [Visual Question Answering on VQAv2](https://chenfei.blob.core.windows.net/data/G/ManagerTower/best_checkpoints/ManagerTower_ftfpt_base_vqav2.ckpt?sv=2020-10-02&st=2023-05-31T03%3A44%3A13Z&se=2123-06-01T03%3A44%3A00Z&sr=b&sp=r&sig=OTopHs3TIMfQyy8W9Ov5YudFPHOVnrjSmrxhDI9E9N8%3D)
  - [Image-Text Retrieval on Flickr30k](https://chenfei.blob.core.windows.net/data/G/ManagerTower/best_checkpoints/ManagerTower_ftfpt_base_flickr30k.ckpt?sv=2020-10-02&st=2023-05-31T03%3A44%3A49Z&se=2123-06-01T03%3A44%3A00Z&sr=b&sp=r&sig=ZgWSDZ98yTGL%2FBDByWmHtRWec3BRIvUczndt3RRB7OQ%3D)
  - [Visual Entailment on SNLI-VE](https://chenfei.blob.core.windows.net/data/G/ManagerTower/best_checkpoints/ManagerTower_ftfpt_base_snlive.ckpt?sv=2020-10-02&st=2023-05-31T03%3A45%3A38Z&se=2123-06-01T03%3A45%3A00Z&sr=b&sp=r&sig=NT%2BNCs85zu6S5S69KYTuoaJ6KYCu777IaX0bk6jblpU%3D)
  - [Visual Reasoning on NLVR$^2$](https://chenfei.blob.core.windows.net/data/G/ManagerTower/best_checkpoints/ManagerTower_ftfpt_base_nlvr2.ckpt?sv=2020-10-02&st=2023-05-31T03%3A45%3A13Z&se=2123-06-01T03%3A45%3A00Z&sr=b&sp=r&sig=pvgiNo%2Ft2VMnaBICrFB3gr2dPZPMOm0eMe%2BIpiJABuo%3D)
- Here is an example for downloading a checkpoint.

  ```Shell
  # download azcopy
  wget https://aka.ms/downloadazcopy-v10-linux
  tar -xvf downloadazcopy-v10-linux
  sudo cp ./azcopy_linux_amd64_*/azcopy /usr/bin/
  sudo chmod -R 777 /usr/bin/azcopy
  # azcopy copy [remote path] [local path]
  azcopy copy "https://chenfei.blob.core.windows.net/data/G/ManagerTower/best_checkpoints/ManagerTower_pt_base.ckpt?sv=2020-10-02&st=2023-05-31T03%3A43%3A27Z&se=2123-06-01T03%3A43%3A00Z&sr=b&sp=r&sig=SD8V1dubKJ6jOko8iQD69Rc0RrwOf8Cc5KMUWRKWNb8%3D" "./ManagerTower_pt_base.ckpt"
  ```

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

```

## Acknowledgement

We are highly grateful for the public code of the following papers, our code is partly based on them:
- Main Code: [BridgeTower](https://github.com/microsoft/BridgeTower) (which is highly based on [ViLT](https://github.com/dandelin/ViLT) and [METER](https://github.com/zdou0830/METER))
- Others: [CLIP](https://github.com/openai/CLIP), [ALBEF](https://github.com/salesforce/ALBEF), [BLIP](https://github.com/salesforce/BLIP)
