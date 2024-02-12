# Self-supervised learning representation quality assessment

Python implementation of methods [RankMe](https://arxiv.org/abs/2210.02885), [α-ReQ](https://openreview.net/forum?id=ii9X4vtZGTZ) and [Cluster Learnability](https://arxiv.org/abs/2206.01251)

# Pretrained Checkpoints that were used


## MAE reimplementation

|Weights|Pretrain|Probe|Probe|k-NN|
|:---:|:---:|:---:|:---:|:---:|
|[ViT-B/16](https://ml.jku.at/research/maect/download/mae_reimpl_base16.th)|[hp](https://github.com/ml-jku/MAE-CT/blob/main/yamls/stage1_mae/base16.yaml)|[66.7](https://github.com/ml-jku/MAE-CT/blob/main/probes/mae_reimpl_base16.th)|[log](https://github.com/ml-jku/MAE-CT/blob/main/logs/probe/mae_base16.th)|51.1|
|[ViT-L/16](https://ml.jku.at/research/maect/download/mae_reimpl_large16.th)|[hp](https://github.com/ml-jku/MAE-CT/blob/main/yamls/stage1_mae/large16.yaml)|[75.9](https://github.com/ml-jku/MAE-CT/blob/main/probes/mae_reimpl_large16.th)|[log](https://github.com/ml-jku/MAE-CT/blob/main/logs/probe/mae_large16.th)|60.6|
|[ViT-H/16](https://ml.jku.at/research/maect/download/mae_reimpl_huge16.th)|[hp](https://github.com/ml-jku/MAE-CT/blob/main/yamls/stage1_mae/huge16.yaml)|[78.0](https://github.com/ml-jku/MAE-CT/blob/main/probes/mae_reimpl_huge16.th)|[log](https://github.com/ml-jku/MAE-CT/blob/main/logs/probe/mae_huge16.th)|61.1|
|ViT-H/14|[original](https://github.com/facebookresearch/mae)|[77.2](https://github.com/ml-jku/MAE-CT/blob/main/probes/mae_huge14.th)|[log](https://github.com/ml-jku/MAE-CT/blob/main/logs/probe/mae_huge14.th)|58.9|

## MAE-CT

|Encoder|Pretrain|Probe|Probe|k-NN|
|:---:|:---:|:---:|:---:|:---:|
|[ViT-B/16](https://ml.jku.at/research/maect/download/maect_base16.th)|[hp](https://github.com/ml-jku/MAE-CT/blob/main/yamls/stage3_maect_contrastive_tuning/base16.yaml)|[73.5](https://github.com/ml-jku/MAE-CT/blob/main/probes/maect_base16.th)|[log](https://github.com/ml-jku/MAE-CT/blob/main/logs/probe/maect_base16.th)|64.1|
|[ViT-L/16](https://ml.jku.at/research/maect/download/maect_large16.th)|[hp](https://github.com/ml-jku/MAE-CT/blob/main/yamls/stage3_maect_contrastive_tuning/large16.yaml)|[80.2](https://github.com/ml-jku/MAE-CT/blob/main/probes/maect_large16.th)|[log](https://github.com/ml-jku/MAE-CT/blob/main/logs/probe/maect_large16.th)|78.0|
|[ViT-H/16](https://ml.jku.at/research/maect/download/maect_huge16.th)|[hp](https://github.com/ml-jku/MAE-CT/blob/main/yamls/stage3_maect_contrastive_tuning/huge16.yaml)|[81.5](https://github.com/ml-jku/MAE-CT/blob/main/probes/maect_huge16.th)|[log](https://github.com/ml-jku/MAE-CT/blob/main/logs/probe/maect_huge16.th)|79.4|
|[ViT-H/14](https://ml.jku.at/research/maect/download/maect_huge14.th)|[hp](https://github.com/ml-jku/MAE-CT/blob/main/yamls/stage3_maect_contrastive_tuning/huge14.yaml)|[81.3](https://github.com/ml-jku/MAE-CT/blob/main/probes/maect_huge14.th)|[log](https://github.com/ml-jku/MAE-CT/blob/main/logs/probe/maect_huge14.th)|79.1|
## MAE-CT<sub>*aug*</sub>

|Encoder|Pretrain|Probe|Probe|k-NN|
|:---:|:---:|:---:|:---:|:---:|
|[ViT-B/16](https://ml.jku.at/research/maect/download/maectaug_base16.th)|[hp](https://github.com/ml-jku/MAE-CT/blob/main/yamls/stage3_maectaug_contrastive_tuning/base16.yaml)|[76.9](https://github.com/ml-jku/MAE-CT/blob/main/probes/maectaug_base16.th)|[log](https://github.com/ml-jku/MAE-CT/blob/main/logs/probe/maectaug_base16.th)|73.4|
|[ViT-L/16](https://ml.jku.at/research/maect/download/maectaug_large16.th)|[hp](https://github.com/ml-jku/MAE-CT/blob/main/yamls/stage3_maectaug_contrastive_tuning/large16.yaml)|[81.5](https://github.com/ml-jku/MAE-CT/blob/main/probes/maectaug_large16.th)|[log](https://github.com/ml-jku/MAE-CT/blob/main/logs/probe/maectaug_large16.th)|79.1|
|[ViT-H/16](https://ml.jku.at/research/maect/download/maectaug_huge16.th)|[hp](https://github.com/ml-jku/MAE-CT/blob/main/yamls/stage3_maectaug_contrastive_tuning/huge16.yaml)|[82.2](https://github.com/ml-jku/MAE-CT/blob/main/probes/maectaug_huge16.th)|[log](https://github.com/ml-jku/MAE-CT/blob/main/logs/probe/maectaug_huge16.th)|79.8|
|[ViT-H/14](https://ml.jku.at/research/maect/download/maectaug_huge14.th)|[hp](https://github.com/ml-jku/MAE-CT/blob/main/yamls/stage3_maectaug_contrastive_tuning/huge14.yaml)|[82.0](https://github.com/ml-jku/MAE-CT/blob/main/probes/maectaug_huge14.th)|[log](https://github.com/ml-jku/MAE-CT/blob/main/logs/probe/maectaug_huge14.th)|78.9|
