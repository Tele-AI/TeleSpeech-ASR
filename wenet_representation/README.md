# 表征训练下游任务

## 框架简介
* 框架基于wenet搭建，支持从预训练模型提取提取中间层表征，作为ASR模型的特征输入，或生成离散单元作为离散ASR模型的输入进行训练
* 支持ASR模型/解码方式：Conformer、Transformer以及对应的ctc_greedy_search、attention、attention_rescore等方式
* 表征训练方法与ASR模型解耦，可添加其他下游ASR模型如Paraformer、Branchformer等

## 参数设置

### 连续表征训练相关
* conf/train_d2v2_ark_conformer.yaml
  * frontend_conf：预训练模型相关
    * finetune_model: 设为true则表示moder_dir中的模型为经过finetune后的模型
    * multilayer_feature：选择是否使用多层表征加权计算，与layer设置共同使用
    * layer：从0开始选择使用指定层表征计算，与multilayer_feature配合共4种：(1) 若为[-1]，且multilayer_feature=true，使用全部层的加权和；(2) 若为[-1]，且multilayer_feature=false，只使用最后一层的表征；(3) 若layer为指定数值，且multilayer_feature=true，则只在指定几层表征上计算加权和；(4) 若layer为指定数值，且multilayer_feature=false，则对指定层层表征求平均值
    * num_layer：模型整体层数，预训练模型为层数+1，finetune模型为层数
  * preencoder_conf：降维相关，如注释掉则表示不使用
  * spec_aug_after_conf：表征层drop_out相关，如注释掉则表示不使用
    * scale：是否根据维度对mask长度进行放缩
  * encoder_conf：
    * input_layer：添加了2倍降采的选项conv2d2
  * dataset_conf：
    * cmvn：对送入预训练模型的数据进行normalize，与预训练时设置要匹配
    * max_length：根据默认positional_encoding长度5000设置，data2vec预训练模型4倍降采，则最大长度可以到20000；若input_layer使用conv2d2，则max_length*2
    * speed_perturb、spec_aug：mfcc特征输入的方式不采用

### 离散单元训练相关
* discrete_token/dump_feat.sh
  * train_km_set：kmeans model训练所用数据集
  * percent：随机挑选$\text{percent} \in [0,1]$占比数据进行训练，-1则代表全部数据训练kmeans model
  * input_type：支持原始音频wav.scp格式文件或wenet data.list格式的文件输入
  * feat_save_type：表征保存格式，支持npy与kaldi格式
* discrete_token/kmeans_d2v.yaml
  * reader_conf设置与conf/train_d2v2_ark_conformer.yaml中frontend_conf设置基本一致
    * max_chunk：如果设置，则在提取预训练模型表征时会对音频按max_chunk进行截断，显存较小时可启用
    * weights：各层表征权重，multilayer_feature为true时可使用，提取的预训练模型表征由指定层表征变为根据权重求和选择；权重可由连续表征训练得到
  * kmeans_conf：MiniBatchKMeans参数设置
    * n_clusters：聚类簇数
* conf/train_d2v2_discrete_conformer.yaml
  * frontend_conf：离散表征相关
    * input_size：与kmeans时n_clusters设置相同
    * padding_idx：训练时的padding
  * 其余设置与conf/train_d2v2_ark_conformer.yaml相同
---