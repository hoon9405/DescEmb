# Unifying Heterogeneous Electronic Health Record Systems via Clinical Text-Based Code Embedding
Kyunghoon Hur, Jiyoung Lee, Jungwoo Oh, Wesley Price, Young-Hak Kim, Edward Choi



This repository provides official Pytorch code to implement DescEmb, a code-agnostic EHR predictive model. [[ArXiv](https://arxiv.org/abs/2108.03625)]



## Execution

Run Model

```
python main.py \
--embed_model=descemb\  # 'codeemb' for CodeEmb
--data=mimic\  # dataset name, 'pooled' for pooled learning
--eval_data=mimic\
--value_embed_type=DSVA_DPE\  # value embedding method
--task=readmission\
--enc_embed_dim=128\   # text encoder model embedding dimension
--enc_hidden_dim=256\  # text encoder model hidden dimension
--rnn_layer=1\    # number of layers for predictive model
--pred_embed_dim=128\  # predictive model embedding dimension
--pred_hidden_dim=256\  # predictiv emodel hidden dimension
--load_pretrained_weights\  # if finetune
--bert_model=bert_tiny\  # text encoder BERT model to load
--init_bert_params\ 

```



Run Masked Language Model (MLM)

```
python main.py \
--data=mimic\
--eval_data=mimic\
--task=mlm\
--mlm_prob=0.3\  # ratio of MLM
--model_path=pretrain.pt  # saving filename
```



Run Word2Vec

```
python main.py \
--data=mimic\
--eval_data=mimic\
--task=w2v\
--model_path=pretrain.pt  # saving filename
```



Run Transfer Learning

```
python main.py \
--data=mimic\
--eval_data=eicu\
--value_embed_type=DSVA_DPE\  # value embedding method
--task=readmission\  # task
--transfer
```

