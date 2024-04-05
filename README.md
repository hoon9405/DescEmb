# Unifying Heterogeneous Electronic Health Record Systems via Clinical Text-Based Code Embedding
Kyunghoon Hur, Jiyoung Lee, Jungwoo Oh, Wesley Price, Young-Hak Kim, Edward Choi

This repository provides official Pytorch code to implement DescEmb, a code-agnostic EHR predictive model.

The paper can be found in this link:
[Unifying Heterogeneous Electronic Health Record Systems via Clinical Text-Based Code Embedding](https://arxiv.org/abs/2108.03625)

# Requirements

* [PyTorch](http://pytorch.org/) version >= 1.8.1
* Python version >= 3.7

# Getting started
## Prepare training data
First, download the dataset from these links: 

[MIMIC-III](https://physionet.org/content/iii/1.4/)

[eICU](https://physionet.org/content/eicu-crd/2.0/)

[ccs_multi_dx_tool_2015](https://www.hcup-us.ahrq.gov/toolssoftware/ccs/Multi_Level_CCS_2015.zip)

[icd10cmtoicd9gem](https://data.nber.org/gem/icd10cmtoicd9gem.csv)

Second, make directory sturcture like below:
```
data_input_path
├─ mimic
│  ├─ ADMISSIONS.csv
│  ├─ PATIENTS.csv
│  ├─ ICUSYAYS.csv
│  ├─ LABEVENTES.csv
│  ├─ PRESCRIPTIONS.csv
│  ├─ PROCEDURES.csv
│  ├─ INPUTEVENTS_CV.csv
│  ├─ INPUTEVENTS_MV.csv
│  ├─ D_ITEMDS.csv
│  ├─ D_ICD_PROCEDURES.csv
│  └─ D_LABITEMBS.csv
├─ eicu
│  ├─ diagnosis.csv
│  ├─ infusionDrug.csv
│  ├─ lab.csv
│  ├─ medication.csv
│  └─ patient.csv
├─ ccs_multi_dx_tool_2015.csv
└─ icd10cmtoicd9gem.csv

```
```
data_output_path
├─mimic
├─eicu
├─pooled
├─label
└─fold
```
Then run preprocessing code
```shell script
$ python preprocess_main.py
    --src_data $data
    --dataset_path $data_src_directory
    --dest_path $run_ready_directory 
```
Note that pre-processing takes about 1hours in 128 cores of AMD EPYC 7502 32-Core Processor, and requires 60GB of RAM.

# Examples
## Pre-training a model
### Pre-train a DescEmb model with Masked Language Modeling (MLM)

```shell script
$ python main.py \
    --distributed_world_size $WORLDSIZE \
    --input_path /path/to/data \
    --src_data $data \
    --task mlm \
    --mlm_prob $percent \
    --model $model
```

### Pre-train a CodeEmb model with Word2Vec

```shell script
$ python main.py \
    --distributed_world_size $WORLDSIZE \
    --input_path /path/to/data \
    --src_data $data \
    --task w2v
    --model codeemb
```
$data should be set to 'mimic' or 'eicu'

$percent should be set to probability (default: 0.3) of masking for MLM

$model should be set to 'descemb_bert' or 'descemb_rnn'

## Training a new model
Other configurations will set to be default, which were used in the DescEmb paper.

`$descemb` should be 'descemb_bert' or 'descemb_rnn'

`$ratio` should be set to one of [10, 30, 50, 70, 100] (default: 100)

`$value` should be set to one of ['NV', 'VA', 'DSVA', 'DSVA_DPE', 'VC']

`$task` should be set to one of ['readmission', 'mortality', 'los_3day', 'los_7day', 'diagnosis']

Note that `--input-path ` should be the root directory containing preprocessed data.
### Train a new CodeEmb model:

```shell script
$ python main.py \
    --distributed_world_size $WORLDSIZE \
    --input_path /path/to/data \
    --model ehr_model \
    --embed_model codeemb \
    --pred_model rnn \
    --src_data $data \
    --ratio $ratio \
    --value_mode $value \
    --task $task
```
### Train a new DescEmb model:

```shell script
$ python main.py \
    --disrtibuted_world_size $WORLDSIZE \
    --input_path /path/to/data \
    --model ehr_model \
    --embed_model $descemb \
    --pred_model rnn \
    --src_data $data \
    --ratio $ratio \
    --value_mode $value \
    --task $task
```
Note: if you want to train with pre-trained BERT model, add command line parameters `--init_bert_params` or `--init_bert_params_with_freeze`. `--init_bert_params_with_freeze` enables the model to load and freeze BERT parameters.

## Fine-tune a pre-trained model

### Fine-tune a pre-trained CodeEmb model:

```shell script
$ python main.py \
    --distributed_world_size $WORLDSIZE \
    --input_path /path/to/data \
    --model_path /path/to/model.pt \
    --load_pretrained_weights \
    --model ehr_model \
    --embed_model codeemb \
    --pred_model rnn \
    --src_data $data \
    --ratio $ratio \
    --value_mode $value \
    --task $task
```
### Fine-tune a pre-trained DescEmb model:
```shell script
$ python main.py \
    --distributed_world_size $WORLDSIZE \
    --input_path /path/to/data \
    --model_path /path/to/model.pt \
    --load_pretrained_weights \
    --model ehr_model \
    --embed_model $descemb \
    --pred_model rnn \
    --src_data $data \
    --ratio $ratio \
    --value_mode $value \
    --task $task
```

## Transfer a trained model
```shell script
$ python main.py \
    --distributed_world_size $WORLDSIZE \
    --input_path /path/to/data \
    --model_path /path/to/model.pt \
    --transfer \
    --model ehr_model \
    --embed_model $embed_model \
    --pred_model rnn \
    --src_data $data \
    --ratio $ratio \
    --value_mode $value \
    --task $task \
```
Note that `--embed_model` and `pred_model` should be matched with the transferred model.

# License
This repository is MIT-lincensed.

# Citation
Please cite as:
```
@misc{hur2021unifying,
      title={Unifying Heterogenous Electronic Health Records Systems via Text-Based Code Embedding}, 
      author={Kyunghoon Hur and Jiyoung Lee and Jungwoo Oh and Wesley Price and Young-Hak Kim and Edward Choi},
      year={2021},
      eprint={2108.03625},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
