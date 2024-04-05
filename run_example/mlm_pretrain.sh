# MLM training value mode should be set on NV
CUDA_VISIBLE_DEVICES=3 python main.py \
    --distributed_world_size 1 \
    --input_path 'preprocess dest_path' \ 
    # input_path must be preproces destination path
    --model descemb_bert \
    --src_data mimiciii \
    --ratio 100 \
    --value_mode NV \
    --task mlm ;


# W2V training 
CUDA_VISIBLE_DEVICES=3 python main.py \
    --distributed_world_size 1 \
    --input_path 'preprocess dest_path' \ 
    # input_path must be preproces destination path
    --model ehr_model \
    --embed_model codeemb \
    --pred_model rnn \
    --src_data mimiciii \
    --ratio 100 \
    --value_mode NV \
    --task w2v ;




