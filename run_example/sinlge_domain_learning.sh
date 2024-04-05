# Run single domain learning

# DescEmb (text encoder as rnn-> descemb_rnn or bert -> descemb_ert)
CUDA_VISIBLE_DEVICES=3 python main.py \
    --distributed_world_size 1 \
    --input_path 'preprocess dest_path' \ 
    # input_path must be preproces destination path
    --model ehr_model \
    --embed_model descemb_rnn \
    --pred_model rnn \
    --src_data mimiciii \
    --ratio 100 \
    --value_mode NV \
    --task mortality ;


CUDA_VISIBLE_DEVICES=3 python main.py \
    --distributed_world_size 1 \
    --input_path 'preprocess dest_path' \ 
    # input_path must be preproces destination path
    --model ehr_model \
    --embed_model descemb_bert \
    --pred_model rnn \
    --src_data mimiciii \
    --ratio 100 \
    --value_mode NV \
    --task mortality ;

    

# CodeEmb
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
    --task mortality ;


# Value mode (NV, VA, DSVA, DPE, VC) 
CUDA_VISIBLE_DEVICES=3 python main.py \
    --distributed_world_size 1 \
    --input_path 'preprocess dest_path' \ 
    # input_path must be preproces destination path
    --model ehr_model \
    --embed_model codeemb \
    --pred_model rnn \
    --src_data mimiciii \
    --ratio 100 \
    --value_mode VC \
    --task mortality ;

CUDA_VISIBLE_DEVICES=3 python main.py \
    --distributed_world_size 1 \
    --input_path 'preprocess dest_path' \ 
    # input_path must be preproces destination path
    --model ehr_model \
    --embed_model descemb_rnn \
    --pred_model rnn \
    --src_data mimiciii \
    --ratio 100 \
    --value_mode DSVA_DPE \
    --task mortality ;