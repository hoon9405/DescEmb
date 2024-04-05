# pooled learning should set eval_data
CUDA_VISIBLE_DEVICES=3 python main.py \
    --distributed_world_size 1 \
    --input_path preprocess dest_path \
    --model ehr_model \
    --embed_model descemb_bert \
    --pred_model rnn \
    --src_data pooled \
    --eval_data mimiciii \
    --ratio 100 \
    --value_mode DSVA_DPE \
    --task mortality ;

CUDA_VISIBLE_DEVICES=3 python main.py \
    --distributed_world_size 1 \
    --input_path 'preprocess dest_path' \
    --model ehr_model \
    --embed_model descemb_bert \
    --pred_model rnn \
    --src_data pooled \
    --eval_data eicu \
    --ratio 100 \
    --value_mode DSVA_DPE \
    --task mortality ;