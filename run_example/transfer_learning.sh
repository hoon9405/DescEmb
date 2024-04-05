# transfer learning should set --transfer and --model_path
CUDA_VISIBLE_DEVICES=3 python main.py \
    --distributed_world_size 1 \
    --input_path 'preprocess dest_path' \ 
    # model_path must be checkpoint file path from model trained by source data
    # E.g. set model_path to model trained with mimiciii and run this script with src_data eicu
    # means using model trained with mimiciii transfer to train with eicu
    --model_path '/path/to/model.pt' \
    --model ehr_model \
    --embed_model descemb_bert \
    --pred_model rnn \
    --src_data mimiciii \
    --ratio 100 \
    --value_mode DSVA_DPE \
    --task mortality ;



# few-shot transfer learning should set ratio [10, 30, 50, 70, 90]
CUDA_VISIBLE_DEVICES=3 python main.py \
    --distributed_world_size 1 \
    --input_path 'preprocess dest_path' \ 
    # model_path must be checkpoint file path from model trained by source data
    # E.g. set model_path to model trained with mimiciii and run this script with src_data eicu
    # means using model trained with mimiciii transfer to train with eicu
    --model_path '/path/to/model.pt' \
    --model ehr_model \
    --embed_model descemb_bert \
    --pred_model rnn \
    --src_data mimiciii \
    --ratio 10 \
    --value_mode DSVA_DPE \
    --task mortality ;