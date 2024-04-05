python ../preprocess/preprocess_main.py \
    --src_data mimiciii \
    --dataset_path /user/mimiciii \
    --dest_path /user/descemb/dataset ;

python ../preprocess/preprocess_main.py \
    --src_data eicu \
    --dataset_path /user/eicu \
    --dest_path /user/descemb/dataset ;


python ../preprocess/preprocess_main.py \
    --src_data pooled \
    --dest_path /user/descemb/dataset ;



python ../preprocess/preprocess_main.py \
    --src_data mimiciii \
    --dest_path /user/descemb/dataset \
    --data_type pretrain ;

python ../preprocess/preprocess_main.py \
    --src_data eicu \
    --dest_path /user/descemb/dataset \
    --data_type pretrain ;

python ../preprocess/preprocess_main.py \
    --src_data pooled \
    --dest_path /user/descemb/dataset \
    --data_type pretrain ;


