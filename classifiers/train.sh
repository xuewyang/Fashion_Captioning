DATA_FOLDER='/home/xuewyang/Xuewen/Research/data/FACAD/jsons'
MODEL_FOLDER='/home/xuewyang/Xuewen/Research/model/fashion/bert'

CUDA_VISIBLE_DEVICES='1' python trainer.py --batch-size 200 \
--data_folder $DATA_FOLDER --category_file $DATA_FOLDER/category_count_129927.json \
--model_folder $MODEL_FOLDER