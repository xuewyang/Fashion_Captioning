DATA_FOLDER=/home/xuewyang/Xuewen/Research/data/FACAD/images/
MODEL_FOLDER=/home/xuewyang/Xuewen/Research/model/fashion/captioning/newfc

CUDA_VISIBLE_DEVICES='1' python tools/eval.py --input_json $DATA_FOLDER/cocotest.json  --num_images -1 \
--model $MODEL_FOLDER/model.pth --infos_path $MODEL_FOLDER/infos_fc.pkl --language_eval 1