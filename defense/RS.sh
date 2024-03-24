# Please change the name of the "ATTACK_METHOD" to eval your method!
# You can run this file directly!
ATTACK_METHOD=mifgsm
model = resnet18
INPUT_DIR=../checkpoint/adv_img/${ATTACK_METHOD}/${model}
LABEL_FILE=../dataset/labels.csv
CHECKPOINT_PATH=/defense/models/rs_imagenet/resnet50/noise_0.50/checkpoint.pth.tar
GPU_ID='0'

python rs/predict.py "${INPUT_DIR}" "${LABEL_FILE}" "${CHECKPOINT_PATH}"  0.50 prediction_outupt --alpha 0.001 --N 1000 --skip 100 --batch 1 --GPU_ID $GPU_ID # --targeted



# '''
# python defense/rs/predict.py /path/to/adv_data /path/to/noise_0.50/checkpoint.pth.tar  0.50 prediction_outupt --alpha 0.001 --N 1000 --skip 100 --batch 1
# '''