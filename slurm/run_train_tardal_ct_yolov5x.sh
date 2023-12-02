#! /bin/bash

#SBATCH -A MST111109
#SBATCH -J run_train_tardal_ct_yolov5x
#SBATCH -p gp4d
#SBATCH -e twcc_logs/run_train_tardal_ct_yolov5x.txt
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gohyixian456@gmail.com

# 546470
python train.py --cfg config/official/train/tardal-ct-yolov5x.yaml --auth 6b305360cd440b1f7432d6f1ba4d81e0c0a60536 --run_name tardal_ct_20231130_yolov5x