#! /bin/bash

#SBATCH -A MST111109
#SBATCH -J run_test_tardal_ct_yolov5x
#SBATCH -p gp4d
#SBATCH -e twcc_logs/run_test_tardal_c_yolov5x.txt
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gohyixian456@gmail.com

# 547028
python infer.py --cfg config/official/infer/tardal-ct-yolov5x.yaml --save_dir experiments/tardal_ct_yolov5x/20231130/infer --eval