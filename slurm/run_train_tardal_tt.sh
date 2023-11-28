#! /bin/bash

#SBATCH -A MST111109
#SBATCH -J run_train_tardal_tt
#SBATCH -p gp4d
#SBATCH -e twcc_logs/run_train_tardal_tt.txt
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gohyixian456@gmail.com

# 544613
python train.py --cfg config/official/train/tardal-tt.yaml --auth 6b305360cd440b1f7432d6f1ba4d81e0c0a60536 --run_name tardal_tt_20231129_default