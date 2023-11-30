#! /bin/bash

#SBATCH -A MST111109
#SBATCH -J run_test_tardal_tt
#SBATCH -p gp4d
#SBATCH -e twcc_logs/run_test_tardal_tt.txt
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gohyixian456@gmail.com

# 544715
python train.py --cfg config/official/infer/tardal-tt.yaml --save_dir experiments/tardal_tt_infer/20231129_default