#! /bin/bash

#SBATCH -A MST111109
#SBATCH -J run_test_tardal_ct
#SBATCH -p gp4d
#SBATCH -e twcc_logs/run_test_tardal_ct.txt
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gohyixian456@gmail.com

# 544715
python train.py --cfg config/official/infer/tardal-ct.yaml --save_dir experiments/tardal_ct_infer/20231129_default