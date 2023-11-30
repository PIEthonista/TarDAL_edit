#! /bin/bash

#SBATCH -A MST111109
#SBATCH -J run_test_tardal_dt
#SBATCH -p gp4d
#SBATCH -e twcc_logs/run_test_tardal_dt.txt
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gohyixian456@gmail.com

# 544715
python train.py --cfg config/official/infer/tardal-dt.yaml --save_dir experiments/tardal_dt_infer/20231129_default