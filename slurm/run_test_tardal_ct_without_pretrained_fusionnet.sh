#! /bin/bash

#SBATCH -A MST111109
#SBATCH -J run_test_tardal_ct_without_pretrained_fusionnet
#SBATCH -p gp4d
#SBATCH -e twcc_logs/run_test_tardal_ct_without_pretrained_fusionnet.txt
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gohyixian456@gmail.com

# 547124
python infer.py --cfg config/official/infer/tardal-ct.yaml --save_dir experiments/tardal_ct/20231130_default_without_pretrained_fusionnet/infer --eval