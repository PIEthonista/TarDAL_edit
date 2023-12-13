#! /bin/bash

#SBATCH -A MST111109
#SBATCH -J run_test_tardal_tt_shipped_pretrained
#SBATCH -p gp4d
#SBATCH -e twcc_logs/run_test_tardal_tt_shipped_pretrained.txt
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gohyixian456@gmail.com

# 549528
# python infer.py --cfg config/official/infer/tardal-tt-shipped-fuse-only.yaml --save_dir experiments/tardal_tt/shipped_pretrained_fuse_only/infer

python infer.py --cfg config/official/infer/tardal-tt-shipped-fuse-only.yaml --save_dir experiments/tardal_tt/20231129_default/infer_all