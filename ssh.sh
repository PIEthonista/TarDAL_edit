#! /bin/bash

eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519_yx
eval "$(conda shell.bash hook)"
conda activate pix2pix_simple
wandb login --relogin 6b305360cd440b1f7432d6f1ba4d81e0c0a60536