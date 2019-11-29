#!/bin/bash
#SBATCH -A delta_one
#SBATCH -n 10
#SBATCH -p short
#SBATCH --mem-per-cpu=2048
#SBATCH --mail-type=END
#SBATCH --time 1-00:00:00
#SBATCH -w gnode01

source /home/delta_one/project/histopathology
python dag_svm.py \
--kirc_train_file /home/delta_one/project/histopathology/cancer_patches_features/train_KIRC.npy \
--kirp_train_file /home/delta_one/project/histopathology/cancer_patches_features/train_KIRP.npy \
--kich_train_file /home/delta_one/project/histopathology/cancer_patches_features/train_KICH.npy \
--kirc_valid_file /home/delta_one/project/histopathology/cancer_patches_features/valid_KIRC.npy \
--kirp_valid_file /home/delta_one/project/histopathology/cancer_patches_features/valid_KIRP.npy \
--kich_valid_file /home/delta_one/project/histopathology/cancer_patches_features/valid_KICH.npy > dag_svm.txt
