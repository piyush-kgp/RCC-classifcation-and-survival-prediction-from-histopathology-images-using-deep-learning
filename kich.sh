#!/bin/bash
#SBATCH -A delta_one
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=1024
#SBATCH --mail-type=END
#SBATCH --time 1-00:00:00
#SBATCH -w gnode26

module load cuda/9.0
module load cudnn/7-cuda-9.0
source /home/delta_one/v3env/bin/activate

python /home/delta_one/project/histopathology/classifier.py \
                     --img_dir /ssd_scratch/cvit/medicalImaging/PATCHES_KICH/train/ \
                     --val_dir /ssd_scratch/cvit/medicalImaging/PATCHES_KICH/valid/ \
                     --num_classes 2 \
                     --classes cancer normal \
                     --num_epochs 10 \
                     --log_dir kich_logs/  \
                     --batch_size 32 \
                     --save_prefix KICH > trg_log_kich.txt
sleep 5
