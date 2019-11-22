#!/bin/bash
#SBATCH -A delta_one
#SBATCH --reservation=non-deadline-queue
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=2048
#SBATCH --mail-type=END
#SBATCH --time 1-00:00:00
#SBATCH -w gnode22

# mkdir -p /ssd_scratch/cvit/piyush/
# mkdir -p /ssd_scratch/cvit/piyush/KIRC/
# mkdir -p /ssd_scratch/cvit/piyush/KIRC/train/
# mkdir -p /ssd_scratch/cvit/piyush/KIRC/valid/
# rsync -aPz delta_one@ada:/share1/dataset/medic_kidney/KIRC/patches/train/ /ssd_scratch/cvit/piyush/KIRC/train/
# rsync -aPz delta_one@ada:/share1/dataset/medic_kidney/KIRC/patches/test/ /ssd_scratch/cvit/piyush/KIRC/train/
# rsync -aPz delta_one@ada:/share1/dataset/medic_kidney/KIRC/patches/valid/ /ssd_scratch/cvit/piyush/KIRC/valid/

module load cuda/9.0
module load cudnn/7-cuda-9.0
source v3env/bin/activate

python classifier.py --img_dir /ssd_scratch/cvit/piyush/KIRC/train/ \
                     --val_dir /ssd_scratch/cvit/piyush/KIRC/valid/ \
                     --num_epochs 30 \
                     --log_dir kirc_logs/  \
                     --model_checkpoint checkpoints/KIRC_model_epoch_0.pth \
                     --optimzer_checkpoint checkpoints/KIRC_optimizer_epoch_0.pth \
                     --save_prefix KIRC > trg_log_kirc.txt
sleep 5
