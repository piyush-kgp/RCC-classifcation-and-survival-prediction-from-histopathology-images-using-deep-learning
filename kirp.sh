#!/bin/bash
#SBATCH -A delta_one
#SBATCH --reservation=non-deadline-queue
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2048
#SBATCH --mail-type=END
#SBATCH --time 1-00:00:00
#SBATCH -w gnode22

# mkdir -p /ssd_scratch/cvit/piyush/
# mkdir -p /ssd_scratch/cvit/piyush/KIRP/
# rsync -aPzq delta_one@ada:/share1/dataset/medic_kidney/KIRP/patches/ /ssd_scratch/cvit/piyush/KIRP/

module load cuda/9.0
module load cudnn/7-cuda-9.0
source v3env/bin/activate

python classifier.py --img_dir /ssd_scratch/cvit/piyush/KIRP/train/ \
                     --val_dir /ssd_scratch/cvit/piyush/KIRP/valid/ \
                     --num_epochs 30 \
                     --log_dir kirp_logs/ \
                     --model_checkpoint checkpoints/KIRP_model_epoch_0.pth \
                     --optimzer_checkpoint checkpoints/KIRP_optimizer_epoch_0.pth \
                     --save_prefix KIRP > trg_log_kirp.txt
sleep 5
