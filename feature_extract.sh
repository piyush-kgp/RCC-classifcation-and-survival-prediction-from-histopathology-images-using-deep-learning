#!/bin/bash
#SBATCH -A delta_one
#SBATCH -n 32
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=1024
#SBATCH --mail-type=END
#SBATCH --time 1-00:00:00
#SBATCH -w gnode22


module load cuda/9.0
module load cudnn/7-cuda-9.0
source /home/delta_one/v3env/bin/activate

# for x in PATCHES PATCHES_KIRP PATCHES_KICH
# do
# 	for y in train valid test
#   do
#     for z in cancer
#   	do
#       echo "Extracting features for ${x} ${y} ${z}"
#       python feature_extract.py --img_dir /ssd_scratch/cvit/medicalImaging/${x}/${y}/${z} \
# 																--npy_file_path /ssd_scratch/medicalImaging/features/${x}_${y}_${z}.npy \
# 																--model_checkpoint /ssd_scratch/medicalImaging/ckpts/SUBTYPE_model_epoch_6.pth
#     done
# 	done
# done


for y in train valid test
do
  for z in KIRC KIRP KICH
	do
    echo "Extracting features for ${y} ${z}"
    python feature_extract.py --img_dir /ssd_scratch/cvit/medicalImaging/subtype_classification/${y}/${z}/cancer \
															--npy_file_path ${y}_${z}.npy \
															--model_checkpoint /home/delta_one/project/histopathology/exports/subtype_FC_Layer_4_1/SUBTYPE_model_epoch_7.pth
  done
done
