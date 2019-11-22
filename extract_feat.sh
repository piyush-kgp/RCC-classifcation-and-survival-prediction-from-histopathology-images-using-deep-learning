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


module load cuda/9.0
module load cudnn/7-cuda-9.0
source v3env/bin/activate

for x in KIRC KIRP
do
	for y in train valid
  do
    for z in cancer normal
  	do
      echo "Extracting features for ${x} ${y} ${z}"
      python feature_extract.py --img_dir /ssd_scratch/cvit/piyush/${x}/${y}/${z} --npy_file_path /ssd_scratch/cvit/piyush/${x}_${y}_${z}.npy
      echo "RSyncing features for ${x} ${y} ${z}"
      rsync -aPz /ssd_scratch/cvit/piyush/${x}_${y}_${z}.npy delta_one@ada:/share1/delta_one/
      echo "RSync Successful for ${x} ${y} ${z}"
    done
	done
done
