#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --gpus-per-task=a40:1
#SBATCH --time=2-00:00:00
#SBATCH --mem=64G
#SBATCH --account=shrikann_35

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate /home1/sbaruah/.conda/envs/story
cd /home1/sbaruah/narrative_understanding/chatter

size=51000
start_id=$(( $size * $1 ))
end_id=$(( $start_id + $size ))
echo $start_id $end_id
python flan_t5_attr_verify.py --batch_size=48 --start_id=$start_id --end_id=$end_id --gpu_id=0 --t5_model_size=xxl