#!/bin/bash
#SBATCH -A iicd
#SBATCH -J JDEngel
#     SBATCH -t 12:00:00
#SBATCH -t 5-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jde2149@columbia.edu
#SBATCH --nodes=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4 
#SBATCH --mem=32gb

source /burg/opt/anaconda3-2022.05/etc/profile.d/conda.sh
conda activate talk_face_env

module load cuda11.1/toolkit
export PYTHONPATH="/burg/home/jde2149/talking_face:$PYTHONPATH"
export PATH=$HOME/ffmpeg/bin:$PATH
python preprocessing/preprocess_audio_video.py --data_root /burg/iicd/users/jde2149/data/mvlrs_v1/main --preprocessed_root /burg/iicd/users/jde2149/data/mvlrs_v1/main_preprocessed > pyoutput_preprocess.txt

conda deactivate
