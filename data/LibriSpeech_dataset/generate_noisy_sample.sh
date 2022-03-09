source /data/asreeram/miniconda3/etc/profile.d/conda.sh
conda activate deeppy

# while read line; do
#     python /data/asreeram/deepspeech.pytorch/noise_inject.py --input-path "/data/asreeram/deepspeech.pytorch/data/LibriSpeech_dataset/val_clean/wav/$line" \
#         --noise-path "/data/asreeram/deepspeech.pytorch/noise_dir/WGN/White.wav" \
#         --output-path "/data/asreeram/deepspeech.pytorch/data/LibriSpeech_dataset/val_wgn/wav/$line"
# done <val_clean_wav.scp

while read line; do
    python /data/asreeram/deepspeech.pytorch/noise_inject.py --input-path "/data/asreeram/deepspeech.pytorch/data/LibriSpeech_dataset/train_clean/wav/$line" \
        --noise-path "/data/asreeram/deepspeech.pytorch/noise_dir/WGN/White.wav" \
        --output-path "/data/asreeram/deepspeech.pytorch/data/LibriSpeech_dataset/train_wgn/wav/$line"
done <train_clean_wav.scp
