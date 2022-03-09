


for i in val train
do
echo "--load_path /data/asreeram/deepspeech.pytorch/data/LibriSpeech_dataset/$i/wav --save_path /data/asreeram/deepspeech.pytorch/data/LibriSpeech_dataset/train_melgan/wav --folder /data/asreeram/deepspeech.pytorch/melgan-neurips/models/linda_johnson.pt"

python3 generate_from_folder.py --load_path /data/asreeram/deepspeech.pytorch/melgan-neurips/models/multi_speaker.pt --save_path /data/asreeram/deepspeech.pytorch/data/LibriSpeech_dataset/${i}_melgan/wav  --folder /data/asreeram/deepspeech.pytorch/data/LibriSpeech_dataset/${i}_clean/wav 

done
