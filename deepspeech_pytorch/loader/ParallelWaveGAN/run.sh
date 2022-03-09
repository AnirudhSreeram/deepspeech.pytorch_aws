for i in train #val
do
#echo "--load_path /data/asreeram/deepspeech.pytorch/data/LibriSpeech_dataset/$i/wav --save_path /data/asreeram/deepspeech.pytorch/data/LibriSpeech_dataset/train_melgan/wav --folder /data/asreeram/deepspeech.pytorch/melgan-neurips/models/linda_johnson.pt"

python3 gen_wave_gan.py --save_path /data/asreeram/deepspeech.pytorch/data/LibriSpeech_dataset/${i}_waveGAN/wav  --folder /data/asreeram/deepspeech.pytorch/data/LibriSpeech_dataset/train_clean_wav.scp   #${i}_clean/wav 

done


