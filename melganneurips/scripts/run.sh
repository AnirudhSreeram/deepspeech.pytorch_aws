


for i in train_clean/wav #val_clean/wav
do
	
	python generate_from_folder.py --load_path /data/asreeram/deepspeech.pytorch/data/LibriSpeech_dataset/$i --save_path /data/asreeram/deepspeech.pytorch/data/LibriSpeech_dataset/train/wav --folder /data/asreeram/deepspeech.pytorch/melgan-neurips/models/linda_johnson.pt 

done
