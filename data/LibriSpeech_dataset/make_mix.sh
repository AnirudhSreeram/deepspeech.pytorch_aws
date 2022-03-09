no_clean_samples=$1 # "2-way-split" or "3-way-split"
split_ratio=$2      # provide split in the integer value like 50, 70 or 100 clean

# ? perform looping over both train and validation set

for folder in train val; do # folders "train" and "val"

	if [ $no_clean_samples == "2-way-split" ]; then # how many splits
		echo "Spliting ${no_clean_samples}"

		other_split=$((100 - ${split_ratio}))

		# ! check if the split is between 0 and 100
		if [ $split_ratio -le 100 ]; then
			echo "split good"
		else
			echo "split should be less than or equal to 100"
			exit
		fi
		path="/data/asreeram/deepspeech.pytorch/data/LibriSpeech_dataset/waveGAN_mix_${split_ratio}_${other_split}_${folder}"
		if [ ! -d $path ]; then
			echo "Creating the path to the folder"
			mkdir -p $path
			mkdir -p ${path}/wav
			cp -r /data/asreeram/deepspeech.pytorch/data/LibriSpeech_dataset/mix_70-30_${folder}/txt $path
		fi
		# * Split training data into 2 pieces and perform the splitting of data
		if [ $folder == "train" ]; then
			split=$(((28539 * ${split_ratio}) / 100))
			sp=$(((28539 * ${other_split}) / 100))
		else
			split=$(((2703 * ${split_ratio}) / 100))
			sp=$(((2703 * ${other_split}) / 100))
		fi
		echo "SPLITS ==== $split and $sp"
		$(head -n $split ${folder}_clean_wav.scp >temp_train_clean) # generate temp files for copying
		$(tail -n $sp ${folder}_clean_wav.scp >temp_train_melgan)   # generate temp files for copying

		# * Copy the data appropriately clean
		while read line; do
			cp -r ${folder}_clean/wav/$line waveGAN_mix_${split_ratio}_${other_split}_${folder}/wav/$line # perform copying of audio files to location
		done <temp_train_clean

		# * Copy the data appropriately	melgan
		while read line; do
			cp -r ${folder}_waveGAN/wav/$line waveGAN_mix_${split_ratio}_${other_split}_${folder}/wav/$line
		done <temp_train_melgan
		# ! remove the necessary temp file generated
		rm temp_train_clean temp_train_melgan

	else
		# ? Perform 3 way split using the clean, melgan and wgn audio files
		echo "Spliting ${no_clean_samples}"
		path="/data/asreeram/deepspeech.pytorch/data/LibriSpeech_dataset/mix_3ws_${folder}/wav/"
		if [ ! -d $path ]; then
			echo "Creating the path to the folder"
			mkdir -p $path
		fi
		# ? Split training data into 2 pieces and perform the splitting of data
		if [ $folder == "train" ]; then
			sp1=$(((28539 * 34) / 100))
			sp2=$(((28539 * 33) / 100))
			sp3=$(((28539 * 33) / 100))
			sps=$((sp1 + sp2))
		else
			sp1=$(((2703 * 34) / 100))
			sp2=$(((2703 * 33) / 100))
			sp3=$(((2703 * 33) / 100))
			sps=$((sp1 + sp2))
		fi

		# * Create splits
		$(head -n $sp1 ${folder}_clean_wav.scp >temp_train_clean)
		$(head -n $sps ${folder}_clean_wav.scp | tail -n $sp2 >temp_train_wgn)
		$(tail -n $sp3 ${folder}_clean_wav.scp >temp_train_melgan)

		#* Copy to appropriate folders
		while read line; do
			cp -r ${folder}_clean/wav/$line mix_3ws_${folder}/wav/$line
		done <temp_train_clean

		while read line; do
			cp -r ${folder}_wgn/wav/$line mix_3ws_${folder}/wav/$line
		done <temp_train_wgn

		while read line; do
			cp -r ${folder}_melgan/wav/$line mix_3ws_${folder}/wav/$line
		done <temp_train_melgan

		# ! Remove the temp files generated
		rm temp_train_clean temp_train_melgan temp_train_wgn
	fi
done
