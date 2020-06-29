使用git clone https://github.com/yfliao/taiwanese_tonal_tlpa_tacotron2.git 將檔案git clone 到這個資料夾裡面
--------------------------------------------------------------------------------------------------------------
hparams.py：以下五行做更動，將檔案位置更改
load_mel_from_disk=False,
training_files='filelists/train-filelist.txt',
validation_files='filelists/eval-filelist.txt',
text_cleaners=['transliteration_cleaners'],
max_wav_value=1.0,
--------------------------------------------------------------------------------------------------------------
