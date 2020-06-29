生成音檔
----------------------------------------------------------------------------------------------------------
利用synthesizer.py檔案來生成音檔
synthesizer.py：以下三行要調整資料夾位置
self.project_name = 'tacotron2'
sys.path.append(self.project_name)
sys.path.append(join(self.project_name, 'waveglow/'))

這個資料夾裡要有
train資料夾(裡面放有音檔.wav)
model資料夾(裡面放有train.py所生成的模型)
txt資料夾(裡面有測試的.txt)
tacotron2資料夾(裡面是原本git clone https://github.com/yfliao/taiwanese_tonal_tlpa_tacotron2.git的檔案)
在tacorron2裡還有一個waveglow資料夾(裡面是git clone https://github.com/NVIDIA/waveglow.git所產生的檔案)

