import warnings
warnings.filterwarnings("ignore")

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import tensorflow as tf
tf.get_logger().setLevel('INFO')

import logging, os
os.environ['KMP_WARNINGS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.disable(logging.WARNING)

from os.path import join
import sys
import io
import numpy as np
import torch
import argparse
import time
from pathlib import Path

import json
from scipy.io.wavfile import write

import sacremoses
import subword_nmt
from unidecode import unidecode
#from fairseq.models.transformer import TransformerModel

project_name = 'tacotron2'
sys.path.append(project_name)
sys.path.append(join(project_name, 'waveglow/'))

from hparams import create_hparams
from model import Tacotron2
from text import text_to_sequence
from glow import WaveGlow

# setting
torch.set_grad_enabled(False)

tacotron_model='model/checkpoint_11500'
waveglow_model='model/waveglow/waveglow_256channels_universal_v5.pt'

class Synthesizer:

    def load(self, tacotron_model, waveglow_model):
        # setting
        self.project_name = 'tacotron2'
        sys.path.append(self.project_name)
        sys.path.append(join(self.project_name, 'waveglow/'))

        # initialize Tacotron2
        self.hparams = create_hparams()
        self.hparams.sampling_rate = 22050
        self.hparams.max_decoder_steps = 1000
        self.hparams.fp16_run = True

        self.tacotron = Tacotron2(self.hparams)
        self.tacotron.load_state_dict(torch.load(tacotron_model)['state_dict'])
        _ = self.tacotron.cuda().eval()

        self.waveglow = torch.load(waveglow_model)['model']
        self.waveglow = self.waveglow.remove_weightnorm(self.waveglow)
        _ = self.waveglow.cuda().eval()
        for k in self.waveglow.convinv:
            k.float()

    def __init__(self, tacotron, waveglow, text):
        self.tacotron = tacotron
        self.waveglow = waveglow
        self.text = text

    def read_text(self):
        with open(self.text, 'r') as rf:
            return rf.readlines()

    def run(self):
        self.load(self.tacotron, self.waveglow)
        for i, text in enumerate(self.read_text(),start=1):
            self.synthesize(text, i)

    def synthesize(self, text, i):

        sequence = np.array(text_to_sequence(text, ['transliteration_cleaners']))[None, :]
        sequence = torch.from_numpy(sequence).to(device='cuda', dtype=torch.int64)
        sequence = sequence.cuda()

        with torch.no_grad():
            _, mel, _, _ = self.tacotron.inference(sequence)
            audio = self.waveglow.infer(mel, sigma=0.666)
            audio = audio[0].data.cpu().numpy()
            audio *= 32767 / max(0.01, np.max(np.abs(audio)))

        self.out = io.BytesIO()
        write('wavs/%02d.wav' %i, self.hparams.sampling_rate, audio.astype(np.int16))
        print('%02d.wav' %i)

        return self.out.getvalue()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--tacotron2', default=tacotron_model, help='Full path to tacotron model')
    parser.add_argument('--waveglow', default=waveglow_model, help='Full path to waveglow model')
    parser.add_argument('--port', type=int, default=80)
    parser.add_argument('--hparams', default='tacotron2/hparams.py',
                        help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    args = parser.parse_args()

    os.environ['KMP_WARNINGS'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    synthesizer = Synthesizer(args.tacotron, args.waveglow, 'txt/test.txt')
    synthesizer.run()
