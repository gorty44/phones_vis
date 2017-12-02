data_root = 'data/'
sr = 48000
total_limit = None # set this to 100 to export 100 samples
length_limit = sr//8 # set this to sr/4 to only export 250ms of audio per sample

from os.path import join
import numpy as np
from utils import * 

samples = np.load(join(data_root, 'samples.npy'))

y = samples[:total_limit, :length_limit].reshape(-1)
ffmpeg_save_audio(data_root + 'spritesheet.mp3', y, sr)

