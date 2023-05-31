import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os

import numpy as np
import torchaudio
import librosa
import librosa.display
import math
import keras
from glob import glob
import math
import matplotlib as plt
import tensorflow as tf
def get_friendly_class(y_class):
    if y_class[0] == 1 and y_class[1] == 0:
        return 'Laugh'
    if y_class[0] == 0 and y_class[1] == 1:
        return 'Dialog'

class RawClip3(object):
    featureFuncs = ['tonnetz', 'spectral_rolloff', 'spectral_contrast',
                    'spectral_bandwidth', 'spectral_flatness', 'mfcc',
                    'chroma_cqt', 'chroma_cens', 'melspectrogram']

    def __init__(self, sourcefile, Y_class=None):
        self.y, self.sr = torchaudio.load(sourcefile)
        self.laughs = None
        self.Y_class = Y_class

    def resample(self, rate, channel):
        return librosa.resample(self.y.T[channel], self.sr, rate)

    def amp(self, rate=22050, n_fft=2048, channel=0):
        D = librosa.amplitude_to_db(librosa.magphase(librosa.stft(
            self.resample(rate, channel), n_fft=n_fft))[0], ref=np.max)
        return D

    def _extract_feature(self, func):
        method = getattr(librosa.feature, func)

        # Construct params for each 'class' of features
        params = {'y': self.raw}
        if 'mfcc' in func:
            params['sr'] = self.sr
            params['n_mfcc'] = 128
        if 'chroma' in func:
            params['sr'] = self.sr

        feature = method(**params)

        return feature

    def _split_features_into_windows(self, data, duration):
        # Apply a moving window
        windows = []

        # Pad the rightmost edge by repeating frames, simplifies stretching
        # the model predictions to the original audio later on.
        data = np.pad(data, [[0, duration], [0, 0]], mode='edge')
        for i in range(data.shape[0] - duration):
            windows.append(data[i:i+duration])

        return np.array(windows)

    def build_features(self, duration=30, milSamplesPerChunk=10):
        # Extract features, one chunk at a time (to reduce memory required)
        # Tip: about 65 million samples for a normal-length episode
        # 10 million samples results in around 1.5GB to 2GB memory use
        features = []

        chunkLen = milSamplesPerChunk * 1000000
        numChunks = math.ceil(self.y.shape[0] / chunkLen)

        for i in range(numChunks):
            # Set raw to the current chunk, for _extract_feature
            self.raw = self.y.T[0][i * chunkLen:(i+1)*chunkLen]

            # For this chunk, run all of our feature extraction functions
            # Each returned array is in the shape (features, steps)
            # Use concatenate to combine (allfeatures, steps)
            chunkFeatures = np.concatenate(
                list(
                    map(self._extract_feature, self.featureFuncs)
                    )
                )
            features.append(chunkFeatures)


        # Transform to be consistent with our LSTM expected input
        features = np.concatenate(features, axis=1).T
        # Combine our chunks along the time-step axis.
        features = self._split_features_into_windows(features, duration)

        return features
    

class DataSet(object):
    def __init__(self, datapath, laughPrefix='/ff*.wav', dialogPrefix='/dd*.wav'):
        self.clips = []
        for y_class, files in [[1., 0.], glob(datapath + laughPrefix)], [[0., 1.], glob(datapath + dialogPrefix)]:
            for ff in files:
                self.clips.append(RawClip3(ff, y_class))
        np.random.seed(seed=0)
        self.X, self.Y_class = self._get_samples()
        self.idx_train, self.idx_cv, self.idx_test = self.split_examples_index(len(self.Y_class))

    def split_examples_index(self, total):
        """Returns shuffled index for 60/20/20 split of train, cv, test"""
        np.random.seed(seed=0)
        idx = np.random.choice(total, size=total, replace=False, )

        #60/20/20 split
        train = idx[0:int(total*0.6)]
        cv    = idx[int(total*0.6):int(total*0.6) + int(total*0.2)]
        test  = idx[int(total*0.8):]

        return train, cv, test

    def _get_samples(self):
        X = []
        y = []
        for clip in self.clips:
            for s in clip.build_features():
                X.append(s)
                y.append(clip.Y_class)
                
        return np.array(X), np.array(y)
    
def predict_graph_clip(filename, model):
    # Load the clip, extract features, and run the model
    rc2 = RawClip3(filename, Y_class=None)
    X = rc2.build_features()
    spec = rc2.amp()
    classes = model.predict(X)
    
    # Plot the results
    plt.figure(figsize=(12,2))
    
    # Spectra
    axes = librosa.display.specshow(spec, y_axis='log', x_axis='frames', cmap='gist_gray')
    
    # Setup second x and y axes
    ax2 = axes.twinx()
    ax2.set_ylabel('Prediction')
    ax3 = ax2.twiny()
    ax3.margins(0,0.1)
    
    # Plot the laugh class as a line graph
    #g = ax3.plot(x2[:,1], linewidth=2, color=[0.8,0.8,0.8])
    g = ax3.plot(classes[:,0], linewidth=2, color=[0.0,1.0,1.0])
    
    # Add a title
    plt.xticks([])
    plt.title('%s spectra and laugh classification' % filename)
    
if __name__ == '__main__':
    print("Started")
    # os.environ['CUDA_VISIBLE_DEVICES'] ="0"
    
    
    model = keras.models.load_model('/home/prsood/projects/def-whkchun/prsood/laughr/assets/trained-model.h5')
    print("model loaded")
    f = "/home/prsood/projects/def-whkchun/prsood/multi-modal-emotion/data/final_context_videos/1_60_c.mp4"
    print(predict_graph_clip(f, model))