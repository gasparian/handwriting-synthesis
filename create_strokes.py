import os
import pickle
import logging
import uuid
import time
import pickle 
import json

import numpy as np
import svgwrite

import drawing
#import lyrics
from tqdm import tqdm
from rnn import rnn
import tensorflow as tf
from PIL import Image, ImageDraw
from scipy.misc import imsave

#docker build -t handwriting-synthesis:latest .
#nvidia-docker run -v /home/temp:/home/imgs -v /home/handwriting-synthesis:/home/handwriting-synthesis -it --rm handwriting-synthesis bash

#drops isolated pixels
# se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
# se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
# img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, se1)
# img = cv2.morphologyEx(img, cv2.MORPH_OPEN, se2)

def coords2img(coords, width=3, autoscale=(64,64), offset=5):

    def min_max(coords):
        max_x, min_x = int(np.max(np.concatenate([coord[:, 0] for coord in coords]))), int(np.min(np.concatenate([coord[:, 0] for coord in coords]))) 
        max_y, min_y = int(np.max(np.concatenate([coord[:, 1] for coord in coords]))), int(np.min(np.concatenate([coord[:, 1] for coord in coords])))
        return min_x, max_x, min_y, max_y

    def dist(self, a, b):
        return np.power((np.power((a[0] - b[0]), 2) + np.power((a[1] - b[1]), 2)), 1./2)
    
    min_dists, dists = {}, [[] for i in range(len(coords))]
    for i, line in enumerate(coords):
        for point in line:
            dists[i].append(dist([0, 0], point))
        min_dists[min(dists[i])] = i
            
    min_dist = min(list(min_dists.keys()))
    min_index = min_dists[min_dist]
    start_point = coords[min_index][dists[min_index].index(min_dist)].copy()
    for i in range(len(coords)):
        coords[i] -= start_point
    min_x, max_x, min_y, max_y = min_max(coords) 
    scaleX = ((max_x - min_x) / (autoscale[0]-(offset*2-1)))
    scaleY = ((max_y - min_y) / (autoscale[1]-(offset*2-1)))
    for line in coords:
        line[:, 0] = line[:, 0] / scaleX
        line[:, 1] = line[:, 1] / scaleY

    min_x, max_x, min_y, max_y = min_max(coords)
        
    w = max_x-min_x+offset*2
    h = max_y-min_y+offset*2

    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)

    start = 1
    for i in range(len(coords)):
        for j in range(len(coords[i]))[start:]:
            x, y = coords[i][j-1]
            x_n, y_n = coords[i][j]
            x -= min_x-offset; y -= min_y-offset
            x_n -= min_x-offset; y_n -= min_y-offset
            draw.line([(x,y), (x_n,y_n)], fill="black", width=width)

    return img

class Hand(object):

    def __init__(self, path, length):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        self.nn = rnn(
            log_dir='logs',
            checkpoint_dir='checkpoints',
            prediction_dir='predictions',
            learning_rates=[.0001, .00005, .00002],
            batch_sizes=[32, 64, 64],
            patiences=[1500, 1000, 500],
            beta1_decays=[.9, .9, .9],
            validation_batch_size=32,
            optimizer='rms',
            num_training_steps=100000,
            warm_start_init_step=17900,
            regularization_constant=0.0,
            keep_prob=1.0,
            enable_parameter_averaging=False,
            min_steps_to_checkpoint=2000,
            log_interval=20,
            logging_level=logging.CRITICAL,
            grad_clip=10,
            lstm_size=400,
            output_mixture_components=20,
            attention_mixture_components=10
        )
        self.nn.restore()
        self.path = path
        self.counter = {}
        self.prt = 1
        self.length = length

    def write(self, filename, lines, biases=None, styles=None, stroke_colors=None, stroke_widths=None):
        valid_char_set = set(drawing.alphabet)
        for line_num, line in enumerate(lines):
            if len(line) > 75:
                raise ValueError(
                    (
                        "Each line must be at most 75 characters. "
                        "Line {} contains {}"
                    ).format(line_num, len(line))
                )

            for char in line:
                if char not in valid_char_set:
                    raise ValueError(
                        (
                            "Invalid character {} detected in line {}. "
                            "Valid character set is {}"
                        ).format(char, line_num, valid_char_set)
                    )

        strokes = self._sample(lines, biases=biases, styles=styles)
        self.offsets2coords(strokes, filename)

    def _sample(self, lines, biases=None, styles=None):
        num_samples = len(lines)
        max_tsteps = 40*max([len(i) for i in lines])
        biases = biases if biases is not None else [0.5]*num_samples

        x_prime = np.zeros([num_samples, 1200, 3])
        x_prime_len = np.zeros([num_samples])
        chars = np.zeros([num_samples, 120])
        chars_len = np.zeros([num_samples])

        if styles is not None:
            for i, (cs, style) in enumerate(zip(lines, styles)):
                x_p = np.load('styles/style-{}-strokes.npy'.format(style))
                c_p = np.load('styles/style-{}-chars.npy'.format(style)).tostring().decode('utf-8')

                c_p = str(c_p) + " " + cs
                c_p = drawing.encode_ascii(c_p)
                c_p = np.array(c_p)

                x_prime[i, :len(x_p), :] = x_p
                x_prime_len[i] = len(x_p)
                chars[i, :len(c_p)] = c_p
                chars_len[i] = len(c_p)

        else:
            for i in range(num_samples):
                encoded = drawing.encode_ascii(lines[i])
                chars[i, :len(encoded)] = encoded
                chars_len[i] = len(encoded)

        [samples] = self.nn.session.run(
            [self.nn.sampled_sequence],
            feed_dict={
                self.nn.prime: styles is not None,
                self.nn.x_prime: x_prime,
                self.nn.x_prime_len: x_prime_len,
                self.nn.num_samples: num_samples,
                self.nn.sample_tsteps: max_tsteps,
                self.nn.c: chars,
                self.nn.c_len: chars_len,
                self.nn.bias: biases
            }
        )
        samples = [sample[~np.all(sample == 0.0, axis=1)] for sample in samples]
        return samples

    def offsets2coords(self, offsets, filename):
        offsets = drawing.offsets_to_coords(offsets[0])
        offsets = drawing.denoise(offsets)
        offsets[:, :2] = drawing.align(offsets[:, :2])
        offsets[:, 1] *= -1
        offsets[:, :2] -= offsets[:, :2].min()
        detachments = [-1]+list(np.where(offsets[:, 2])[0])
        coords = np.array([offsets[detachments[i]+1:detachments[i+1], :2] for i in range(len(detachments)-1)])

        self.counter.update({filename:coords})
        current_length = len(self.counter)

        print(self.counter, current_length)

        if current_length % 5000 == 0 or current_length == self.length:
            self.prt += 1
            pickle.dump(self.counter, open(self.path+'_prt_%s.pickle.dat' % self.prt, 'wb'))
            self.counter = {}

if __name__ == '__main__':
    with tf.device('/gpu:0'):
        start = time.time()

        path = '/home/imgs/pics_strokes'
        #words = [i[:-1] for i in open("/home/imgs/words.txt").readlines()]
        words = ["0102", "hello world!", "1) 2) 3)"]

        biases = [.75]
        styles = [9]
        stroke_colors = ['black']
        stroke_widths = [3]

        words_count = len(words)*len(biases)*len(styles)*len(stroke_colors)*len(stroke_widths)
        hand = Hand(path=path, length=words_count)

        for line in tqdm(words, desc='words'):
            for style in styles: 
                for bias in biases:
                    hand.write(
                        filename='%s_b%s_s%s' % (line, bias, style),
                        lines=[line],
                        biases=[bias],
                        styles=[style],
                        stroke_colors=stroke_colors,
                        stroke_widths=stroke_widths)

        print('Prediction time: %i words, %s s' % (words_count, time.time()-start))
        os.system('find %s -name "*.pickle.dat" | exec tar -czvf %s.tar.gz -T -' % ('/'.join(path.split('/')[:-1])+'/', path))