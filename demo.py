import os
import logging
import uuid

import numpy as np
import svgwrite

import drawing
#import lyrics
from rnn import rnn
import tensorflow as tf
from PIL import Image, ImageDraw
from scipy.misc import imsave

#docker build -t handwriting-synthesis:latest .
#nvidia-docker run -v /home/temp:/home/imgs -v /home/handwriting-synthesis:/home/handwriting-synthesis -it --rm handwriting-synthesis bash

class Hand(object):

    def __init__(self):
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
        #self._draw(strokes, lines, filename, stroke_colors=stroke_colors, stroke_widths=stroke_widths)
        self._draw(strokes, lines, filename, width=3)
        #self.coords2img(strokes, filename, autoscale=(64,64), width=3, offset=5)

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

    # def _draw(self, strokes, lines, filename, stroke_colors=None, stroke_widths=None):
    #     stroke_colors = stroke_colors or ['black']*len(lines)
    #     stroke_widths = stroke_widths or [2]*len(lines)

    #     line_height = 60
    #     view_width = 1000
    #     view_height = line_height*(len(strokes) + 1)

    #     dwg = svgwrite.Drawing(filename=filename)
    #     dwg.viewbox(width=view_width, height=view_height)
    #     dwg.add(dwg.rect(insert=(0, 0), size=(view_width, view_height), fill='white'))

    #     initial_coord = np.array([0, -(3*line_height / 4)])
    #     for offsets, line, color, width in zip(strokes, lines, stroke_colors, stroke_widths):

    #         if not line:
    #             initial_coord[1] -= line_height
    #             continue

    #         offsets[:, :2] *= 1.5
    #         strokes = drawing.offsets_to_coords(offsets)
    #         strokes = drawing.denoise(strokes)
    #         strokes[:, :2] = drawing.align(strokes[:, :2])

    #         strokes[:, 1] *= -1
    #         strokes[:, :2] -= strokes[:, :2].min() + initial_coord
    #         strokes[:, 0] += (view_width - strokes[:, 0].max()) / 2

    #         prev_eos = 1.0
    #         p = "M{},{} ".format(0, 0)
    #         for x, y, eos in zip(*strokes.T):
    #             p += '{}{},{} '.format('M' if prev_eos == 1.0 else 'L', x, y)
    #             prev_eos = eos
    #         path = svgwrite.path.Path(p)
    #         path = path.stroke(color=color, width=width, linecap='round').fill("none")
    #         dwg.add(path)

    #         initial_coord[1] -= line_height

    #     dwg.save()

    def _draw(self, strokes, lines, filename, width=3, offset=5):

        line_height = 60
        initial_coord = np.array([0, -(3*line_height / 4)])
        for offsets, line in zip(strokes, lines):

            if not line:
                initial_coord[1] -= line_height
                continue

            offsets[:, :2] *= 1.5
            strokes = drawing.offsets_to_coords(offsets)
            strokes = drawing.denoise(strokes)
            strokes[:, :2] = drawing.align(strokes[:, :2])

            strokes[:, 1] *= -1
            strokes[:, :2] -= strokes[:, :2].min() + initial_coord
            # strokes[:, 0] += (view_width - strokes[:, 0].max()) / 2

            view_width = (strokes[:, 0].max() - strokes[:, 0].min()) + offset*2
            view_height = (strokes[:, 1].max() - strokes[:, 1].min()) + offset*2

            dwg = svgwrite.Drawing(filename=filename)
            dwg.viewbox(width=view_width, height=view_height)
            dwg.add(dwg.rect(insert=(0, 0), size=(view_width, view_height), fill='white'))

            prev_eos = 1.0
            p = "M{},{} ".format(0, 0)
            for x, y, eos in zip(*strokes.T):
                p += '{}{},{} '.format('M' if prev_eos == 1.0 else 'L', x, y)
                prev_eos = eos
            path = svgwrite.path.Path(p)
            path = path.stroke(color='black', width=width, linecap='round').fill("none")
            dwg.add(path)

            initial_coord[1] -= line_height

        dwg.save()

    def dist(self, a, b):
        return np.power((np.power((a[0] - b[0]), 2) + np.power((a[1] - b[1]), 2)), 1./2)

    def coords2img(self, strokes, filename, width=3, autoscale=(64,64), offset=5):

        def min_max(coords):
            max_x, min_x = int(np.max(np.concatenate([coord[:, 0] for coord in coords]))), int(np.min(np.concatenate([coord[:, 0] for coord in coords]))) 
            max_y, min_y = int(np.max(np.concatenate([coord[:, 1] for coord in coords]))), int(np.min(np.concatenate([coord[:, 1] for coord in coords])))
            return min_x, max_x, min_y, max_y

        for coords in strokes:
            coords = drawing.offsets_to_coords(coords)
            coords = drawing.denoise(coords)
            coords[:, :2] = drawing.align(coords[:, :2])
            coords[:, 1] *= -1
            coords[:, :2] -= coords[:, :2].min()
            
            detachments = [-1]+list(np.where(coords[:, 2])[0])
            coords = np.array([coords[detachments[i]+1:detachments[i+1], :2] for i in range(len(detachments)-1)])
            
            min_dists, dists = {}, [[] for i in range(len(coords))]
            for i, line in enumerate(coords):
                for point in line:
                    dists[i].append(self.dist([0, 0], point))
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

            imsave(filename, np.array(img).astype(np.uint8))

if __name__ == '__main__':
    with tf.device('/gpu:0'):
        hand = Hand()
        #words = [i[:-1] for i in open("/home/imgs/words.txt").readlines()]
        words = ["0102", "hello world!", "1) 2) 3)"]

        lines = {word:str(uuid.uuid4()) for word in words}

        biases = [.75 for i in lines]
        styles = [9 for i in lines]
        stroke_colors = ['black']
        stroke_widths = [3]

        for key, value in lines.items():
            for style in styles: 
                for bias in biases:
                    hand.write(
                        filename='/home/imgs/%s.svg' % value,
                        #filename='/home/imgs/%s.png' % value,
                        lines=[key],
                        biases=[bias],
                        styles=[style],
                        stroke_colors=stroke_colors,
                        stroke_widths=stroke_widths
                    )