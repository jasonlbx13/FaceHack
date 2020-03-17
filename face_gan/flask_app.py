from flask import Flask, render_template, request
import random
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import pickle
import time
import tensorflow as tf
import imageio
import os
import keras.backend.tensorflow_backend as KTF
import shutil
from tqdm import tqdm
import argparse
import pretrained_networks
import projector
import dataset_tool
from training import dataset
from training import misc
import bz2
from keras.utils import get_file
from ffhq_dataset.face_alignment import image_align
from ffhq_dataset.landmarks_detector import LandmarksDetector


class FaceHack():
    def __init__(self):
        self.network_pkl = './networks/normal_face.pkl'
        tflib.init_tf()
        self.session = tf.get_default_session()
        self.graph = tf.get_default_graph()
        with open(self.network_pkl, "rb") as f:
            self.generator_network, self.discriminator_network, self.Gs_network = pickle.load(f)
        self.noise_vars = [var for name, var in self.Gs_network.components.synthesis.vars.items() if
                           name.startswith('noise')]
        tflib.set_vars({var: np.random.randn(*var.shape.as_list()) for var in self.noise_vars})  # [height, width]
        self.Gs_syn_kwargs = dnnlib.EasyDict()
        self.Gs_syn_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        self.Gs_syn_kwargs.randomize_noise = False
        self.Gs_syn_kwargs.minibatch_size = 1
        self.truncation_psi = 0.5
        self.Gs_syn_kwargs.truncation_psi = self.truncation_psi


    def random_generate(self):

        with self.graph.as_default():
            with self.session.as_default():
                z = np.random.randn(1, *self.Gs_network.input_shape[1:])  # [minibatch, component]
                # Generate image
                images = self.Gs_network.run(z, None, **self.Gs_syn_kwargs)  # [minibatch, height, width, channel]
        PIL.Image.fromarray(images[0], 'RGB').save(
            dnnlib.make_run_dir_path('./static/random_face.jpg'))

    def make_app(self):
        app = Flask(__name__)

        @app.route('/')
        def hello():
            return render_template('base.html', the_title='Welcome FaceHack!')

        @app.route('/givemeaface')
        def random_face():

            self.random_generate()

            return render_template('random_face.html')


        return app

if __name__ == '__main__':

    facehack = FaceHack()
    app = facehack.make_app()
    app.run()