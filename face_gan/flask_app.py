from flask import Flask, render_template, request
import random
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import pickle
import time
import tensorflow as tf
import urllib
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
        self.network_pkl = './networks/star_face.pkl'
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

        self.smile_drt = np.load('latent_directions/smile.npy')
        self.age_drt = np.load('latent_directions/age.npy')
        self.gender_drt = np.load('latent_directions/gender.npy')
        self.beauty_drt = np.load('latent_directions/beauty.npy')
        self.angleh_drt = np.load('latent_directions/angle_horizontal.npy')
        self.anglep_drt = np.load('latent_directions/angle_pitch.npy')
        self.raceblack_drt = np.load('latent_directions/race_black.npy')
        self.raceyellow_drt = np.load('latent_directions/race_yellow.npy')
        self.racewhite_drt = np.load('latent_directions/race_white.npy')
        self.glasses_drt = np.load('latent_directions/glasses.npy')


    def random_generate(self):

        with self.graph.as_default():
            with self.session.as_default():
                z = np.random.randn(1, *self.Gs_network.input_shape[1:])  # [minibatch, component]
                # Generate image
                images = self.Gs_network.run(z, None, **self.Gs_syn_kwargs)  # [minibatch, height, width, channel]
        PIL.Image.fromarray(images[0], 'RGB').save(
            dnnlib.make_run_dir_path('./static/random_face.jpg'))

    # def move_latent(latent_vector, Gs_network, Gs_syn_kwargs):
    #     new_latent_vector = latent_vector.copy()
    #     new_latent_vector[0][:8] = (latent_vector[0] + smile * smile_drt + age * age_drt + gender * gender_drt
    #                                 + beauty * beauty_drt + angleh * angleh_drt + anglep * anglep_drt
    #                                 + raceblack * raceblack_drt + raceyellow * raceyellow_drt + racewhite * racewhite_drt
    #                                 + glasses * glasses_drt)[:8]
    #     images = Gs_network.components.synthesis.run(new_latent_vector, **Gs_syn_kwargs)
    #     result = PIL.Image.fromarray(images[0], 'RGB')
    #     return result

    def make_app(self):
        app = Flask(__name__)

        @app.route('/')
        def hello():
            return render_template('base.html', the_title='Welcome FaceHack!')

        @app.route('/givemeaface', methods=["POST","GET"])
        def random_face():

            self.random_generate()
            return render_template('random_face.html')

        @app.route('/edit', methods=['POST'])
        def edit_face():
            face_dir = urllib.request.unquote(request.form.get('face_dir'))
            w = np.load(face_dir)[np.newaxis, :]
            with self.graph.as_default():
                with self.session.as_default():
                    image = self.Gs_network.components.synthesis.run(w, **self.Gs_syn_kwargs)
            img = PIL.Image.fromarray(image[0], 'RGB')
            save_dir = urllib.request.unquote(request.form.get('save_dir'))
            img.save(save_dir)
            return 'done'
        @app.route('/genface', methods=["POST","GET"])
        def genface():
            print(request.form)  # 格式 ImmutableMultiDict([('username', '123'), ('pwd', '123')])
            print(request.form.to_dict())  # 格式 {'username': '123', 'pwd': '123'}
            return render_template("test.html")

        return app




if __name__ == '__main__':

    facehack = FaceHack()
    app = facehack.make_app()
    app.run(host='0.0.0.0', port=8080)