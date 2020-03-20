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
from datetime import timedelta
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
            dnnlib.make_run_dir_path('./static/img/random_face.jpg'))

    def restore(self, npy_dir, img_dir):
        print (npy_dir)
        w = np.load(npy_dir)[np.newaxis, :]
        with self.graph.as_default():
            with self.session.as_default():
                images = self.Gs_network.components.synthesis.run(w, **self.Gs_syn_kwargs)
            PIL.Image.fromarray(images[0], 'RGB').save(dnnlib.make_run_dir_path(img_dir))


    def move_latent(self, npy_dir, Gs_network, Gs_syn_kwargs, *args):
        latent_vector = np.load(npy_dir)[np.newaxis, :]
        smile, age, gender, beauty, angleh, anglep, raceblack, raceyellow, racewhite = args
        new_latent_vector = latent_vector.copy()
        new_latent_vector[0][:8] = (latent_vector[0] + smile * self.smile_drt + age * self.age_drt + gender * self.gender_drt
                                    + beauty * self.beauty_drt + angleh * self.angleh_drt + anglep * self.anglep_drt
                                    + raceblack * self.raceblack_drt + raceyellow * self.raceyellow_drt + racewhite * self.racewhite_drt)[:8]
        with self.graph.as_default():
            with self.session.as_default():
                images = Gs_network.components.synthesis.run(new_latent_vector, **Gs_syn_kwargs)
        PIL.Image.fromarray(images[0], 'RGB').save(
            dnnlib.make_run_dir_path('./static/img/edit_face.jpg'))


    def make_app(self):
        app = Flask(__name__)
        app.send_file_max_age_default = timedelta(seconds=0.2)
        @app.route('/')
        def hello():
            return render_template('index.html', the_title='Welcome FaceHack!')

        @app.route('/guide')
        def guide():
            return render_template('guide.html')

        @app.route('/givemeaface', methods=['GET', 'POST'])
        def random_face():

            self.random_generate()
            return render_template('random_face.html')


        @app.route('/genface', methods=['GET', 'POST'])
        def genface():
            if request.method == 'POST':
                npy_dir = './static/npy_file/genface.npy'
                if os.path.exists(npy_dir):
                    os.remove(npy_dir)
                f = request.files['file']
                f.save(npy_dir)
                self.restore(npy_dir, './static/img/restore_face.jpg')
                return render_template('restore_face.html')
            return render_template('upload.html')

        @app.route('/editupload', methods=['GET', 'POST'])
        def edit_upload():
            if request.method == 'GET':
                return render_template('edit_upload.html')
            if request.method == 'POST':
                npy_dir = './static/npy_file/edit_face.npy'
                f = request.files['file']
                f.save(npy_dir)
                self.restore(npy_dir, './static/img/edit_face.jpg')
                return render_template('upload_down.html')


        @app.route('/edit', methods=['POST'])
        def edit_face():
            if request.method == 'POST':
                npy_dir = './static/npy_file/edit_face.npy'
                if len(request.form) != 0:
                    smile = float(request.form['smile'])
                    age = float(request.form['age'])
                    gender = float(request.form['gender'])
                    beauty = float(request.form['beauty'])
                    angleh = float(request.form['angleh'])
                    anglep = float(request.form['anglep'])
                    raceblack = float(request.form['raceblack'])
                    raceyellow = float(request.form['raceyellow'])
                    racewhite = float(request.form['racewhite'])
                    feature_book = [smile, age, gender, beauty, angleh, anglep, raceblack, raceyellow, racewhite]
                else:
                    feature_book = [0, 0, 0, 0, 0, 0, 0, 0, 0]
                self.move_latent(npy_dir, self.Gs_network, self.Gs_syn_kwargs, *feature_book)
                return render_template('edit_face.html')



        return app




if __name__ == '__main__':

    facehack = FaceHack()
    app = facehack.make_app()
    app.jinja_env.auto_reload = True
    app.run(host='0.0.0.0', port=8080, debug=True)