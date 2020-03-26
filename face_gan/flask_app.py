from flask import Flask, render_template, request, send_from_directory
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
import cv2
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
        self.angry_drt = np.load('latent_directions/emotion_angry.npy')
        self.sad_drt = np.load('latent_directions/emotion_sad.npy')
        self.eye_drt = np.load('latent_directions/eyes_open.npy')


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

    def style_transform(self, pic1, pic2, alpha):
        pic1[9:] = (1 - alpha) * pic1[9:] + alpha * pic2[9:]
        with self.graph.as_default():
            with self.session.as_default():
                img = self.Gs_network.components.synthesis.run(pic1[np.newaxis, :], **self.Gs_syn_kwargs)
        img = PIL.Image.fromarray(img[0])
        img.save('./static/img/transform_face.jpg')

    def merge(self, pic1, pic2, alpha):
        pic1[:] = (1 - alpha) * pic1[:] + alpha * pic2[:]
        with self.graph.as_default():
            with self.session.as_default():
                img = self.Gs_network.components.synthesis.run(pic1[np.newaxis, :], **self.Gs_syn_kwargs)
        img = PIL.Image.fromarray(img[0])
        img.save('./static/img/merge_face.jpg')

    def enhance(self, pic):
        with self.graph.as_default():
            with self.session.as_default():
                for i in range(18):
                    scale1 = random.uniform(-10., 5.)
                    scale2 = random.uniform(-5., 5.)
                    scale3 = random.uniform(-10., 10.)
                    scale4 = random.uniform(-10., 10.)
                    scale5 = random.uniform(-15., 15.)
                    scale6 = random.uniform(-15., 15.)
                    latent_vector = pic[np.newaxis, :]
                    new_latent_vector = latent_vector.copy()
                    new_latent_vector[0][:8] = (latent_vector[0] + scale1 * self.smile_drt + scale2 * self.eye_drt
                                                + scale3 * self.angleh_drt + scale4 * self.anglep_drt
                                                + scale5 * self.angry_drt + scale6 * self.sad_drt)[:8]
                    img = self.Gs_network.components.synthesis.run(new_latent_vector, **self.Gs_syn_kwargs)
                    img = PIL.Image.fromarray(img[0])
                    img.save('./static/img/enhance/{}.png'.format(str(i).zfill(4)))

    def puzzle(self, img_dir, h, l):
        img_name_book = sorted(os.listdir(img_dir))
        print(img_name_book)
        img_book = [cv2.resize(cv2.imread('{}/{}'.format(img_dir, img_name)), (512, 512)) for img_name in img_name_book]
        for i in range(h):
            tmp = img_book[i * l]
            for j in range(1, l):
                tmp = np.hstack((tmp, img_book[i * l + j]))
            if i == 0:
                res = tmp
            else:
                res = np.vstack((res, tmp))
        cv2.imwrite('./static/img/enhance/puzzle.jpg', res)

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
                if request.form['select']=='None':
                    f = request.files['file']
                    f.save(npy_dir)
                    self.restore(npy_dir, './static/img/restore_face.jpg')
                else:
                    name = request.form['select']
                    self.restore('./static/npy_file/{}.npy'.format(name), './static/img/restore_face.jpg')
                return render_template('restore_face.html')
            return render_template('upload.html')

        @app.route('/editupload', methods=['GET', 'POST'])
        def edit_upload():
            if request.method == 'GET':
                return render_template('edit_upload.html')
            if request.method == 'POST':
                npy_dir = './static/npy_file/edit_face.npy'
                if request.form['select'] == 'None':
                    f = request.files['file']
                    f.save(npy_dir)
                    self.restore(npy_dir, './static/img/edit_face.jpg')
                else:
                    name = request.form['select']
                    self.restore('./static/npy_file/{}.npy'.format(name), './static/img/edit_face.jpg')
                    npfile = np.load('./static/npy_file/{}.npy'.format(name))
                    np.save(npy_dir, npfile)
                return render_template('upload_done.html')

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

        @app.route('/tfupload', methods=['GET', 'POST'])
        def tfupload():
            if request.method == 'GET':
                return render_template('tfupload.html')
            if request.method == 'POST':
                npy_dir1 = './static/npy_file/trans_src.npy'
                npy_dir2 = './static/npy_file/trans_dst.npy'
                if request.form['select1'] == 'None':
                    f1 = request.files['file1']
                    f1.save(npy_dir1)
                    self.restore(npy_dir1, './static/img/trans_src.jpg')
                else:
                    name1 = request.form['select1']
                    self.restore('./static/npy_file/{}.npy'.format(name1), './static/img/trans_src.jpg')
                    npfile1 = np.load('./static/npy_file/{}.npy'.format(name1))
                    np.save(npy_dir1, npfile1)
                if request.form['select2'] == 'None':
                    f2 = request.files['file2']
                    f2.save(npy_dir2)
                    self.restore(npy_dir2, './static/img/trans_dst.jpg')
                else:
                    name2 = request.form['select2']
                    self.restore('./static/npy_file/{}.npy'.format(name2), './static/img/trans_dst.jpg')
                    npfile2 = np.load('./static/npy_file/{}.npy'.format(name2))
                    np.save(npy_dir2, npfile2)
                return render_template('upload_done2.html')

        @app.route('/tfupload2', methods=['GET', 'POST'])
        def tfupload2():
            if request.method == 'GET':
                return render_template('tfupload2.html')
            if request.method == 'POST':
                npy_dir1 = './static/npy_file/merge_src.npy'
                npy_dir2 = './static/npy_file/merge_dst.npy'
                if request.form['select1'] == 'None':
                    f1 = request.files['file1']
                    f1.save(npy_dir1)
                    self.restore(npy_dir1, './static/img/merge_src.jpg')
                else:
                    name1 = request.form['select1']
                    self.restore('./static/npy_file/{}.npy'.format(name1), './static/img/merge_src.jpg')
                    npfile1 = np.load('./static/npy_file/{}.npy'.format(name1))
                    np.save(npy_dir1, npfile1)
                if request.form['select2'] == 'None':
                    f2 = request.files['file2']
                    f2.save(npy_dir2)
                    self.restore(npy_dir2, './static/img/merge_dst.jpg')
                else:
                    name2 = request.form['select2']
                    self.restore('./static/npy_file/{}.npy'.format(name2), './static/img/merge_dst.jpg')
                    npfile2 = np.load('./static/npy_file/{}.npy'.format(name2))
                    np.save(npy_dir2, npfile2)
                return render_template('upload_done3.html')

        @app.route('/transform', methods=['POST'])
        def transform():
            pic1 = np.load('./static/npy_file/trans_src.npy')
            pic2 = np.load('./static/npy_file/trans_dst.npy')
            if len(request.form) != 0:
                alpha = float(request.form['alpha'])
            else:
                alpha = 0
            self.style_transform(pic1, pic2, alpha)
            return render_template('transform.html')

        @app.route('/merge', methods=['POST'])
        def merge_face():
            pic1 = np.load('./static/npy_file/merge_src.npy')
            pic2 = np.load('./static/npy_file/merge_dst.npy')
            if len(request.form) != 0:
                alpha = float(request.form['alpha'])
            else:
                alpha = 0
            self.merge(pic1, pic2, alpha)
            return render_template('merge.html')

        @app.route('/enhanceupload', methods=['GET', 'POST'])
        def enhance_upload():
            if request.method == 'GET':
                return render_template('enhance_upload.html')
            if request.method == 'POST':
                npy_dir = './static/npy_file/enhance_face.npy'
                if request.form['select'] == 'None':
                    f = request.files['file']
                    f.save(npy_dir)
                    self.restore(npy_dir, './static/img/enhance_face.jpg')
                else:
                    name = request.form['select']
                    self.restore('./static/npy_file/{}.npy'.format(name), './static/img/enhance_face.jpg')
                    npfile = np.load('./static/npy_file/{}.npy'.format(name))
                    np.save(npy_dir, npfile)
                return render_template('upload_done4.html')

        @app.route('/enhance', methods=['GET', 'POST'])
        def enhance_face():
            pic = np.load('./static/npy_file/enhance_face.npy')
            self.enhance(pic)
            self.puzzle('./static/img/enhance', 3, 6)
            return render_template('enhance.html')

        @app.route('/download')
        def download():
            return send_from_directory("./tmp/", filename="npy_file.tar", as_attachment=True)

        return app


if __name__ == '__main__':

    facehack = FaceHack()
    app = facehack.make_app()
    app.jinja_env.auto_reload = True
    app.run(host='0.0.0.0', port=8080, debug=True)