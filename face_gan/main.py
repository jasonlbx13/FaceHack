# Thanks to StyleGAN2 provider —— Copyright (c) 2019, NVIDIA CORPORATION.
# Thanks the work by BUPT_GWY, seeprettyface.com.
import random
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import pickle
import imageio
import os
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

smile_drt = np.load('latent_directions/smile.npy')
age_drt = np.load('latent_directions/age.npy')
gender_drt = np.load('latent_directions/gender.npy')
beauty_drt = np.load('latent_directions/beauty.npy')
angleh_drt = np.load('latent_directions/angle_horizontal.npy')
anglep_drt = np.load('latent_directions/angle_pitch.npy')
raceblack_drt = np.load('latent_directions/race_black.npy')
raceyellow_drt = np.load('latent_directions/race_yellow.npy')
racewhite_drt = np.load('latent_directions/race_white.npy')
glasses_drt = np.load('latent_directions/glasses.npy')
angry_drt = np.load('latent_directions/emotion_angry.npy')
sad_drt = np.load('latent_directions/emotion_sad.npy')
eye_drt = np.load('latent_directions/eyes_open.npy')


def text_save(file, data):  # save generate code, which can be modified to generate customized style
    for i in range(len(data[0])):
        s = str(data[0][i])+'\n'
        file.write(s)

def png2gif(args):
    print ('starting generate gif')
    images = []
    filenames=sorted((fn for fn in os.listdir(args.tmp_dir) if fn.endswith('.png')))
    for filename in filenames:
        images.append(imageio.imread(args.tmp_dir+'/'+filename))
    imageio.mimsave('{}/result.gif'.format(args.dst_dir), images, duration=0.04)
    print ('gif done')

def generate_images(args):

    os.makedirs(args.dst_dir, exist_ok=True)
    os.makedirs('{}/images/'.format(args.dst_dir), exist_ok=True)
    os.makedirs('{}/generate_zcodes/'.format(args.dst_dir), exist_ok=True)
    os.makedirs('{}/generate_wcodes/'.format(args.dst_dir), exist_ok=True)
    print('Loading networks from "{}"...'.format(args.network_pkl))
    tflib.init_tf()
    with open(args.network_pkl, "rb") as f:
        generator_network, discriminator_network, Gs_network = pickle.load(f)

    Gs_syn_kwargs = dnnlib.EasyDict()
    Gs_syn_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_syn_kwargs.randomize_noise = False
    Gs_syn_kwargs.minibatch_size = 1

    for i in tqdm(range(int(args.num))):

        # Generate random latent
        w_avg = Gs_network.get_var('dlatent_avg')
        noise_vars = [var for name, var in Gs_network.components.synthesis.vars.items() if name.startswith('noise')]
        z = np.random.randn(1, *Gs_network.input_shape[1:])
        tflib.set_vars({var: np.random.randn(*var.shape.as_list()) for var in noise_vars})
        # Generate dlatent
        w = Gs_network.components.mapping.run(z, None)
        w = w_avg + (w - w_avg) * args.truncation_psi

        # Save latent
        if args.save_latent:
            ztxt_filename = '{}/generate_zcodes/{}.txt'.format(args.dst_dir, str(i).zfill(4))
            with open(ztxt_filename, 'w') as f:
                text_save(f, z)

        # Save dlatent
        if args.save_dlatent:
            wtxt_filename = '{}/generate_wcodes/{}.npy'.format(args.dst_dir, str(i).zfill(4))
            np.save(wtxt_filename, w[0])

        # Generate image

        images = Gs_network.components.synthesis.run(w, **Gs_syn_kwargs)

        # Save image
        PIL.Image.fromarray(images[0], 'RGB').save(dnnlib.make_run_dir_path('{}/images/{}.png'.format(args.dst_dir, str(i).zfill(4))))

def edit_images(args):
    os.makedirs(args.dir, exist_ok=True)
    print('Loading networks from "{}"...'.format(args.network_pkl))
    tflib.init_tf()
    with open(args.network_pkl, "rb") as f:
        generator_network, discriminator_network, Gs_network = pickle.load(f)

    Gs_syn_kwargs = dnnlib.EasyDict()
    Gs_syn_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_syn_kwargs.randomize_noise = False
    Gs_syn_kwargs.minibatch_size = 1
    if args.edit_mod == 'mul_images':
        for vector_dir in tqdm(sorted(os.listdir(args.src_dir))):
            latent_vector = np.load(args.src_dir+'/'+vector_dir)[np.newaxis, :]
            new_latent_vector = latent_vector.copy()
            new_latent_vector[0][:8] = (latent_vector[0] + float(args.smile) * smile_drt + float(args.age) * age_drt + float(args.gender) * gender_drt
                                        + float(args.beauty) * beauty_drt + float(args.angleh) * angleh_drt + float(args.anglep) * anglep_drt
                                        + float(args.raceblack) * raceblack_drt + float(args.raceyellow) * raceyellow_drt + float(args.racewhite) * racewhite_drt
                                        + float(args.glasses) * glasses_drt)[:8]
            images = Gs_network.components.synthesis.run(new_latent_vector, **Gs_syn_kwargs)
            # Save image
            PIL.Image.fromarray(images[0], 'RGB').save(dnnlib.make_run_dir_path('{}/{}_edit.png'.format(args.dst_dir, vector_dir[:-4])))

    elif args.edit_mod == 'one_image':
        latent_vector = np.load(args.src_dir)[np.newaxis, :]
        new_latent_vector = latent_vector.copy()
        new_latent_vector[0][:8] = (latent_vector[0] + float(args.smile) * smile_drt + float(args.age) * age_drt + float(args.gender) * gender_drt
                                    + float(args.beauty) * beauty_drt + float(args.angleh) * angleh_drt + float(args.anglep) * anglep_drt
                                    + float(args.raceblack) * raceblack_drt + float(args.raceyellow) * raceyellow_drt + float(args.racewhite) * racewhite_drt
                                    + float(args.glasses) * glasses_drt)[:8]
        images = Gs_network.components.synthesis.run(new_latent_vector, **Gs_syn_kwargs)
        # Save image
        PIL.Image.fromarray(images[0], 'RGB').save(dnnlib.make_run_dir_path('{}/{}_edit.png'.format(args.dst_dir, args.src_dir[:-4])))
    else:
        print ('someting wrong about edit_mod')

def transform_image(args):

    if os.path.exists(args.tmp_dir):
        shutil.rmtree(args.tmp_dir)
    os.makedirs(args.tmp_dir, exist_ok=True)
    os.makedirs(args.dst_dir, exist_ok=True)
    frame_num = 25*int(args.duration)
    scale = [x / frame_num for x in range(0, frame_num+1)]
    tflib.init_tf()
    with open(args.network_pkl, "rb") as f:
        generator_network, discriminator_network, Gs_network = pickle.load(f)

    Gs_syn_kwargs = dnnlib.EasyDict()
    Gs_syn_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_syn_kwargs.randomize_noise = False
    Gs_syn_kwargs.minibatch_size = 1

    pic1 = np.load(args.src_dir1)
    if args.transform_mod=='pic2pic':
        pic2 = np.load(args.src_dir2)
    elif args.transform_mod == 'gradual':
        latent_vector = pic1[np.newaxis, :]
        pic2 = latent_vector.copy()
        pic2[0][:8] = (latent_vector[0] + float(args.smile) * smile_drt + float(args.age) * age_drt + float(args.gender) * gender_drt
                                    + float(args.beauty) * beauty_drt + float(args.angleh) * angleh_drt + float(args.anglep) * anglep_drt
                                    + float(args.raceblack) * raceblack_drt + float(args.raceyellow) * raceyellow_drt + float(args.racewhite) * racewhite_drt
                                    + float(args.glasses) * glasses_drt)[:8]
        pic2 = pic2[0]
    else:
        print ('something wrong with transform_mod')
        return

    for ind,i in tqdm(enumerate(scale)):
        new_pic = pic2*i + pic1*(1-i)
        img = Gs_network.components.synthesis.run(new_pic[np.newaxis, :], **Gs_syn_kwargs)
        img = PIL.Image.fromarray(img[0])
        img.save('{}/{}.png'.format(args.tmp_dir, str(ind).zfill(4)))

    if args.gif:
        png2gif(args)
        if args.del_tmp:
            shutil.rmtree(args.tmp_dir)

def unpack_bz2(src_path):
    data = bz2.BZ2File(src_path).read()
    dst_path = src_path[:-4]
    with open(dst_path, 'wb') as fp:
        fp.write(data)
    return dst_path

def align(args):
    LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
    landmarks_model_path = unpack_bz2(get_file('shape_predictor_68_face_landmarks.dat.bz2',
                                               LANDMARKS_MODEL_URL, cache_subdir='temp'))
    RAW_IMAGES_DIR = args.align_dir
    ALIGNED_IMAGES_DIR = args.src_dir
    if os.path.exists(ALIGNED_IMAGES_DIR):
        shutil.rmtree(ALIGNED_IMAGES_DIR)
    os.makedirs(args.align_dir, exist_ok=True)
    landmarks_detector = LandmarksDetector(landmarks_model_path)
    for img_name in [f for f in os.listdir(RAW_IMAGES_DIR) if f[0] not in '._']:
        raw_img_path = os.path.join(RAW_IMAGES_DIR, img_name)
        for i, face_landmarks in enumerate(landmarks_detector.get_landmarks(raw_img_path), start=1):
            face_img_name = '%s_%02d.png' % (os.path.splitext(img_name)[0], i)
            aligned_face_path = os.path.join(ALIGNED_IMAGES_DIR, face_img_name)
            os.makedirs(ALIGNED_IMAGES_DIR, exist_ok=True)
            image_align(raw_img_path, aligned_face_path, face_landmarks)

def project_image(proj, src_file, dst_dir, tmp_dir):

    data_dir = '%s/dataset' % tmp_dir
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    image_dir = '%s/images' % data_dir
    tfrecord_dir = '%s/tfrecords' % data_dir
    os.makedirs(image_dir, exist_ok=True)
    shutil.copy(src_file, image_dir + '/')
    dataset_tool.create_from_images(tfrecord_dir, image_dir, shuffle=0)
    dataset_obj = dataset.load_dataset(
        data_dir=data_dir, tfrecord_dir='tfrecords',
        max_label_size=0, repeat=False, shuffle_mb=0
    )

    print('Projecting image "%s"...' % os.path.basename(src_file))
    images, _labels = dataset_obj.get_minibatch_np(1)
    images = misc.adjust_dynamic_range(images, [0, 255], [-1, 1])
    proj.start(images)

    while proj.get_cur_step() < proj.num_steps:
        print('\r%d / %d ... ' % (proj.get_cur_step(), proj.num_steps), end='', flush=True)
        proj.step()

    print('\r%-30s\r' % '', end='', flush=True)

    os.makedirs(dst_dir, exist_ok=True)
    filename = os.path.join(dst_dir, os.path.basename(src_file)[:-4] + '.png')
    misc.save_image_grid(proj.get_images(), filename, drange=[-1,1])
    filename = os.path.join(dst_dir, os.path.basename(src_file)[:-4] + '.npy')
    np.save(filename, proj.get_dlatents()[0])

def encode_image(args):
    print('Loading networks from "%s"...' % args.network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(args.network_pkl)
    proj = projector.Projector(
        vgg16_pkl=args.vgg16_pkl,
        num_steps=args.num_steps,
        initial_learning_rate=args.initial_learning_rate,
        initial_noise_factor=args.initial_noise_factor,
        verbose=args.verbose
    )
    proj.set_network(Gs)
    if args.align:
        align(args)
    src_files = sorted([os.path.join(args.src_dir, f) for f in os.listdir(args.src_dir) if f[0] not in '._'])
    for src_file in src_files:
        project_image(proj, src_file, args.dst_dir, args.tmp_dir)

def augmentation(args):

    os.makedirs(args.dst_dir+'/augmentation', exist_ok=True)

    print('Loading networks from "{}"...'.format(args.network_pkl))
    tflib.init_tf()
    with open(args.network_pkl, "rb") as f:
        generator_network, discriminator_network, Gs_network = pickle.load(f)

    Gs_syn_kwargs = dnnlib.EasyDict()
    Gs_syn_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_syn_kwargs.randomize_noise = False
    Gs_syn_kwargs.minibatch_size = 1
    pic = np.load(args.src_dir)
    for i in tqdm(range(int(args.num))):

        scale1 = random.uniform(-float(args.smile_range), float(args.smile_range))
        scale2 = random.uniform(-float(args.age_range), float(args.age_range))
        scale3 = random.uniform(-float(args.angleh_range), float(args.angleh_range))
        scale4 = random.uniform(-float(args.anglep_range), float(args.anglep_range))
        scale5 = random.uniform(-float(args.angry_range), float(args.angry_range))
        scale6 = random.uniform(-float(args.sad_range), float(args.sad_range))
        scale7 = random.uniform(-float(args.eye_range), float(args.eye_range))

        latent_vector = pic[np.newaxis, :]
        new_latent_vector = latent_vector.copy()
        new_latent_vector[0][:8] = (latent_vector[0] + scale1 * smile_drt + scale2 * age_drt
                                    + scale3 * angleh_drt + scale4 * anglep_drt
                                    + scale5 * angry_drt + scale6 * sad_drt
                                    + scale7 * eye_drt)[:8]
        img = Gs_network.components.synthesis.run(new_latent_vector, **Gs_syn_kwargs)
        img = PIL.Image.fromarray(img[0])
        img.save('{}/augmentation/{}.png'.format(args.dst_dir, str(i).zfill(4)))

def style_transform(args):
    tflib.init_tf()
    with open(args.network_pkl, 'rb') as f:
        generator_network, discriminator_network, Gs_network = pickle.load(f)
    Gs_syn_kwargs = dnnlib.EasyDict()
    Gs_syn_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_syn_kwargs.randomize_noise = False
    Gs_syn_kwargs.minibatch_size = 1

    pic1 = np.load(args.src_dir)
    pic2 = np.load(args.style_dir)
    pic1[9:] = pic2[9:]
    img = Gs_network.components.synthesis.run(pic1[np.newaxis, :], **Gs_syn_kwargs)
    img = PIL.Image.fromarray(img[0])
    img.save('{}/style_transform.png'.format(args.dst_dir))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='generate or edit pictures')


    subparsers = parser.add_subparsers()

    subparser = subparsers.add_parser("generate", help="generate pictures.")
    subparser.set_defaults(func=generate_images)
    subparser.add_argument('--truncation_psi', default=0.5, help='truncation_psi')
    subparser.add_argument('--num', default=10, help='generate images num')
    subparser.add_argument('--network_pkl', default='./networks/normal_face.pkl', help='stylegan pretrained module')
    subparser.add_argument('--dst_dir', default='output/', help='output direction')
    subparser.add_argument('--save_latent', default=True, help='is save latenent or not')
    subparser.add_argument('--save_dlatent', default=True, help='is save dlatenent or not')

    subparser = subparsers.add_parser("edit", help="edit face.")
    subparser.set_defaults(func=edit_images)
    subparser.add_argument('--edit_mod', default='mul_images', help='edit one_image or mul_images.')
    subparser.add_argument('--network_pkl', default='./networks/normal_face.pkl', help='stylegan pretrained module')
    subparser.add_argument('--dst_dir', default='output/generate_edit', help='edit images direction')
    subparser.add_argument('--src_dir', default='output/generate_wcodes', help='edit images direction')
    subparser.add_argument('--truncation_psi', default=0.5, help='truncation_psi')
    subparser.add_argument('--smile', default=0, help='change face smile')
    subparser.add_argument('--age', default=0, help='change face age')
    subparser.add_argument('--gender', default=0, help='change face gender')
    subparser.add_argument('--beauty', default=0, help='change face beauty')
    subparser.add_argument('--angleh', default=0, help='change face angleh')
    subparser.add_argument('--anglep', default=0, help='change face anglep')
    subparser.add_argument('--raceblack', default=0, help='change face raceblack')
    subparser.add_argument('--raceyellow', default=0, help='change face raceyellow')
    subparser.add_argument('--racewhite', default=0, help='change face racewhite')
    subparser.add_argument('--glasses', default=0, help='change face glasses')

    subparser = subparsers.add_parser("transform", help="generate gradual transform face")
    subparser.set_defaults(func=transform_image)
    subparser.add_argument('--transform_mod', default='gradual', help='pic1 to pic2(pic2pic) or pic1 feature gradual transform(gradual)')
    subparser.add_argument('--gif', default=False, help='generate gif or not')
    subparser.add_argument('--duration', default=5, help='duration of gif')
    subparser.add_argument('--del_tmp', default=False, help='after generate gif if delete tmp pictures')
    subparser.add_argument('--network_pkl', default='./networks/normal_face.pkl', help='stylegan pretrained module')
    subparser.add_argument('--dst_dir', default='./output', help='edit images direction')
    subparser.add_argument('--src_dir1', default='./latent_representations/sjl.npy', help='pic1 dir')
    subparser.add_argument('--src_dir2', default='./latent_representations/star.npy', help='pic2 dir(if you choose gradual mod you can pass it)')
    subparser.add_argument('--tmp_dir', default='.tmp_transform', help='tmp file')
    subparser.add_argument('--truncation_psi', default=0.5, help='truncation_psi')
    subparser.add_argument('--smile', default=0, help='change face smile')
    subparser.add_argument('--age', default=0, help='change face age')
    subparser.add_argument('--gender', default=0, help='change face gender')
    subparser.add_argument('--beauty', default=0, help='change face beauty')
    subparser.add_argument('--angleh', default=0, help='change face angleh')
    subparser.add_argument('--anglep', default=0, help='change face anglep')
    subparser.add_argument('--raceblack', default=0, help='change face raceblack')
    subparser.add_argument('--raceyellow', default=0, help='change face raceyellow')
    subparser.add_argument('--racewhite', default=0, help='change face racewhite')
    subparser.add_argument('--glasses', default=0, help='change face glasses')

    subparser = subparsers.add_parser("encode", help="encode real face")
    subparser.set_defaults(func=encode_image)
    subparser.add_argument('--align', default=True, help='find face in images and output 1024x1024 face(recommend)')
    subparser.add_argument('--align_dir', default='raw_images/', help='Directory with raw images for align')
    subparser.add_argument('--src_dir', default='aligned_images/', help='Directory with aligned images for projection')
    subparser.add_argument('--dst_dir', default='generated_images/', help='Output directory')
    subparser.add_argument('--tmp_dir', default='.stylegan2-tmp', help='Temporary directory for tfrecords and video frames')
    subparser.add_argument('--network-pkl', default='networks/normal_face.pkl', help='StyleGAN2 network pickle filename')
    subparser.add_argument('--vgg16-pkl', default='./encoder_model/vgg16_zhang_perceptual.pkl', help='VGG16 network pickle filename')
    subparser.add_argument('--num-steps', type=int, default=1000, help='Number of optimization steps')
    subparser.add_argument('--initial-learning-rate', type=float, default=0.1, help='Initial learning rate')
    subparser.add_argument('--initial-noise-factor', type=float, default=0.05, help='Initial noise factor')
    subparser.add_argument('--verbose', type=bool, default=False, help='Verbose output')

    subparser = subparsers.add_parser("augmentation", help="face augmentation")
    subparser.set_defaults(func=augmentation)
    subparser.add_argument('--network_pkl', default='./networks/normal_face.pkl', help='stylegan pretrained module')
    subparser.add_argument('--src_dir', default='./latent_representations/sjl.npy', help='Directory with encoded images for augmentation')
    subparser.add_argument('--dst_dir', default='./output', help='Output directory')
    subparser.add_argument('--num', default=10, help='generate face num')
    subparser.add_argument('--smile_range', default=5., help='range of smile change')
    subparser.add_argument('--age_range', default=1., help='range of age change')
    subparser.add_argument('--angleh_range', default=5., help='range of angleh change')
    subparser.add_argument('--anglep_range', default=5., help='range of anglep change')
    subparser.add_argument('--angry_range', default=5., help='range of angry change')
    subparser.add_argument('--sad_range', default=5., help='range of sad change')
    subparser.add_argument('--eye_range', default=5., help='range of eye-open change')

    subparser = subparsers.add_parser("style")
    subparser.set_defaults(func=style_transform)
    subparser.add_argument('--network_pkl', default='./networks/normal_face.pkl', help='stylegan pretrained module')
    subparser.add_argument('--src_dir', default='./latent_representations/sjl.npy', help='Directory with encoded images for augmentation')
    subparser.add_argument('--dst_dir', default='./output', help='Output directory')
    subparser.add_argument('--style_dir', default='./latent_representations/', help='style file dir')


    args = parser.parse_args()
    args.func(args)
    print ('done')