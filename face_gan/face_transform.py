from PIL import Image
import numpy as np
import argparse
import imageio
import dnnlib
import dnnlib.tflib as tflib
import os
import pickle
from tqdm import tqdm

parser = argparse.ArgumentParser(description='transform pic1 to pic2')
parser.add_argument('pic1', default='1.npy', help='pic1 directory')
parser.add_argument('pic2', default='2.npy', help='pic2 directory')
args = parser.parse_args()

pic1_dir = './latent_representations/'+args.pic1
pic2_dir = './latent_representations/'+args.pic2

pic1 = np.load(pic1_dir)
pic2 = np.load(pic2_dir)

def png2gif(file_name):
    images = []
    filenames=sorted((fn for fn in os.listdir(file_name) if fn.endswith('.png')))
    for filename in filenames:
        images.append(imageio.imread(file_name+'/'+filename))
    imageio.mimsave('result.gif', images, duration=0.04)


def main():
    scale = [x/100 for x in range(0,101)]
    tflib.init_tf()
    with open('./networks/normal_face.pkl', "rb") as f:
        generator_network, discriminator_network, Gs_network = pickle.load(f)

    Gs_syn_kwargs = dnnlib.EasyDict()
    Gs_syn_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_syn_kwargs.randomize_noise = False
    Gs_syn_kwargs.minibatch_size = 1


    for ind,i in tqdm(enumerate(scale)):
        new_pic = pic2*i + pic1*(1-i)
        img = Gs_network.components.synthesis.run(new_pic[np.newaxis, :], **Gs_syn_kwargs)
        img = Image.fromarray(img[0])
        img.save('./transform_data/%05d.png' % ind)

if __name__ == '__main__':

    main()
    png2gif('./transform_data')
