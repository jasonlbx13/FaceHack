import pickle
import PIL.Image
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import os
import time
import streamlit as st

st.title('欢迎使用海马云人脸图像生成平台')
st.header('Face Hack Web App')

npy_dir = './latent_representations/' + st.sidebar.selectbox('请选择要变化的图片:', os.listdir('./latent_representations/'), index=7)
model_dir = './networks/'+st.sidebar.selectbox('请选择生成模型:', os.listdir('./networks/'), index=2)
save = st.sidebar.checkbox('生成随机人脸的同时保存它的掩码')

smile = st.sidebar.slider('控制微笑程度:', -30.0, 30.0, 0.0, 0.1)
age = st.sidebar.slider('控制衰老程度:', -30.0, 30.0, 0.0, 0.1)
gender = st.sidebar.slider('性别倾向:', -30.0, 30.0, 0.0, 0.1)
beauty = st.sidebar.slider('颜值:', -30.0, 30.0, 0.0, 0.1)
angleh = st.sidebar.slider('偏转方向:', -30.0, 30.0, 0.0, 0.1)
anglep = st.sidebar.slider('俯仰方向:', -30.0, 30.0, 0.0, 0.1)
raceblack = st.sidebar.slider('黑色人种倾向:', -10.0, 10.0, 0.0, 0.1)
raceyellow = st.sidebar.slider('黄色人种倾向:', -30.0, 30.0, 0.0, 0.1)
racewhite = st.sidebar.slider('白色人种倾向:', -30.0, 30.0, 0.0, 0.1)
glasses = st.sidebar.slider('戴眼镜倾向:', -30.0, 30.0, 0.0, 0.1)

randome_generate = st.button('随机生成一张人脸')
run = st.button('开始生成')

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
@st.cache(suppress_st_warning=True)
def move_latent(latent_vector, Gs_network, Gs_syn_kwargs):

    new_latent_vector = latent_vector.copy()
    new_latent_vector[0][:8] = (latent_vector[0]+smile*smile_drt+age*age_drt+gender*gender_drt
                                +beauty*beauty_drt+angleh*angleh_drt+anglep*anglep_drt
                                +raceblack*raceblack_drt+raceyellow*raceyellow_drt+racewhite*racewhite_drt
                                +glasses*glasses_drt)[:8]
    images = Gs_network.components.synthesis.run(new_latent_vector, **Gs_syn_kwargs)
    result = PIL.Image.fromarray(images[0], 'RGB')
    return result


@st.cache(suppress_st_warning=True)
def demo():

    tflib.init_tf()
    with open(model_dir, "rb") as f:
        generator_network, discriminator_network, Gs_network = pickle.load(f)

    Gs_syn_kwargs = dnnlib.EasyDict()
    Gs_syn_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_syn_kwargs.randomize_noise = False
    Gs_syn_kwargs.minibatch_size = 1
    nw = np.load(npy_dir)[np.newaxis, :]
    image = move_latent(nw, Gs_network, Gs_syn_kwargs)
    return image


@st.cache(suppress_st_warning=True)
def random_generate_demo():

    tflib.init_tf()
    with open(model_dir, "rb") as f:
        generator_network, discriminator_network, Gs_network = pickle.load(f)

    w_avg = Gs_network.get_var('dlatent_avg')
    noise_vars = [var for name, var in Gs_network.components.synthesis.vars.items() if name.startswith('noise')]
    Gs_syn_kwargs = dnnlib.EasyDict()
    Gs_syn_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_syn_kwargs.randomize_noise = False
    Gs_syn_kwargs.minibatch_size = 1
    truncation_psi = 0.5

    z = np.random.randn(1, *Gs_network.input_shape[1:])
    tflib.set_vars({var: np.random.randn(*var.shape.as_list()) for var in noise_vars})  # [height, width]
    w = Gs_network.components.mapping.run(z, None)
    w = w_avg + (w - w_avg) * truncation_psi

    image = move_latent(w, Gs_network, Gs_syn_kwargs)
    if save:
        np.save('./latent_representations/{}'.format(time.ctime()),w[0])
    return image



if __name__ == '__main__':

    if randome_generate:
        image = random_generate_demo()
        st.image(image, caption='生成结果', use_column_width=True)
    if not run:
        pass
    else:
        image = demo()
        st.image(image, caption='生成结果', use_column_width=True)