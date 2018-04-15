import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from keras.utils import plot_model
from political_classification import PoliticalClassification
import configparser
import pydot

cf = configparser.ConfigParser()
cf.read("../file_path.properties")
path = dict(cf.items("file_path"))
dir_model = path['dir_model']
pc = PoliticalClassification('cnn_s300.h5', 'cnn_s300.npy', 18)
plot_model(pc.model, to_file='model.png', show_shapes=True, show_layer_names=True)