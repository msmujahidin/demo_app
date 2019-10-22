# import torch
# import torch.nn.functional as F
# from torchvision import transforms

from keras.models import load_model
import pickle


from boot import app



keras_model = 'model.h5'
model = load_model(keras_model)

label = open('lable.pickle', 'rb').read()
lb = pickle.loads(label)