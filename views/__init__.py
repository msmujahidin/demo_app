from PIL import Image
from flask import Blueprint, render_template, request, jsonify

import numpy as np  

import tensorflow as tf
from util import model, lb

graph = tf.get_default_graph()
base = Blueprint('base', __name__)
THRESHOLD = 1.5

TF_EnableXLACompilation(sess_opts,true);


@base.route('/')
def index():
    return render_template('index.html')


@base.route('/predict', methods=['post'])
def predict():
    files = request.files
    img_left = Image.open(files.get('imgLeft'))
    img_cnn = img_left.resize((256, 256))
    img_cnn = np.array(img_cnn)
    # tambahkan dimensi gamb`arnnya
    Image.fromarray(img_cnn).save('tes.jpg')
    img_cnn = np.expand_dims(img_cnn, axis = 0)
    print(img_cnn.shape)
    print(model.summary())


    # melakukan prediksi


    # proses kembali sehingga didapatkan label kelas keluaran hasil prediksinya
    with graph.as_default():
        pred = model.predict(img_cnn)
    i = pred.argmax(axis = 1)[0]
    label_class = lb.classes_[i]
    # label_class = 'ngarang'
    # pred = np.array([1,2])
    return jsonify(klasifikasi=label_class,
                   score=pred.max().item()*100)
