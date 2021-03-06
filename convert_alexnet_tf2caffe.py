import os

import caffe
import numpy as np
import skimage
import tensorflow as tf

from Preprocessor import Preprocessor
from datasets.ImageNet import ImageNet
from models.AlexNetConverter import AlexNetConverter
from models.SDNet import SDNet
from train.SDNetTrainer import SDNetTrainer

im_s = 227


def preprocess(img):
    out = np.copy(img)
    out = out[:, :, [2, 1, 0]]  # swap channel from RGB to BGR
    out = out.transpose((2, 0, 1))  # h, w, c -> c, h, w
    return out


def load_image(path):
    # load image
    img = np.float32(skimage.io.imread(path))
    img /= 127.5
    img -= 1.0
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (im_s, im_s))
    return resized_img


model = SDNet(num_layers=5, target_shape=[im_s, im_s, 3], batch_size=16, disc_pad='VALID')
data = ImageNet()
preprocessor = Preprocessor(target_shape=[im_s, im_s, 3])
trainer = SDNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=80, tag='refactored',
                       lr_policy='const', optimizer='adam')

model_dir = '../test_converter'
proto_path = 'deploy.prototxt'
ckpt = '../test_converter/model.ckpt-800722'
save_path = os.path.join(model_dir, 'alexnet_v2.caffemodel')

np.random.seed(42)
img = load_image('cat.jpg')

converter = AlexNetConverter(model_dir, model, trainer.sess, ckpt=ckpt, remove_bn=True, scale=1.0, bgr=True,
                             im_size=(im_s, im_s), with_fc=False, use_classifier=False)
with converter.sess:
    converter.extract_and_store()
    net, encoded = model.discriminator.discriminate(tf.constant(img, shape=[1, im_s, im_s, 3], dtype=tf.float32),
                                                    with_fc=converter.with_fc, reuse=True, training=False)
    result_tf = encoded.eval()

converter.load_and_set_caffe_weights(proto_path=proto_path, save_path=save_path)

# Testing
net_caffe = caffe.Net(proto_path, save_path, caffe.TEST)
net_caffe.blobs['data'].data[0] = preprocess(img)
assert net_caffe.blobs['data'].data[0].shape == (3, im_s, im_s)
net_caffe.forward()
result_caffe = net_caffe.blobs['Convolution5'].data[0]
result_caffe = result_caffe.transpose((1, 2, 0))  # h, w, c -> c, h, w
print(np.linalg.norm(result_tf - result_caffe))
