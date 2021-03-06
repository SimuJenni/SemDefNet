from Preprocessor import Preprocessor
from eval.SDNetTester import SDNetTester
from datasets.ImageNet import ImageNet
from models.SDNet_avgDisc_concat_new_spatial_drop_noise_2 import SDNet

target_shape = [227, 227, 3]
model = SDNet(num_layers=5, batch_size=128, target_shape=target_shape)
data = ImageNet()
preprocessor = Preprocessor(target_shape=target_shape)
tester = SDNetTester(model, data, preprocessor, tag='baseline_lr001')
tester.test_classifier(num_conv_trained=5)
