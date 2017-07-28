from Preprocessor import Preprocessor
from eval.SDNetTester import SDNetTester
from datasets.STL10 import STL10
from models.SDNet_drop_and_pool import SDNet

target_shape = [96, 96, 3]

model = SDNet(num_layers=4, batch_size=200, target_shape=target_shape, pool5=False, disc_pad='SAME', tag='elu')
data = STL10()
preprocessor = Preprocessor(target_shape=target_shape)
tester = SDNetTester(model, data, preprocessor)
tester.test_classifier(num_conv_trained=5)