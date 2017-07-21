from Preprocessor import Preprocessor
from eval.SDNetTester import SDNetTester
from datasets.STL10 import STL10
from models.SDNet import SDNet

target_shape = [96, 96, 3]

for fold in range(10):
    model = SDNet(num_layers=4, batch_size=200, target_shape=target_shape, pool5=False)
    data = STL10()
    preprocessor = Preprocessor(target_shape=target_shape)
    tester = SDNetTester(model, data, preprocessor, tag='baseline')
    tester.test_classifier_cv(num_conv_trained=5, fold=fold)
