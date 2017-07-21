from Preprocessor import VOCPreprocessor
from eval.SDNetTester import SDNetTester
from datasets.VOC2007 import VOC2007
from models.SDNet import SDNet

im_shape = [227, 227, 3]
model = SDNet(num_layers=5, batch_size=1, fix_bn=False, target_shape=im_shape)
data = VOC2007()
preprocessor = VOCPreprocessor(target_shape=im_shape, augment_color=False, area_range=(0.1, 1.0))
tester = SDNetTester(model, data, preprocessor, tag='baseline')
tester.test_classifier_voc(num_conv_trained=5)
