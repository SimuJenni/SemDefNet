from Preprocessor import Preprocessor
from eval.SDNetTester import SDNetTester
from datasets.ImageNet import ImageNet
from models.SDNet import SDNet

target_shape = [128, 128, 3]
model = SDNet(num_layers=4, batch_size=128, target_shape=target_shape)
data = ImageNet()
preprocessor = Preprocessor(target_shape=target_shape, augment_color=False)
tester = SDNetTester(model=model, dataset=data, pre_processor=preprocessor, tag='baseline')
tester.test_reconstruction()
