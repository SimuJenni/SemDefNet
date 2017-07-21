from Preprocessor import VOCPreprocessor
from train.SDNetTrainer import SDNetTrainer
from datasets.ImageNet import ImageNet
from datasets.VOC2007 import VOC2007
from models.SDNet import SDNet
from utils import get_checkpoint_path

im_shape = [227, 227, 3]
model = SDNet(num_layers=5, batch_size=16, target_shape=im_shape, fix_bn=False)
data = ImageNet()
preprocessor = VOCPreprocessor(target_shape=im_shape, augment_color=True, area_range=(0.1, 1.0))
trainer = SDNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=250, tag='baseline',
                       lr_policy='linear', optimizer='adam')
chpt_path = get_checkpoint_path(trainer.get_save_dir())

trainer.dataset = VOC2007()
trainer.transfer_finetune(chpt_path, num_conv2train=5, num_conv2init=5)
