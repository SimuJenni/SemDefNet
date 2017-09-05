from Preprocessor import Preprocessor
from train.SDNetTrainer_linear_decay import SDNetTrainer
from datasets.ImageNet import ImageNet
from models.SDNet_avgDisc_concat_new_3 import SDNet

target_shape = [160, 160, 3]
model = SDNet(num_layers=5, batch_size=64, target_shape=target_shape, pool5=True, disc_pad='SAME', tag='new_default')
data = ImageNet()
preprocessor = Preprocessor(target_shape=target_shape, augment_color=True)
trainer = SDNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=180, tag='new_default',
                       lr_policy='linear', optimizer='adam', init_lr=0.0003, end_lr=0.000003)
trainer.train()
