from Preprocessor import Preprocessor
from train.SDNetTrainer import SDNetTrainer
from datasets.ImageNet import ImageNet
from models.SDNet_avgDisc_concat_new_2 import SDNet

target_shape = [128, 128, 3]
model = SDNet(num_layers=5, batch_size=128, target_shape=target_shape)
data = ImageNet()
preprocessor = Preprocessor(target_shape=target_shape, augment_color=True)
trainer = SDNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=180, tag='baseline',
                       lr_policy='linear', optimizer='adam', init_lr=0.0003, end_lr=1e-8)
trainer.train()
