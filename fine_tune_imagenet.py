from Preprocessor import Preprocessor
from train.SDNetTrainer_linear_decay import SDNetTrainer
from datasets.ImageNet import ImageNet
from models.SDNet_avgDisc_concat_new_spatial_drop_noise import SDNet
from utils import get_checkpoint_path

target_shape = [227, 227, 3]
model = SDNet(num_layers=5, batch_size=256, target_shape=target_shape)
data = ImageNet()
preprocessor = Preprocessor(target_shape=target_shape)
trainer = SDNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=100, tag='baseline_lr001',
                       lr_policy='linear', optimizer='adam', init_lr=0.001)
chpt_path = get_checkpoint_path(trainer.get_save_dir())
trainer.transfer_finetune(chpt_path, num_conv2train=5, num_conv2init=0)
