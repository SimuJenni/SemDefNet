from Preprocessor import Preprocessor
from train.SDNetTrainer import SDNetTrainer
from datasets.STL10 import STL10
from models.SDNet_avgDisc_concat_new_4 import SDNet
from utils import get_checkpoint_path

target_shape = [96, 96, 3]
model = SDNet(num_layers=4, num_res=1, batch_size=200, target_shape=target_shape, pool5=False, disc_pad='SAME', tag='lin_decay_larger')
data = STL10()
preprocessor = Preprocessor(target_shape=target_shape, augment_color=True)
trainer = SDNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=400,
                       lr_policy='linear', optimizer='adam')
ckpt_path = get_checkpoint_path(trainer.get_save_dir())
trainer.transfer_finetune(ckpt_path, num_conv2train=5, num_conv2init=5)
