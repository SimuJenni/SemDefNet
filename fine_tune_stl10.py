from Preprocessor import Preprocessor
from train.SDNetTrainer import SDNetTrainer
from datasets.STL10 import STL10
from models.SDNet import SDNet
from utils import get_checkpoint_path

model = SDNet(num_layers=4, batch_size=200, target_shape=[64, 64, 3], pool5=False)
data = STL10()
preprocessor = Preprocessor(target_shape=[64, 64, 3], im_shape=[96, 96, 3], augment_color=True)
trainer = SDNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=400, tag='refactored',
                       lr_policy='linear', optimizer='adam')
ckpt_path = get_checkpoint_path(trainer.get_save_dir())
trainer.transfer_finetune(ckpt_path, num_conv2train=5, num_conv2init=5)
