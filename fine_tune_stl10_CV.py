from Preprocessor import Preprocessor
from train.SDNetTrainer import SDNetTrainer
from datasets.STL10 import STL10
from models.SDNet import SDNet
from utils import get_checkpoint_path

target_shape = [96, 96, 3]

for fold in range(10):
    model = SDNet(num_layers=4, batch_size=200, target_shape=target_shape, pool5=False)
    data = STL10()
    preprocessor = Preprocessor(target_shape=target_shape)
    trainer = SDNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=400, tag='baseline',
                           lr_policy='linear', optimizer='adam')
    chpt_path = get_checkpoint_path(trainer.get_save_dir())
    trainer.finetune_cv(chpt_path, num_conv2train=5, num_conv2init=5, fold=fold)
