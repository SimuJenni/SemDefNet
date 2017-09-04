from Preprocessor import Preprocessor
from train.SDNetTrainer_linear_decay import SDNetTrainer
from datasets.STL10 import STL10
from models.SDNet_avgDisc_concat_new_3 import SDNet

target_shape = [96, 96, 3]
model = SDNet(num_layers=4, num_res=1, batch_size=128, target_shape=target_shape, pool5=False, disc_pad='SAME', tag='new_default')
data = STL10()
preprocessor = Preprocessor(target_shape=target_shape, augment_color=True)
trainer = SDNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=500, init_lr=0.0003,
                       end_lr=0.000003)
trainer.train()
