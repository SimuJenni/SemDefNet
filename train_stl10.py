from Preprocessor import Preprocessor
from train.SDNetTrainer_linear_decay import SDNetTrainer
from datasets.STL10 import STL10
from models.SDNet_avgDisc import SDNet

target_shape = [64, 64, 3]
model = SDNet(num_layers=4, num_res=5, batch_size=200, target_shape=target_shape, pool5=False, disc_pad='SAME', tag='elu')
data = STL10()
preprocessor = Preprocessor(target_shape=target_shape, augment_color=True)
trainer = SDNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=400, tag='linear_decay')
trainer.train()
