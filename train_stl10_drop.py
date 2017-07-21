from Preprocessor import Preprocessor
from train.SDNetTrainer import SDNetTrainer
from datasets.STL10 import STL10
from models.SDNet_drop_pred import SDNet

target_shape = [64, 64, 3]
model = SDNet(num_layers=4, batch_size=200, target_shape=target_shape, pool5=False, disc_pad='SAME')
data = STL10()
preprocessor = Preprocessor(target_shape=target_shape, augment_color=True)
trainer = SDNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=300, tag='drop',
                       lr_policy='linear', optimizer='adam', init_lr=0.0003)
trainer.train()
