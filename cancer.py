#%%
# Imports
import os
from torch import torch, nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
# import importlib
from tqdm import tqdm
from imageio import imread, imsave, imwrite
#get the other funcions
import dataset
import model
import trainer
#%%
from dataset import Cancer
from trainer import Trainer
from model import get_model
from torch.autograd import Variable
from visuals import plot_losses

import matplotlib.pyplot as plt
#%%
# CULaneDataset = importlib.reload(dataset).CULaneDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%%
data_dir = './data'
labels_dir = './data/train_labels.csv'

# WhaleDataset = Whales('train', data_dir, labels_dir, aug=False, preprocess=True, ToTensor=True)
# WhaleDataset_val = Whales('val', data_dir, labels_dir, aug=False, preprocess=True, ToTensor=True)

#images are (96,96,3)
CancerDataset = Cancer('train', data_dir, labels_dir, aug=False, preprocess=True, ToTensor=True)
CancerDataset_val = Cancer('val', data_dir, labels_dir, aug=False, preprocess=True, ToTensor=True)

BATCH_SIZE = 20
NUM_WORKERS = 6

# WhaleDataLoader = DataLoader(WhaleDataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
# WhaleDataLoader_val = DataLoader(WhaleDataset_val, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)

CancerDataLoader = DataLoader(CancerDataset, 
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=NUM_WORKERS,
                              drop_last=True)

CancerDataLoader_val = DataLoader(CancerDataset_val,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=NUM_WORKERS,
                                  drop_last=True,
                                  sampler=None)

# save_model_path = './test.pkl'
save_model_path = './test.pt'

Cancer_model = get_model(device= device,
                path = save_model_path)

optim = torch.optim.Adam(Cancer_model.parameters(), lr=1e-3)
loss_fcn = nn.CrossEntropyLoss()

NUM_EPOCHS = 10
losses = []
trainer = Trainer(device=device, 
                  model=Cancer_model, 
                  train_loader=CancerDataLoader, 
                  val_loader=CancerDataLoader_val, 
                  optimizer=optim, 
                  loss_fcn=loss_fcn)

big_train_loss_list = []
big_val_loss_list = []

#loading a model from saved state dictionary
# loaded_model = get_model(device=device)
# loaded_model.load_state_dict(torch.load(save_model_path))
# loaded_model.eval()

#to plot the losses
plot = True

for ep in tqdm(range(NUM_EPOCHS), desc='Epochs'):
    #train one epoch    
    train_loss_list = trainer.train_epoch(save_model=True)
    val_loss_list = trainer.validate(sample_size=20)

    big_train_loss_list += train_loss_list
    big_val_loss_list += val_loss_list
    
    if plot:
        plot_losses(train_loss_list=big_train_loss_list,
                    val_loss_list=big_val_loss_list)
