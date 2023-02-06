import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from torch import nn
import torch 
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import cv2
from torch.utils.data import Dataset
import torch
import time
import wandb
from datetime import datetime


from model_architecture import UNET
from data_handler import get_dataloaders

### EARLY STOPPING ###
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                print('\n Early Stopper : Stopping Train')
                return True
        return False

   
def train(model, train_dl, valid_dl, loss_fn, optimizer, acc_fn, epochs=1):
    start = time.time()
    model.cuda()

    train_loss, valid_loss = [], []
    early_stopper = EarlyStopper(patience = 5, min_delta= 0)


    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                #model.train(True)  # Set trainind mode = true
                dataloader = train_dl
            else:
                #model.train(False)  # Set model to evaluate mode
                dataloader = valid_dl

            running_loss = 0.0
            running_acc = 0.0

            step = 0

            # iterate over data
            for i, sample in tqdm(enumerate(dataloader), total = len(dataloader)):
                x = sample['image']
                x = (x - x.mean())/x.std() #normalize
                y = sample['mask']
                x = x.cuda()
                y = y.cuda()
                step += 1

                # forward pass
                if phase == 'train':
                    # zero the gradients
                    optimizer.zero_grad()
                    outputs = model(x)
                    loss = loss_fn(outputs, y.float())

                    # the backward pass frees the graph memory, so there is no 
                    # need for torch.no_grad in this training pass
                    loss.backward()
                    optimizer.step()
                    # scheduler.step()

                else:
                    with torch.no_grad():
                        outputs = model(x)
                        loss = loss_fn(outputs, y.float())

                # stats - whatever is the phase
                acc = acc_fn(outputs, y)

                running_acc  += acc*dataloader.batch_size
                running_loss += loss*dataloader.batch_size 

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_acc / len(dataloader.dataset)

            print('{} Loss: {:.4f} Acc: {}'.format(phase, epoch_loss, epoch_acc))
            print(f"AllocMem (Mb) {torch.cuda.memory_allocated()/1024/1024}")

            train_loss.append(epoch_loss) if phase=='train' else valid_loss.append(epoch_loss)

        if phase == 'valid' and early_stopper.early_stop(epoch_loss):
            break            
    
    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))    
    
    return train_loss, valid_loss    

def acc_metric(predb, yb):
    #return (predb.argmax(dim=1) == yb.cuda()).float().mean()
    return ((torch.sigmoid(predb) > 0.5) == yb.cuda()).float().mean()


if __name__ =='__main__':
    image_patches_dir = 'data/images_patches'
    masks_patches_dir = 'data/masks_patches'
    
    assert len(os.listdir(image_patches_dir)) == len(os.listdir(masks_patches_dir)), 'mistmatch'
    print(len(os.listdir(image_patches_dir)), len(os.listdir(masks_patches_dir)))   
    
    image_patches = np.array(sorted([os.path.join(image_patches_dir, x) for x in os.listdir(image_patches_dir)]))
    mask_patches = np.array(sorted([os.path.join(masks_patches_dir, x) for x in os.listdir(masks_patches_dir)]))
    assert len(image_patches) == len(mask_patches)
    
    
    ### SPLIT ###
    total_idxs = np.arange(len(image_patches))
    train_idxs, val_idxs = train_test_split(total_idxs, train_size = 0.7)
    val_idxs, test_idxs = train_test_split(val_idxs, train_size = 0.8)

    train_images = image_patches[train_idxs][:10]
    val_images = image_patches[val_idxs][:10]
    test_images = image_patches[test_idxs][:10]

    train_masks = mask_patches[train_idxs][:10]
    val_masks = mask_patches[val_idxs][:10]
    test_masks = mask_patches[test_idxs][:10]
            
    train_dl, val_dl, test_dl = get_dataloaders(train_images, train_masks, val_images, val_masks, test_images, test_masks)
    
    unet = UNET(4,1) 
    
    #check shape on one sample
    sample = next(iter(train_dl))
    xb, yb = sample['image'], sample['mask']
    print(xb.shape, yb.shape)
    
    pred = unet(xb)
    print(pred.shape)
    
    #hyperparams 
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(unet.parameters(), lr = 0.01)
    EPOCHS = 1
    train_loss, val_loss = train(unet, train_dl, val_dl, loss_fn, opt, acc_metric, epochs = EPOCHS)
    
    torch.save(unet.state_dict(), 'unet2.pth')
    
    