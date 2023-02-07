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


from model.model_architecture import UNET
from data_manager.data_handler import get_dataloaders

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

def save_losses(train_loss:list, val_loss:list, test_loss:list)->None:
    test_loss= [loss.cpu().detach().numpy() for loss in test_loss]
    val_loss= [loss.cpu().detach().numpy() for loss in val_loss]
    train_loss= [loss.cpu().detach().numpy() for loss in train_loss]
    
    plt.figure(figsize=(10,5))
    plt.title("Training and Validation Loss")
    plt.plot(val_loss, label="val")
    plt.plot(train_loss,label="train")
    plt.plot(test_loss,label="test")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('train_losses.png')

def train(model, train_dl, valid_dl, test_dl, loss_fn, optimizer, acc_fn, epochs=1):
    start = time.time()
    model.cuda()

    train_loss, valid_loss, test_loss = [], [], []
    early_stopper = EarlyStopper(patience = 5, min_delta= 0.1)


    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        for phase in ['train', 'valid', 'test']:
            if phase == 'train':
                #model.train(True)  # Set trainind mode = true
                dataloader = train_dl
            elif phase == 'valid':
                #model.train(False)  # Set model to evaluate mode
                dataloader = valid_dl
            else:
                dataloader = test_dl
                
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
            #print(f"AllocMem (Mb) {torch.cuda.memory_allocated()/1024/1024}")
            if phase == 'train':
                train_loss.append(epoch_loss) 
            elif phase=='valid': 
                valid_loss.append(epoch_loss)
            elif phase=='test':
                test_loss.append(epoch_loss)

        if phase == 'valid' and early_stopper.early_stop(epoch_loss):
            break            
    
    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))    
    
    save_losses(train_loss, valid_loss, test_loss)
    return train_loss, valid_loss, test_loss    

def acc_metric(predb, yb):
    #return (predb.argmax(dim=1) == yb.cuda()).float().mean()
    return ((torch.sigmoid(predb) > 0.5) == yb.cuda()).float().mean()


def run_experiment(lr = 0.01, epochs = 3):
    image_patches_dir = 'data/images_patches'
    masks_patches_dir = 'data/masks_patches'
    
    assert len(os.listdir(image_patches_dir)) == len(os.listdir(masks_patches_dir)), 'mistmatch'
    
    image_patches = np.array(sorted([os.path.join(image_patches_dir, x) for x in os.listdir(image_patches_dir)]))
    mask_patches = np.array(sorted([os.path.join(masks_patches_dir, x) for x in os.listdir(masks_patches_dir)]))
    assert len(image_patches) == len(mask_patches)
    
    
    ### SPLIT ###
    total_idxs = np.arange(len(image_patches))
    train_idxs, val_idxs = train_test_split(total_idxs, train_size = 0.7)
    val_idxs, test_idxs = train_test_split(val_idxs, train_size = 0.8)

    train_images = image_patches[train_idxs]
    val_images = image_patches[val_idxs]
    test_images = image_patches[test_idxs]

    train_masks = mask_patches[train_idxs]
    val_masks = mask_patches[val_idxs]
    test_masks = mask_patches[test_idxs]
            
    train_dl, val_dl, test_dl = get_dataloaders(train_images, train_masks, val_images, val_masks, test_images, test_masks)
    
    unet = UNET(4,1) 
    
    #check shape on one sample
    sample = next(iter(train_dl))
    xb, yb = sample['image'], sample['mask']
    pred = unet(xb)
    print('data shape : ' , xb.shape, yb.shape)
    print('prediction shape: ', pred.shape)
    
    #hyperparams 
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(unet.parameters(), lr = lr)
    train_loss, val_loss, test_loss = train(unet, train_dl, val_dl, test_dl, loss_fn, opt, acc_metric, epochs = epochs)    
    torch.save(unet.state_dict(), 'unet.pth')
    
    
    

if __name__ =='__main__':
    run_experiment()
    