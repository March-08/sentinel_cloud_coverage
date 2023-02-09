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
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
from data_manager.data_handler import SegmentationDataset

from model.model_architecture import UNET
from data_manager.data_handler import get_dataloaders, get_train_transforms

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

def save_metrics(file_out:str, train_metric:list, val_metric:list, metric:str)->None:
    assert len(train_metric) == len(val_metric), 'len metrics dont match!'
    plt.figure(figsize=(10,5))
    plt.title("Training and Validation")
    plt.plot(val_metric, label="val")
    plt.plot(train_metric,label="train")
    plt.xlabel("iterations")
    plt.ylabel(metric)
    plt.legend()
    plt.savefig(file_out)

def train(model, train_dl, valid_dl, loss_fn, optimizer, acc_fn, epochs=1):
    start = time.time()
    model.cuda()

    train_loss, valid_loss= [], []
    train_acc , valid_acc =[], []
    early_stopper = EarlyStopper(patience = 5, min_delta= 0.1)


    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                #model.train(True)  # Set trainind mode = true
                dataloader = train_dl
            elif phase == 'valid':
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
            #print(f"AllocMem (Mb) {torch.cuda.memory_allocated()/1024/1024}")
            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc) 
            elif phase=='valid': 
                valid_loss.append(epoch_loss)
                valid_acc.append(epoch_acc)
        if phase == 'valid' and early_stopper.early_stop(epoch_loss):
            break            
    
    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))    
    return list_to_memory(train_loss), list_to_memory(train_acc), list_to_memory(valid_loss), list_to_memory(valid_acc)   

def acc_metric(predb, yb):
    #return (predb.argmax(dim=1) == yb.cuda()).float().mean()
    return ((torch.sigmoid(predb) > 0.5) == yb.cuda()).float().mean()

def list_to_memory(l:list):
    return [x.detach().cpu().numpy() for x in l]
    
def compute_average(l:list):
    return np.average(np.array(l), axis = 0)

def run_experiment(kfolds:int = 2, lr:float = 0.01, epochs:int = 10, batch_size:int = 64):
    image_patches_dir = 'data/images_patches'
    masks_patches_dir = 'data/masks_patches'
    
    assert len(os.listdir(image_patches_dir)) == len(os.listdir(masks_patches_dir)), 'mistmatch'
    
    image_patches = np.array(sorted([os.path.join(image_patches_dir, x) for x in os.listdir(image_patches_dir)]))
    mask_patches = np.array(sorted([os.path.join(masks_patches_dir, x) for x in os.listdir(masks_patches_dir)]))
    assert len(image_patches) == len(mask_patches)
    
    
    ### SPLIT ###
    total_idxs = np.arange(len(image_patches))
    images = image_patches[total_idxs]
    masks = mask_patches[total_idxs]
    transform = get_train_transforms()
    dataset = SegmentationDataset(images, masks, transforms= None)
    transformed_dataset = SegmentationDataset(images, masks, transforms= transform)
    splits=KFold(n_splits= kfolds, shuffle=True,random_state=42)

    avg_train_loss , avg_train_acc = [], []
    avg_valid_loss, avg_valid_acc = [], []

    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(total_idxs)))):
        print(f"FOLD : {fold}\n")
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(dataset, batch_size= batch_size, sampler=train_sampler)
        val_loader = DataLoader(transformed_dataset, batch_size= batch_size, sampler=val_sampler)

        unet = UNET(4,1)
        loss_fn = nn.BCEWithLogitsLoss()
        opt = torch.optim.Adam(unet.parameters(), lr = lr)
        train_loss, train_acc, valid_loss, valid_acc = train(model= unet, train_dl= train_loader, \
            valid_dl=val_loader, loss_fn= loss_fn, optimizer= opt, acc_fn= acc_metric, epochs= epochs)
        avg_train_loss.append(train_loss)
        avg_train_acc.append(train_acc)
        avg_valid_loss.append(valid_loss) 
        avg_valid_acc.append(valid_acc)
    
    avg_train_loss= compute_average(avg_train_loss)
    avg_train_acc = compute_average(avg_train_acc)
    avg_valid_loss = compute_average(avg_valid_loss)
    avg_valid_acc =compute_average(avg_valid_acc)
    
    print('avg train loss: ', avg_train_loss)
    print('avg train acc: ', avg_train_acc)
    print('avg valid loss: ', avg_valid_loss)
    print('avg valid acc: ', avg_valid_acc)
    
    save_metrics('outputs/losses.png', avg_train_loss, avg_valid_loss, 'loss')
    save_metrics('outputs/accuracies.png', avg_train_acc, avg_valid_acc, 'accuracy')
    
    torch.save(unet.state_dict(), 'outputs/unet.pth')


if __name__ =='__main__':
    run_experiment()
    







# import pandas as pd
# import numpy as np
# import os
# import matplotlib.pyplot as plt
# from torch import nn
# import torch 
# from sklearn.model_selection import train_test_split
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from tqdm import tqdm
# import cv2
# from torch.utils.data import Dataset
# import torch
# import time
# from sklearn.model_selection import KFold


# from model.model_architecture import UNET
# from data_manager.data_handler import get_dataloaders

# ### EARLY STOPPING ###
# class EarlyStopper:
#     def __init__(self, patience=1, min_delta=0):
#         self.patience = patience
#         self.min_delta = min_delta
#         self.counter = 0
#         self.min_validation_loss = np.inf

#     def early_stop(self, validation_loss):
#         if validation_loss < self.min_validation_loss:
#             self.min_validation_loss = validation_loss
#             self.counter = 0
#         elif validation_loss > (self.min_validation_loss + self.min_delta):
#             self.counter += 1
#             if self.counter >= self.patience:
#                 print('\n Early Stopper : Stopping Train')
#                 return True
#         return False

# def save_losses(train_loss:list, val_loss:list, test_loss:list)->None:
#     test_loss= [loss.cpu().detach().numpy() for loss in test_loss]
#     val_loss= [loss.cpu().detach().numpy() for loss in val_loss]
#     train_loss= [loss.cpu().detach().numpy() for loss in train_loss]
    
#     plt.figure(figsize=(10,5))
#     plt.title("Training and Validation Loss")
#     plt.plot(val_loss, label="val")
#     plt.plot(train_loss,label="train")
#     plt.plot(test_loss,label="test")
#     plt.xlabel("iterations")
#     plt.ylabel("Loss")
#     plt.legend()
#     plt.savefig('train_losses.png')

# def train(model, train_dl, valid_dl, test_dl, loss_fn, optimizer, acc_fn, epochs=1):
#     start = time.time()
#     model.cuda()

#     train_loss, valid_loss, test_loss = [], [], []
#     early_stopper = EarlyStopper(patience = 5, min_delta= 0.1)


#     for epoch in range(epochs):
#         print('Epoch {}/{}'.format(epoch, epochs - 1))
#         print('-' * 10)

#         for phase in ['train', 'valid', 'test']:
#             if phase == 'train':
#                 #model.train(True)  # Set trainind mode = true
#                 dataloader = train_dl
#             elif phase == 'valid':
#                 #model.train(False)  # Set model to evaluate mode
#                 dataloader = valid_dl
#             else:
#                 dataloader = test_dl
                
#             running_loss = 0.0
#             running_acc = 0.0

#             step = 0

#             # iterate over data
#             for i, sample in tqdm(enumerate(dataloader), total = len(dataloader)):
#                 x = sample['image']
#                 x = (x - x.mean())/x.std() #normalize
#                 y = sample['mask']
#                 x = x.cuda()
#                 y = y.cuda()
#                 step += 1

#                 # forward pass
#                 if phase == 'train':
#                     # zero the gradients
#                     optimizer.zero_grad()
#                     outputs = model(x)
#                     loss = loss_fn(outputs, y.float())

#                     # the backward pass frees the graph memory, so there is no 
#                     # need for torch.no_grad in this training pass
#                     loss.backward()
#                     optimizer.step()
#                     # scheduler.step()

#                 else:
#                     with torch.no_grad():
#                         outputs = model(x)
#                         loss = loss_fn(outputs, y.float())

#                 # stats - whatever is the phase
#                 acc = acc_fn(outputs, y)

#                 running_acc  += acc*dataloader.batch_size
#                 running_loss += loss*dataloader.batch_size 

#             epoch_loss = running_loss / len(dataloader.dataset)
#             epoch_acc = running_acc / len(dataloader.dataset)

#             print('{} Loss: {:.4f} Acc: {}'.format(phase, epoch_loss, epoch_acc))
#             #print(f"AllocMem (Mb) {torch.cuda.memory_allocated()/1024/1024}")
#             if phase == 'train':
#                 train_loss.append(epoch_loss) 
#             elif phase=='valid': 
#                 valid_loss.append(epoch_loss)
#             elif phase=='test':
#                 test_loss.append(epoch_loss)

#         if phase == 'valid' and early_stopper.early_stop(epoch_loss):
#             break            
    
#     time_elapsed = time.time() - start
#     print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))    
    
#     save_losses(train_loss, valid_loss, test_loss)
#     return train_loss, valid_loss, test_loss    

# def acc_metric(predb, yb):
#     #return (predb.argmax(dim=1) == yb.cuda()).float().mean()
#     return ((torch.sigmoid(predb) > 0.5) == yb.cuda()).float().mean()


# def run_experiment(lr = 0.01, epochs = 3):
#     image_patches_dir = 'data/images_patches'
#     masks_patches_dir = 'data/masks_patches'
    
#     assert len(os.listdir(image_patches_dir)) == len(os.listdir(masks_patches_dir)), 'mistmatch'
    
#     image_patches = np.array(sorted([os.path.join(image_patches_dir, x) for x in os.listdir(image_patches_dir)]))
#     mask_patches = np.array(sorted([os.path.join(masks_patches_dir, x) for x in os.listdir(masks_patches_dir)]))
#     assert len(image_patches) == len(mask_patches)
    
    
#     ### SPLIT ###
#     splits=KFold(n_splits=3,shuffle=True,random_state=42)

#     total_idxs = np.arange(len(image_patches))
#     train_idxs, val_idxs = train_test_split(total_idxs, train_size = 0.7)
#     val_idxs, test_idxs = train_test_split(val_idxs, train_size = 0.8)

#     train_images = image_patches[train_idxs]
#     val_images = image_patches[val_idxs]
#     test_images = image_patches[test_idxs]

#     train_masks = mask_patches[train_idxs]
#     val_masks = mask_patches[val_idxs]
#     test_masks = mask_patches[test_idxs]
            
#     train_dl, val_dl, test_dl = get_dataloaders(train_images, train_masks, val_images, val_masks, test_images, test_masks)
    
#     unet = UNET(4,1) 
    
#     #check shape on one sample
#     sample = next(iter(train_dl))
#     xb, yb = sample['image'], sample['mask']
#     pred = unet(xb)
#     print('data shape : ' , xb.shape, yb.shape)
#     print('prediction shape: ', pred.shape)
    
#     #hyperparams 
#     loss_fn = nn.BCEWithLogitsLoss()
#     opt = torch.optim.Adam(unet.parameters(), lr = lr)

#     performance = {
#         'train_loss':[],
#         'train_acc':[],
#         'val_loss':[],
#         'val_acc':[],
#         'test_loss':[],
#         'test_acc':[]
#     }
    
#     train_loss, val_loss, test_loss = train(unet, train_dl, val_dl, test_dl, loss_fn, opt, acc_metric, epochs = epochs)    
#     torch.save(unet.state_dict(), 'unet.pth')
    
    
    

# if __name__ =='__main__':
#     run_experiment()
    