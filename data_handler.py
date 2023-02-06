
from torchvision import transforms
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def get_train_transforms():
  return transforms.Compose([
      transforms.RandomHorizontalFlip(p = 0.5),
      transforms.RandomVerticalFlip(p = 0.5),
      transforms.RandomRotation(degrees = 90),
      transforms.GaussianBlur(kernel_size=5),
      transforms.RandomCrop(size=112),
              ])
  
  
class SegmentationDataset(Dataset):
  def __init__(self, imagePaths, maskPaths, transforms):
    self.imagePaths = imagePaths
    self.maskPaths = maskPaths
    self.transforms = transforms
  
  def __len__(self):
    return len(self.imagePaths)
  
  def __getitem__(self, idx):
    imagePath = self.imagePaths[idx]
    image = np.load(imagePath)
    image = torch.from_numpy(image).permute(2,0,1)
    
    #normalize
    image = (image - image.mean())/image.std()
    
    mask = np.load(self.maskPaths[idx])
    mask = torch.from_numpy(mask)
    mask =mask.unsqueeze(0)

    if self.transforms is not None:
      image = self.transforms(image)
      mask = self.transforms(mask)

    return {
        'imagePath': imagePath,
        'maskPath':self.maskPaths[idx],
        'image':image,
        'mask':mask.long()
    }
    
def get_dataloaders(train_images, train_masks, val_images, val_masks, test_images, test_masks):
    assert len(train_images) == len(train_masks), 'mismatch!'
    assert len(val_images) == len(val_masks), 'mismatch!'
    assert len(test_images) == len(test_masks), 'mismatch!'
    #assert len(train_images) + len(val_images) + len(test_images) == len(image_patches), 'tot mismatch!'    

    transform =   get_train_transforms()
    train_dataset = SegmentationDataset(train_images, train_masks, transform)
    val_dataset = SegmentationDataset(val_images, val_masks, None)
    test_dataset = SegmentationDataset(test_images, test_masks, None)

    train_dl = DataLoader(train_dataset, batch_size=6, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=6, shuffle=False)
    test_dl = DataLoader(test_dataset, batch_size=6, shuffle=False)
    return train_dl, val_dl, test_dl