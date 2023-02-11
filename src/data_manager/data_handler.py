
from torchvision import transforms
import numpy as np
import torch
from torch.utils.data import Dataset

class SegmentationDataset(Dataset):
  def __init__(self, image_paths, mask_paths, transforms):
    self.image_paths = image_paths
    self.mask_paths = mask_paths
    self.transforms = transforms
  
  def __len__(self):
    return len(self.image_paths)
  
  def __getitem__(self, idx):
    imagePath = self.image_paths[idx]
    image = np.load(imagePath)
    image = torch.from_numpy(image).permute(2,0,1) #(224,224,4) -> (4,224,224)
    
    #normalize
    #image = (image - image.mean())/image.std()
    
    mask = np.load(self.mask_paths[idx])
    mask = torch.from_numpy(mask)
    mask = mask.unsqueeze(0)

    if self.transforms is not None:
      image = self.transforms(image)
      mask = self.transforms(mask)

    return {
        'imagePath': imagePath,
        'maskPath':self.mask_paths[idx],
        'image':image,
        'mask':mask.long()
    }
    
def get_train_transforms():
  return transforms.Compose([
      transforms.RandomHorizontalFlip(p = 0.5),
      transforms.RandomVerticalFlip(p = 0.5),
      transforms.RandomRotation(degrees = 90),
      transforms.GaussianBlur(kernel_size=5),
      transforms.RandomCrop(size=112),
              ])