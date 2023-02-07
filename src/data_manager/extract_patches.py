#!pip install 

from patchify import patchify
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


STEP = 146

def get_subscene_channels(subscene, channels:list):
  return subscene[...,channels]

def mask_2_binary_mask(mask:np.ndarray) -> np.ndarray:
  clear_annotation = mask[:,:,0]
  cloud_annotation = mask[:,:,1]
  cloud_shadow_annotation = mask[:,:,2]
  assert clear_annotation.shape == cloud_annotation.shape == cloud_shadow_annotation.shape, 'shape not matching'
  
  #1 where cloud is present 0 if clear of cloud shadow 
  binary_mask = cloud_annotation
  return binary_mask


def image_2_patches(image, patch_shape:tuple = (224,224,4) , step:int = STEP):
    patches = patchify(image = image, patch_size = patch_shape , step = STEP)
    patches = patches.reshape(-1,patch_shape[0],patch_shape[1],patch_shape[2])
    return patches

def create_patches(output, images_paths, masks_paths) -> None:
    '''
    params:
      output : out directory path
      image_paths : paths of  subscenes
      masks_paths: paths of masks
    
    creates 2 subfolder with all patches of subscenes and masks
    '''
    print('creating patches...')
    images_pathces_path = os.path.join(output, 'images_patches')
    masks_pathces_path = os.path.join(output, 'masks_patches')
    
    if not os.path.isdir(images_pathces_path):
      os.makedirs(images_pathces_path)

    if not os.path.isdir(masks_pathces_path):
      os.makedirs(masks_pathces_path)
    
    total_edge_cases = 0
    included_edge_cases = 0
    for i, data in enumerate(zip(images_paths, masks_paths)):
      image = np.load(data[0])
      image = get_subscene_channels(image, RGBN_CHANNELS)

      mask = np.load(data[1])
      mask = mask_2_binary_mask(mask)

      #image
      image_patches = patchify(image = image, patch_size = (224,224,4), step = STEP)
      image_patches = image_patches.reshape(-1, 224, 224, 4)

      mask_patches = patchify(image = mask, patch_size = (224,224), step = STEP)
      mask_patches = mask_patches.reshape(-1, 224, 224)

      for n, data in enumerate(zip(image_patches, mask_patches)):
        img, mask = data[0], data[1]
        if mask.sum() == 0 or mask.sum() == 224*224:
            total_edge_cases +=1
            if np.random.uniform(0,1) >= 0.98:
              included_edge_cases +=1
              continue
        
        #img
        out = os.path.join(images_pathces_path,f"img_{i}_patch_{n}.npy")
        np.save(out, img)
        #mask
        out = os.path.join(masks_pathces_path,f"mask_{i}_patch_{n}.npy")
        np.save(out, mask)
      
      del image_patches 
      del mask_patches
    print('patches created!')
    print('total edge cases : ', total_edge_cases)
    print('included edge cases : ', included_edge_cases)


def extract():
  #extracted
  masks_extracted = 'data/masks'
  subscenes_extracted = 'data/subscenes'
  
  subscenes_paths = [os.path.join(subscenes_extracted,x) for x in os.listdir(subscenes_extracted)]
  masks_paths = [os.path.join(masks_extracted,x) for x in os.listdir(masks_extracted)]
  assert len(subscenes_paths) == len(masks_paths), 'not matching'
  
  NIR_BAND = 9 #ask
  RGBN_CHANNELS = [3,2,1, NIR_BAND]
  RGB_CHANNELS = [3,2,1]
  
  create_patches('data', subscenes_paths, masks_paths)
  


if __name__ == '__main__':
  extract()