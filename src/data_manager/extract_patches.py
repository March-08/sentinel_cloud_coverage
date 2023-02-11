from patchify import patchify
import numpy as np
import os

STEP = 146
NIR_BAND = 7 #ask
RGBN_CHANNELS = [3,2,1, NIR_BAND]

def get_subscene_channels(subscene, channels:list):
  return subscene[...,channels]

def mask_2_binary_mask(mask:np.ndarray) -> np.ndarray:
  return mask[:,:,1]

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

      image_patches = patchify(image = image, patch_size = (224,224,4), step = STEP)
      image_patches = image_patches.reshape(-1, 224, 224, 4)

      mask_patches = patchify(image = mask, patch_size = (224,224), step = STEP)
      mask_patches = mask_patches.reshape(-1, 224, 224)

      for n, data in enumerate(zip(image_patches, mask_patches)):
        img, mask = data[0], data[1]
        if mask.sum() < 2500 or mask.sum() >= 4800: #2500 is 5% of 224^2, 4800 is almost total
            total_edge_cases +=1
            if np.random.uniform(0,1) >= 0.9:
              included_edge_cases +=1
              continue
        #img
        out = os.path.join(images_pathces_path,f"img_{i}_patch_{n}.npy")
        np.save(out, img)
        #mask
        out = os.path.join(masks_pathces_path,f"mask_{i}_patch_{n}.npy")
        np.save(out, mask)
      
      #free memory
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
  create_patches('data', subscenes_paths, masks_paths)
  

if __name__ == '__main__':
  extract()