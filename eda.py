import matplotlib.pyplot as plt
import numpy as np
import os

if __name__ == '__main__':
    image_patches_dir = 'data/images_patches'
    masks_patches_dir = 'data/masks_patches'
    
    assert len(os.listdir(image_patches_dir)) == len(os.listdir(masks_patches_dir)), 'mistmatch'
    print(len(os.listdir(image_patches_dir)), len(os.listdir(masks_patches_dir)))   
    
    images_paths = sorted([os.path.join(image_patches_dir, x) for x in os.listdir(image_patches_dir)])
    masks_paths = sorted([os.path.join(masks_patches_dir, x) for x in os.listdir(masks_patches_dir)])
    assert len(images_paths) == len(masks_paths)
    
    for i in range(len(images_paths)):
        img = np.load(images_paths[i])
        mask = np.load(masks_paths[i])
        
        plt.subplot(1,2,1)
        plt.imshow(img[:,:,[0,1,2]]) # for visualization we have to transpose back to HWC
        plt.subplot(1,2,2)
        plt.imshow(mask)  # for visualization we have to remove 3rd dimension of mask
        plt.show()
        plt.savefig(f'{i}.png')
        
        
        
        