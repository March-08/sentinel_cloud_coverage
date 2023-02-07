### check onnx model ###
import onnx
import onnxruntime
import numpy as np
import torch
from src.model.model_architecture import UNET
import matplotlib.pyplot as plt

def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


if __name__ == '__main__':
    x = torch.randn(1, 4, 224, 224, requires_grad=True)
        
    onnx_model = onnx.load("unet.onnx")
    onnx.checker.check_model(onnx_model)

    ### onnx runtime ###
    ort_session = onnxruntime.InferenceSession("unet.onnx")
        
    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    
    #run on image
    from PIL import Image
    import torchvision.transforms as transforms

    
    img = torch.from_numpy(np.load("data/images_patches/img_377_patch_6.npy"))
    mask = torch.from_numpy(np.load("data/masks_patches/mask_377_patch_6.npy"))
     
    img_norm = (img - img.mean())/img.std()
    img_norm = img_norm.permute(2,0,1).unsqueeze(0) 

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_norm)}
    ort_outs = ort_session.run(None, ort_inputs)
    img_out_y = ort_outs[0]
    pred = img_out_y.squeeze(0)
    pred = torch.sigmoid(torch.from_numpy(pred)) > 0.5
    
    
    plt.subplot(1,3,1)
    plt.imshow(img) # for visualization we have to transpose back to HWC
    plt.subplot(1,3,2)
    plt.imshow(mask)
    plt.subplot(1,3,3)
    plt.imshow(pred)  # for visualization we have to remove 3rd dimension of mask
    plt.show()
    plt.savefig(f'onnx.png')
    

    