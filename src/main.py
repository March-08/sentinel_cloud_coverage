import sys
from data_manager.extract_patches import extract
from model.model_training import run_experiment
from onnx_manager.check_onnx import check_onnx
from onnx_manager.torch_2_onnx import torch_2_onnx
from onnx_manager.inference_onnx import inference


if __name__ == '__main__':
    command = sys.argv[1]
    
    if command.lower() == 'extract':
       extract()
       
    if command.lower() == 'train':
        run_experiment(lr= 0.01, epochs = 1)
       
    if command.lower() == 'onnx':
        torch_2_onnx(torch_model='unet2.pth' , onnx_model= 'unet2.onnx')
        assert check_onnx(torch_model= 'unet2.pth', onnx_model='unet2.onnx'), 'onnx output is not consistent'
    
    if command.lower() == 'inference':
        inference(onnx_model='unet2.onnx', img_path= 'data/images_patches/img_0_patch_6.npy')
        pass
    
   