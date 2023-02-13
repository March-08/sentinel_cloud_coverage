import sys
from data_manager.extract_patches import extract
from model.model_training import run_experiment
from onnx_manager.check_onnx import check_onnx
from onnx_manager.torch_2_onnx import torch_2_onnx


if __name__ == '__main__':
    command =sys.argv[1]
    
    if command.lower() == 'extract':
       extract()
       
    if command.lower() == 'train':
        run_experiment(kfolds=3, lr = 0.005, epochs= 12, batch_size= 64)
       
    if command.lower() == 'onnx':
        torch_2_onnx(torch_model='outputs/unet.pth' , onnx_model= 'outputs/unet.onnx')
        assert check_onnx(torch_model= 'outputs/unet.pth', onnx_model='outputs/unet.onnx'), 'onnx output is not consistent'
    
    
   