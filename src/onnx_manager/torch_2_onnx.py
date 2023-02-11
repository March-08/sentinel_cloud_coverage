import torch.onnx
from model.model_architecture import UNET

def torch_2_onnx(torch_model:str = 'outputs/unet.pth',  onnx_model= 'outputs/unet.onnx'):
    unet = UNET(4,1)
    unet.load_state_dict(torch.load(torch_model))
    unet.eval()
    x = torch.randn(1, 4, 224, 224, requires_grad=True)
    torch_out = unet(x)
    
    # Export the model
    torch.onnx.export(unet,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  onnx_model,   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})
    

if __name__ == '__main__':
    torch_2_onnx()
    
    