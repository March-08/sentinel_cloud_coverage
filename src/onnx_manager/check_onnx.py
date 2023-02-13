### check onnx model ###
import onnx
import onnxruntime
import numpy as np
import torch
import os
from model.model_architecture import UNET

def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def check_onnx(torch_model:str = 'outputs/unet.pth', onnx_model:str = 'outputs/unet.onnx') -> bool:
        unet = UNET(4,2)
        unet.load_state_dict(torch.load(torch_model))
        unet.eval()
        x = torch.randn(1, 4, 224, 224, requires_grad=True)
        torch_out = unet(x)
        
        assert os.path.getsize(onnx_model)<2500000, 'onnx model is to large'

        onnx_model = onnx.load(onnx_model)
        onnx.checker.check_model(onnx_model)

        ### onnx runtime ###
        ort_session = onnxruntime.InferenceSession(onnx_model.SerializeToString())
                
        # compute ONNX Runtime output prediction
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
        ort_outs = ort_session.run(None, ort_inputs)

        # compare ONNX Runtime and PyTorch results
        np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

        print("Exported model has been tested with ONNXRuntime, and the result looks good!")
        return True
        
if __name__ == '__main__':
    check_onnx()