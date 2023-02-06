### check onnx model ###
import onnx
import onnxruntime
import numpy as np
import torch
from model_architecture import UNET

def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

if __name__ == '__main__':
    unet = UNET(4,1)
    unet.load_state_dict(torch.load('unet.pth'))
    unet.eval()
    x = torch.randn(1, 4, 224, 224, requires_grad=True)
    torch_out = unet(x)
    
    
    onnx_model = onnx.load("unet.onnx")
    onnx.checker.check_model(onnx_model)

    ### onnx runtime ###
    ort_session = onnxruntime.InferenceSession("unet.onnx")
        
    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")