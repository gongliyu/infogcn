from pathlib import Path
from argparse import ArgumentParser
import time

import torch

from engine import SpVGCNTraining

import os


def main(ckpt_path, torch_model_save_to=None, onnx_model_save_to=None):
    model = SpVGCNTraining.load_from_checkpoint(ckpt_path)
    model.eval()
    model.freeze()
    model = model.model
    model.eval().to('cpu')

    if torch_model_save_to:
        x = torch.randn(1, 2, 15, 17, 1)
        y = model(x) # run an extra inference to warm-up
        t0 = time.time()
        y = model(x).detach().cpu().numpy()
        t = time.time() - t0
        print(f'PyTorch inference latency: {t}s')

        if torch.cuda.is_available():
            x = x.to('cuda')
            model = model.to('cuda')
            y = model(x)
            t0 = time.time()
            y = model(x).detach().cpu().numpy()
            t = time.time() - t0
            print(f'PyTorch cuda inference latency: {t}s')

        torch.save(model.state_dict(), torch_model_save_to)

    if onnx_model_save_to:
        import onnxruntime as ort
        x = torch.randn(1, 2, 15, 17, 1)
        torch.onnx.export(model,  # model being run
                          x,  # model input (or a tuple for multiple inputs)
                          onnx_model_save_to,  # where to save the model (can be a file or file-like object)
                          # export_params=True,  # store the trained parameter weights inside the model file
                          opset_version=14,  # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names=['input'],  # the model's input names
                          output_names=['output'],  # the model's output names
                          dynamic_axes={'input': {0: 'batch_size', 2: 'image_height', 3: 'image_width'},
                                        # variable length axes
                                        'output': {0: 'batch_size'}})
        sess = ort.InferenceSession(str(onnx_model_save_to),
                                    providers=['TensorrtExecutionProvider',
                                               'CUDAExecutionProvider',
                                               'CPUExecutionProvider'])
        input_name = sess.get_inputs()[0].name
        _ = sess.run(None, {input_name: x.detach().cpu().numpy()})
        _ = sess.run(None, {input_name: x.detach().cpu().numpy()})
        t0 = time.time()
        _ = sess.run(None, {input_name: x.detach().cpu().numpy()})[0]
        t = time.time() - t0
        print(f'onnx model latency: {t}s')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('ckpt_path', type=Path)
    parser.add_argument('--torch-model-save-to', type=Path, default=None)
    parser.add_argument('--onnx-model-save-to', type=Path, default=None)
    clargs = parser.parse_args()

    main(**vars(clargs))