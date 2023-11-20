# coding:utf-8
import time

import numpy as np
import onnxruntime
import torch
from Object_Detection.nailfold_classifier.model_backbone import Backbone


def pytorch_2_onnx(model_path, export_model_path, model_name, out_dimensions):
    batch_size = 8

    backbone = Backbone(out_dimension=out_dimensions, model_name=model_name, pretrained=False)
    model, _, _ = backbone.build_model()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    input_names = ["input_data"]
    output_names = ["output"]
    dummy_input = torch.rand(batch_size, 1, 64, 1001)

    torch.onnx.export(model, dummy_input, export_model_path, verbose=True,
                      input_names=input_names, output_names=output_names,
                      dynamic_axes={
                          "input_data": {0: "batch_size"},
                          "output": {0: "batch_size"}
                      })
    return export_model_path


def onnx_test(export_model_path):
    session = onnxruntime.InferenceSession(export_model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print("Input Name:", input_name)
    print("Output Name:", output_name)

    input_data = np.random.randn(224, 224)
    input_data = input_data[np.newaxis, np.newaxis, :, :]
    print("input_data: ", input_data.shape)

    total_time = 0
    for _ in range(100):
        start_time = time.time()
        result = session.run(None, {input_name: input_data})
        result = result[0]
        print(result.shape)

        end_time = time.time()
        duration_time = int((end_time - start_time) * 1000)
        total_time += duration_time
    avg_time = total_time / 100
    print("avg time: {} ms".format(avg_time))
