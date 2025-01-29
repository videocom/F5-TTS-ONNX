
import onnxruntime.tools.add_openvino_win_libs as utils
utils.add_openvino_libs_to_path()
import onnxruntime as ort
import numpy as np

# Print the available execution providers
print(ort.get_available_providers())

# Load the ONNX model
# https://huggingface.co/onnx-community/tiny-random-LlamaForCausalLM-ONNX/resolve/main/onnx/model_fp16.onnx

model_path  = "c:/Test/F5/models/model_fp16.onnx" 
cache_dir   = "c:/temp/ov"

ORT_Accelerate_Providers = ['OpenVINOExecutionProvider']  # 'CPUExecutionProvider'
# ORT_Accelerate_Providers = ['CPUExecutionProvider']   

if "OpenVINOExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [
        {
            'device_type': 'GPU',
            #'precision': 'ACCURACY',
            'precision': 'FP16',
            #'num_of_threads': MAX_THREADS,
            'num_streams': 1,
            'enable_opencl_throttling': True,
            'enable_qdq_optimizer': False,
            'cache_dir': cache_dir
        }
    ]
else:
    provider_options = None
 

session_opts = ort.SessionOptions()
session_opts.log_severity_level = 3

#session_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

session = ort.InferenceSession(model_path, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
 

# Prepare feeds (model inputs)
feeds = {
    "input_ids": np.array([[1]], dtype=np.int64), 
    "attention_mask": np.array([[1]], dtype=np.int64),  
    "position_ids": np.array([[0]], dtype=np.int64),  
    "past_key_values.0.key": np.empty([1, 2, 0, 16], dtype=np.float16),  
    "past_key_values.0.value": np.empty([1, 2, 0, 16], dtype=np.float16), 
}

# Run the model
results = session.run(None, feeds) 

# Print the results
for name, output in zip(session.get_outputs(), results):
    print(f"{name.name}: {output}")

