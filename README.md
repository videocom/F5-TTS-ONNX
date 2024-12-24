---

## F5-TTS-ONNX  
Run **F5-TTS** using ONNX Runtime for efficient and flexible text-to-speech processing.

### Updates  
- **2024/12/24 Update**: The code has been updated to support the latest version of [**SWivid/F5-TTS**](https://github.com/SWivid/F5-TTS), enabling successful export to ONNX format. Resolved issues with missing Python package imports. If you encountered errors with previous versions, please download the latest code and try again.
- The latest version accepts audio in `int16` format (short) and also outputs in `int16` format. The previous version supported the float format, but it is no longer supported in the current Inference.py.

### Features  
1. **AMD GPU + Windows OS**:  
   - Easy solution using ONNX-DirectML for AMD GPUs on Windows.  
   - Install ONNX Runtime DirectML:  
     ```bash
     pip install onnxruntime-directml --upgrade
     ```
2. **CPU Only**:
   - For users with 'CPU only' setups, including Intel or AMD, you can try using `['OpenVINOExecutionProvider']` and adding `provider_options` for a slight performance boost of around 20%.
   - ```python
     provider_options =
     [{
        'device_type' : 'CPU',
        'precision' : 'ACCURACY',
        'num_of_threads': MAX_THREADS,
        'num_streams': 1,
        'enable_opencl_throttling' : True,
        'enable_qdq_optimizer': True
     }]
     ```
   - Remember `pip install onnxruntime-openvino --upgrade` first.
   - Next, set `session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC` to reduce conflicts.
3. **Intel OpenVINO**:
   - If you are using a recent Intel chip, you can try `['OpenVINOExecutionProvider']` with provider_options `'device_type': 'XXX'`, where `XXX` can be one of the following options:  (No guarantee that it will work or function well.)
     - `CPU`  
     - `GPU`  
     - `NPU`  
     - `AUTO:NPU,CPU`  
     - `AUTO:NPU,GPU`  
     - `AUTO:GPU,CPU`  
     - `AUTO:NPU,GPU,CPU`  
     - `HETERO:NPU,CPU`  
     - `HETERO:NPU,GPU`  
     - `HETERO:GPU,CPU`  
     - `HETERO:NPU,GPU,CPU`
   - Remember `pip install onnxruntime-openvino --upgrade` first.
   - Next, set `session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC` to reduce conflicts.
4. **Simple GUI Version**:  
   - Try the easy-to-use GUI version:  
     [F5-TTS-ONNX GUI](https://github.com/patientx/F5-TTS-ONNX-gui)

5. **NVIDIA TensorRT Support**:  
   - For NVIDIA GPU optimization with TensorRT, visit:  
     [F5-TTS-TRT](https://github.com/Bigfishering/f5-tts-trtllm/tree/main)

### Learn More  
- Explore more related projects and resources:  
  [Project Overview](https://dakeqq.github.io/overview/)

---

## F5-TTS-ONNX  
通过 ONNX Runtime 运行 **F5-TTS**，实现高效灵活的文本转语音处理。

### 更新  
- **2024/12/24 更新**：代码已更新以支持最新版本的 [**SWivid/F5-TTS**](https://github.com/SWivid/F5-TTS)，成功导出为 ONNX 格式。修复了Python包导入丢失的问题。如果您之前遇到错误，请下载最新代码并重试。
- 最新版本接收的音频格式为 `int16`（short），输出也是 `int16` 格式。上一版本支持 float 格式，但在当前的 Inference.py 中已不再支持。

### 功能  
1. **AMD GPU + Windows 操作系统**：  
   - 针对 AMD GPU 的简单解决方案，通过 ONNX-DirectML 在 Windows 上运行。  
   - 安装 ONNX Runtime DirectML：  
     ```bash
     pip install onnxruntime-directml --upgrade
     ```
**2. 仅CPU：**  
   - 对于仅使用CPU的用户（包括Intel或AMD），可以尝试使用`['OpenVINOExecutionProvider']`并添加`provider_options`，以获得大约20%的性能提升。
   - 示例代码：  
     ```python
     provider_options =
     [{
        'device_type': 'CPU',
        'precision': 'ACCURACY',
        'num_of_threads': MAX_THREADS,
        'num_streams': 1,
        'enable_opencl_throttling': True,
        'enable_qdq_optimizer': True
     }]
     ```  
   - 请记得先执行 `pip install onnxruntime-openvino --upgrade`。  
   - 接下来，将`session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC`，以减少冲突。  

**3. Intel OpenVINO：**  
   - 如果您使用的是近期的Intel芯片，可以尝试`['OpenVINOExecutionProvider']`，并设置`provider_options`中的`'device_type': 'XXX'`，其中`XXX`可以是以下选项之一： (不能保证其能够正常运行或运行良好。)
     - `CPU`  
     - `GPU`  
     - `NPU`  
     - `AUTO:NPU,CPU`  
     - `AUTO:NPU,GPU`  
     - `AUTO:GPU,CPU`  
     - `AUTO:NPU,GPU,CPU`  
     - `HETERO:NPU,CPU`  
     - `HETERO:NPU,GPU`  
     - `HETERO:GPU,CPU`  
     - `HETERO:NPU,GPU,CPU`
   - 请记得先执行 `pip install onnxruntime-openvino --upgrade`。  
   - 接下来，将`session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC`，以减少冲突。  
3. **简单的图形界面版本**：  
   - 体验简单易用的图形界面版本：  
     [F5-TTS-ONNX GUI](https://github.com/patientx/F5-TTS-ONNX-gui)

4. **支持 NVIDIA TensorRT**：  
   - 针对 NVIDIA GPU 的 TensorRT 优化，请访问：  
     [F5-TTS-TRT](https://github.com/Bigfishering/f5-tts-trtllm/tree/main)

### 了解更多  
- 探索更多相关项目和资源：  
  [项目概览](https://dakeqq.github.io/overview/)

---  
