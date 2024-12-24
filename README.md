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

2. **Simple GUI Version**:  
   - Try the easy-to-use GUI version:  
     [F5-TTS-ONNX GUI](https://github.com/patientx/F5-TTS-ONNX-gui)

3. **NVIDIA TensorRT Support**:  
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

2. **简单的图形界面版本**：  
   - 体验简单易用的图形界面版本：  
     [F5-TTS-ONNX GUI](https://github.com/patientx/F5-TTS-ONNX-gui)

3. **支持 NVIDIA TensorRT**：  
   - 针对 NVIDIA GPU 的 TensorRT 优化，请访问：  
     [F5-TTS-TRT](https://github.com/Bigfishering/f5-tts-trtllm/tree/main)

### 了解更多  
- 探索更多相关项目和资源：  
  [项目概览](https://dakeqq.github.io/overview/)

---  
