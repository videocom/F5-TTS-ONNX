# F5-TTS-ONNX
Running the F5-TTS  by ONNX Runtime
- 我们已经尽力优化了代码，但ONNXRuntime-CPU的性能仍然不如PyTorch-CPU。即便将其量化为8位，运行速度反而更慢。但仍值得试试其他的backend providers。
- We have optimized the code as much as possible, but ONNXRuntime-CPU's performance is still not as good as PyTorch-CPU. Quantizing to 8-bit makes it even slower. However, it is still worth trying other backend providers.
- 我们已经更新了代码，以适配截至2024年11月5日的SWivid/F5-TTS，成功地导出为ONNX格式。如果您之前遇到了错误，可以下载最新的代码并再试一次。
- We have updated the code to adapt to SWivid/F5-TTS as of 2024/11/5, successfully exporting to the ONNX format. If you encountered an error before, you can download the latest code and try again.
- See more 看更多項目 -> https://dakeqq.github.io/overview/
