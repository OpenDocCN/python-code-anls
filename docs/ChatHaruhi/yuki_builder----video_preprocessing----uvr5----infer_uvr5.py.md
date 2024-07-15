# `.\Chat-Haruhi-Suzumiya\yuki_builder\video_preprocessing\uvr5\infer_uvr5.py`

```py
if __name__ == "__main__":
    # 设置设备为 CUDA，用于 GPU 加速
    device = "cuda"
    # 是否使用半精度浮点数进行计算
    is_half = True
    # 设置模型路径为 DeEchoNormal.pth
    model_path = "uvr5_weights/DeEchoNormal.pth"
    
    # 创建 _audio_pre_new 的实例 pre_fun，传入模型路径、设备、半精度标志和 agg 参数
    pre_fun = _audio_pre_new(model_path=model_path, device=device, is_half=True, agg=10)
    
    # 设置音频文件路径
    audio_path = "雪雪伴奏对消HP5.wav"
    # 设置保存路径
    save_path = "opt"
    # 调用 _path_audio_ 方法处理音频文件，并保存结果到指定路径
    pre_fun._path_audio_(audio_path, save_path, save_path)


这段代码的作用是在主程序中设置设备、加载模型、处理音频文件，并保存处理结果。
```