# `so-vits-svc\flask_api_full_song.py`

```py
# 导入所需的库
import io
import numpy as np
import soundfile
from flask import Flask, request, send_file
# 从自定义的模块中导入函数
from inference import infer_tool, slicer

# 创建一个 Flask 应用
app = Flask(__name__)

# 定义一个路由，处理 POST 请求，将音频文件转换为另一种格式的音频文件
@app.route("/wav2wav", methods=["POST"])
def wav2wav():
    # 获取 POST 请求的表单数据
    request_form = request.form
    # 获取音频文件地址
    audio_path = request_form.get("audio_path", None)
    # 获取音调参数
    tran = int(float(request_form.get("tran", 0)))
    # 获取说话人参数
    spk = request_form.get("spk", 0)
    # 获取目标音频文件格式
    wav_format = request_form.get("wav_format", 'wav')
    
    # 格式化音频文件为 WAV 格式
    infer_tool.format_wav(audio_path)
    # 对音频文件进行分段处理
    chunks = slicer.cut(audio_path, db_thresh=-40)
    # 将分段后的音频数据转换为完整的音频数据和采样率
    audio_data, audio_sr = slicer.chunks2audio(audio_path, chunks)

    audio = []
    # 遍历每个分段的音频数据
    for (slice_tag, data) in audio_data:
        # 打印分段开始的标记和时长
        print(f'#=====segment start, {round(len(data) / audio_sr, 3)}s======')

        # 计算音频数据的长度
        length = int(np.ceil(len(data) / audio_sr * svc_model.target_sample))
        if slice_tag:
            # 如果是空段，则跳过
            print('jump empty segment')
            _audio = np.zeros(length)
        else:
            # 对数据进行填充
            pad_len = int(audio_sr * 0.5)
            data = np.concatenate([np.zeros([pad_len]), data, np.zeros([pad_len])])
            raw_path = io.BytesIO()
            # 将数据写入到 WAV 文件中
            soundfile.write(raw_path, data, audio_sr, format="wav")
            raw_path.seek(0)
            # 使用模型进行推理
            out_audio, out_sr = svc_model.infer(spk, tran, raw_path)
            svc_model.clear_empty()
            _audio = out_audio.cpu().numpy()
            pad_len = int(svc_model.target_sample * 0.5)
            _audio = _audio[pad_len:-pad_len]

        # 将处理后的音频数据添加到列表中
        audio.extend(list(infer_tool.pad_array(_audio, length)))
    
    # 创建一个字节流，将音频数据写入到其中
    out_wav_path = io.BytesIO()
    soundfile.write(out_wav_path, audio, svc_model.target_sample, format=wav_format)
    out_wav_path.seek(0)
    # 返回生成的音频文件
    return send_file(out_wav_path, download_name=f"temp.{wav_format}", as_attachment=True)

# 如果当前脚本作为主程序运行，则执行以下代码
if __name__ == '__main__':
    # 模型地址
    model_name = "logs/44k/G_60000.pth"
    # 配置文件地址
    config_name = "configs/config.json"
    # 创建一个名为svc_model的对象，使用infer_tool.Svc类和给定的model_name和config_name参数
    svc_model = infer_tool.Svc(model_name, config_name)
    # 运行应用程序，监听端口1145，绑定到所有网络接口，关闭调试模式，不使用多线程
    app.run(port=1145, host="0.0.0.0", debug=False, threaded=False)
```