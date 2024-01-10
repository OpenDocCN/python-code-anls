# `so-vits-svc\flask_api.py`

```
# 导入所需的库
import io
import logging
import soundfile
import torch
import torchaudio
from flask import Flask, request, send_file
from flask_cors import CORS
# 从自定义的推理工具中导入实时语音转换和声码器
from inference.infer_tool import RealTimeVC, Svc

# 创建一个 Flask 应用
app = Flask(__name__)

# 允许跨域请求
CORS(app)

# 设置 numba 库的日志级别为 WARNING
logging.getLogger('numba').setLevel(logging.WARNING)

# 定义一个路由，处理 POST 请求，用于语音转换模型
@app.route("/voiceChangeModel", methods=["POST"])
def voice_change_model():
    # 获取请求表单数据
    request_form = request.form
    # 获取上传的音频文件
    wave_file = request.files.get("sample", None)
    # 获取变调信息
    f_pitch_change = float(request_form.get("fPitchChange", 0))
    # 获取 DAW 所需的采样率
    daw_sample = int(float(request_form.get("sampleRate", 0)))
    # 获取说话者的 ID
    speaker_id = int(float(request_form.get("sSpeakId", 0)))
    # 从 HTTP 请求中获取 WAV 文件并转换为字节流
    input_wav_path = io.BytesIO(wave_file.read())

    # 模型推理
    if raw_infer:
        # 使用 SVC 模型进行推理，获取输出音频和采样率
        out_audio, out_sr = svc_model.infer(speaker_id, f_pitch_change, input_wav_path, cluster_infer_ratio=0,
                                            auto_predict_f0=False, noice_scale=0.4, f0_filter=False)
        # 将输出音频进行重采样
        tar_audio = torchaudio.functional.resample(out_audio, svc_model.target_sample, daw_sample)
    else:
        # 使用 SVC 模型进行推理，获取输出音频
        out_audio = svc.process(svc_model, speaker_id, f_pitch_change, input_wav_path, cluster_infer_ratio=0,
                                auto_predict_f0=False, noice_scale=0.4, f0_filter=False)
        # 将输出音频进行重采样
        tar_audio = torchaudio.functional.resample(torch.from_numpy(out_audio), svc_model.target_sample, daw_sample)
    # 将输出音频写入字节流
    out_wav_path = io.BytesIO()
    soundfile.write(out_wav_path, tar_audio.cpu().numpy(), daw_sample, format="wav")
    out_wav_path.seek(0)
    # 返回输出音频文件
    return send_file(out_wav_path, download_name="temp.wav", as_attachment=True)

# 如果当前脚本被直接执行，则执行以下代码
if __name__ == '__main__':
    # 设置是否进行原始推理
    raw_infer = True
    # 模型名称
    model_name = "logs/32k/G_174000-Copy1.pth"
    # 定义配置文件的路径
    config_name = "configs/config.json"
    # 定义集群模型的路径
    cluster_model_path = "logs/44k/kmeans_10000.pt"
    # 使用给定的模型名称、配置文件路径和集群模型路径创建 SVC 对象
    svc_model = Svc(model_name, config_name, cluster_model_path=cluster_model_path)
    # 创建实时语音转换对象
    svc = RealTimeVC()
    # 启动应用程序，监听指定端口，指定主机，关闭调试模式，不使用多线程
    # 此处与vst插件对应，不建议更改
    app.run(port=6842, host="0.0.0.0", debug=False, threaded=False)
```