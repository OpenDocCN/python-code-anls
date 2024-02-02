# `so-vits-svc\pretrain\meta.py`

```py
# 定义一个函数，返回一个包含多个模型下载信息的字典
def download_dict():
    return {
        "vec768l12": {
            "url": "https://ibm.ent.box.com/shared/static/z1wgl1stco8ffooyatzdwsqn2psd9lrr",
            "output": "./pretrain/checkpoint_best_legacy_500.pt"
        },
        "vec256l9": {
            "url": "https://ibm.ent.box.com/shared/static/z1wgl1stco8ffooyatzdwsqn2psd9lrr",
            "output": "./pretrain/checkpoint_best_legacy_500.pt"
        },
        "hubertsoft": {
            "url": "https://github.com/bshall/hubert/releases/download/v0.1/hubert-soft-0d54a1f4.pt",
            "output": "./pretrain/hubert-soft-0d54a1f4.pt"
        },
        "whisper-ppg-small": {
            "url": "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt",
            "output": "./pretrain/small.pt"
        },
        "whisper-ppg": {
            "url": "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt",
            "output": "./pretrain/medium.pt"
        },
        "whisper-ppg-large": {
            "url": "https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt",
            "output": "./pretrain/large-v2.pt"
        }
    }

# 定义一个函数，根据配置文件路径获取语音编码器的下载链接和输出路径
def get_speech_encoder(config_path="configs/config.json"):
    import json

    # 打开配置文件
    with open(config_path, "r") as f:
        # 读取文件内容
        data = f.read()
        # 将文件内容解析为 JSON 格式
        config = json.loads(data)
        # 获取配置文件中的语音编码器信息
        speech_encoder = config["model"]["speech_encoder"]
        # 调用下载字典函数获取下载信息
        dict = download_dict()

        # 返回语音编码器的下载链接和输出路径
        return dict[speech_encoder]["url"], dict[speech_encoder]["output"]
```