# `Bert-VITS2\onnx_modules\__init__.py`

```py
# 从 utils 模块中导入 get_hparams_from_file 和 load_checkpoint 函数
from utils import get_hparams_from_file, load_checkpoint
# 导入 json 模块
import json

# 定义 export_onnx 函数，接受导出路径、模型路径、配置路径、novq 和 dev 作为参数
def export_onnx(export_path, model_path, config_path, novq, dev):
    # 从配置文件中获取超参数
    hps = get_hparams_from_file(config_path)
    # 获取模型版本号
    version = hps.version[0:3]
    # 根据模型版本号和 novq 参数导入对应的模型和符号
    if version == "2.0" or (version == "2.1" and novq):
        from .V200 import SynthesizerTrn, symbols
    elif version == "2.1" and (not novq):
        from .V210 import SynthesizerTrn, symbols
    elif version == "2.2":
        if novq and dev:
            from .V220_novq_dev import SynthesizerTrn, symbols
        else:
            from .V220 import SynthesizerTrn, symbols
    elif version == "2.3":
        from .V230 import SynthesizerTrn, symbols
    # 根据超参数和符号创建 SynthesizerTrn 对象
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    )
    # 将网络设置为评估模式
    _ = net_g.eval()
    # 加载模型检查点
    _ = load_checkpoint(model_path, net_g, None, skip_optimizer=True)
    # 将网络移动到 CPU
    net_g.cpu()
    # 导出模型为 ONNX 格式
    net_g.export_onnx(export_path)

    # 构建说话人列表
    spklist = []
    for key in hps.data.spk2id.keys():
        spklist.append(key)

    # 构建 MoeVSConf 字典
    MoeVSConf = {
        "Folder": f"{export_path}",
        "Name": f"{export_path}",
        "Type": "BertVits",
        "Symbol": symbols,
        "Cleaner": "",
        "Rate": hps.data.sampling_rate,
        "CharaMix": True,
        "Characters": spklist,
        "LanguageMap": {"ZH": [0, 0], "JP": [1, 6], "EN": [2, 8]},
        "Dict": "BasicDict",
        "BertPath": [
            "chinese-roberta-wwm-ext-large",
            "deberta-v2-large-japanese",
            "bert-base-japanese-v3",
        ],
        "Clap": "clap-htsat-fused",
    }

    # 将 MoeVSConf 字典以 JSON 格式写入文件
    with open(f"onnx/{export_path}.json", "w") as MoeVsConfFile:
        json.dump(MoeVSConf, MoeVsConfFile, indent=4)
```