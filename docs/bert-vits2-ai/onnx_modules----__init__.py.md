# `d:/src/tocomm/Bert-VITS2\onnx_modules\__init__.py`

```
﻿from utils import get_hparams_from_file, load_checkpoint  # 导入自定义的工具函数
import json  # 导入json模块


def export_onnx(export_path, model_path, config_path, novq, dev):
    hps = get_hparams_from_file(config_path)  # 从配置文件中获取超参数
    version = hps.version[0:3]  # 从超参数中获取模型版本号的前三位
    if version == "2.0" or (version == "2.1" and novq):  # 如果版本号是"2.0"或者是"2.1"并且novq为True
        from .V200 import SynthesizerTrn, symbols  # 导入对应版本的模型和符号
    elif version == "2.1" and (not novq):  # 如果版本号是"2.1"并且novq为False
        from .V210 import SynthesizerTrn, symbols  # 导入对应版本的模型和符号
    elif version == "2.2":  # 如果版本号是"2.2"
        if novq and dev:  # 如果novq和dev都为True
            from .V220_novq_dev import SynthesizerTrn, symbols  # 导入对应版本的模型和符号
        else:  # 否则
            from .V220 import SynthesizerTrn, symbols  # 导入对应版本的模型和符号
    elif version == "2.3":  # 如果版本号是"2.3"
        from .V230 import SynthesizerTrn, symbols  # 导入对应版本的模型和符号
    net_g = SynthesizerTrn(  # 创建SynthesizerTrn对象
        len(symbols),  # 传入符号的长度作为参数
        hps.data.filter_length // 2 + 1,  # 计算滤波器长度的一半加1
        hps.train.segment_size // hps.data.hop_length,  # 计算训练段的大小除以跳跃长度
        n_speakers=hps.data.n_speakers,  # 设置说话者数量
        **hps.model,  # 使用模型超参数
    )
    _ = net_g.eval()  # 将网络设置为评估模式
    _ = load_checkpoint(model_path, net_g, None, skip_optimizer=True)  # 加载模型的检查点
    net_g.cpu()  # 将网络移动到 CPU 上
    net_g.export_onnx(export_path)  # 导出网络模型为 ONNX 格式

    spklist = []  # 创建一个空列表用于存储说话者列表
    for key in hps.data.spk2id.keys():  # 遍历说话者到ID的映射
        spklist.append(key)  # 将说话者添加到列表中

    MoeVSConf = {  # 创建一个包含模型导出配置的字典
        "Folder": f"{export_path}",  # 设置导出文件夹路径
        "Name": f"{export_path}",  # 设置导出文件名
        "Type": "BertVits",  # 设置模型类型为BertVits
        "Symbol": symbols,  # 设置符号
        "Cleaner": "",  # 设置清洁器
        "Rate": hps.data.sampling_rate,  # 设置采样率为hps.data.sampling_rate
        "CharaMix": True,  # 设置CharaMix为True
        "Characters": spklist,  # 设置Characters为spklist
        "LanguageMap": {"ZH": [0, 0], "JP": [1, 6], "EN": [2, 8]},  # 设置LanguageMap为指定的字典
        "Dict": "BasicDict",  # 设置Dict为"BasicDict"
        "BertPath": [  # 设置BertPath为包含三个路径的列表
            "chinese-roberta-wwm-ext-large",
            "deberta-v2-large-japanese",
            "bert-base-japanese-v3",
        ],
        "Clap": "clap-htsat-fused",  # 设置Clap为"clap-htsat-fused"
    }

    with open(f"onnx/{export_path}.json", "w") as MoeVsConfFile:  # 打开文件"onnx/{export_path}.json"，以写入模式
        json.dump(MoeVSConf, MoeVsConfFile, indent=4)  # 将MoeVSConf以JSON格式写入MoeVsConfFile，并使用缩进格式化
```