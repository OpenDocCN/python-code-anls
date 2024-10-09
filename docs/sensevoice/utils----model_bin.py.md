# `.\SenseVoiceSmall-src\utils\model_bin.py`

```
#!/usr/bin/env python3
# 指定脚本的解释器为 Python 3

# -*- encoding: utf-8 -*-
# 设置文件的编码格式为 UTF-8

# Copyright FunASR (https://github.com/FunAudioLLM/SenseVoice). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)
# 版权声明，包含项目的链接和许可证信息

import os.path
# 导入 os.path 模块，用于处理文件路径

from pathlib import Path
# 从 pathlib 模块导入 Path 类，用于更方便的文件路径操作

from typing import List, Union, Tuple
# 从 typing 模块导入类型提示相关的类型

import torch
# 导入 PyTorch 库，用于深度学习模型的推理

import librosa
# 导入 librosa 库，用于音频处理和分析

import numpy as np
# 导入 NumPy 库，用于数值计算和数组操作

from utils.infer_utils import (
    CharTokenizer,
    Hypothesis,
    ONNXRuntimeError,
    OrtInferSession,
    TokenIDConverter,
    get_logger,
    read_yaml,
)
# 从自定义的 infer_utils 模块导入所需的工具和类

from utils.frontend import WavFrontend
# 从自定义的 frontend 模块导入 WavFrontend 类，用于音频前处理

from utils.infer_utils import pad_list
# 从 infer_utils 模块导入 pad_list 函数，用于填充列表

logging = get_logger()
# 获取日志记录器，用于记录日志信息


class SenseVoiceSmallONNX:
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    类的文档字符串，包含作者和项目的简要介绍

    Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition
    指明该模型是一个高效的并行变换器，用于端到端语音识别
    https://arxiv.org/abs/2206.08317
    该模型的论文链接
    """

    def __init__(
        self,
        model_dir: Union[str, Path] = None,
        # 初始化方法的参数，model_dir 为模型文件所在的目录，可以是字符串或 Path 对象

        batch_size: int = 1,
        # batch_size 设置每次推理处理的样本数量，默认为 1

        device_id: Union[str, int] = "-1",
        # device_id 指定使用的设备 ID，默认为 "-1" 表示使用 CPU

        plot_timestamp_to: str = "",
        # plot_timestamp_to 可选参数，用于指定绘图时间戳的输出位置

        quantize: bool = False,
        # quantize 指示是否使用量化模型，默认为 False

        intra_op_num_threads: int = 4,
        # intra_op_num_threads 设置每个操作内部使用的线程数，默认为 4

        cache_dir: str = None,
        # cache_dir 指定缓存目录，默认为 None

        **kwargs,
        # 接受额外的关键字参数
    ):
        if quantize:
            # 如果量化为真，则构建量化模型文件的路径
            model_file = os.path.join(model_dir, "model_quant.onnx")
        else:
            # 否则，构建标准模型文件的路径
            model_file = os.path.join(model_dir, "model.onnx")

        # 构建配置文件的路径
        config_file = os.path.join(model_dir, "config.yaml")
        # 构建均值方差文件的路径
        cmvn_file = os.path.join(model_dir, "am.mvn")
        # 读取配置文件的内容
        config = read_yaml(config_file)

        # token_list = os.path.join(model_dir, "tokens.json")
        # 读取 token 列表的路径（注释掉的代码）

        # with open(token_list, "r", encoding="utf-8") as f:
        #     token_list = json.load(f)
        # 读取 token 列表的内容（注释掉的代码）

        # self.converter = TokenIDConverter(token_list)
        # 创建 TokenIDConverter 对象以处理 token 列表（注释掉的代码）

        self.tokenizer = CharTokenizer()
        # 创建 CharTokenizer 对象，用于字符级的文本标记

        config["frontend_conf"]['cmvn_file'] = cmvn_file
        # 将均值方差文件路径添加到前端配置中

        self.frontend = WavFrontend(**config["frontend_conf"])
        # 初始化 WavFrontend 对象，用于音频信号的预处理

        self.ort_infer = OrtInferSession(
            model_file, device_id, intra_op_num_threads=intra_op_num_threads
        )
        # 创建 OrtInferSession 对象，用于运行 ONNX 模型推理

        self.batch_size = batch_size
        # 保存批处理大小

        self.blank_id = 0
        # 初始化空白 ID，通常用于 CTC 解码
    # 定义可调用类，接收音频内容、语言和文本规范等参数
        def __call__(self, 
                     wav_content: Union[str, np.ndarray, List[str]], 
                     language: List, 
                     textnorm: List,
                     tokenizer=None,
                     **kwargs) -> List:
            # 加载音频数据并获取采样频率
            waveform_list = self.load_data(wav_content, self.frontend.opts.frame_opts.samp_freq)
            # 计算音频数据的数量
            waveform_nums = len(waveform_list)
            # 初始化 ASR 结果列表
            asr_res = []
            # 遍历每批音频数据进行处理
            for beg_idx in range(0, waveform_nums, self.batch_size):
                # 确定当前批次的结束索引
                end_idx = min(waveform_nums, beg_idx + self.batch_size)
                # 提取当前批次的特征和特征长度
                feats, feats_len = self.extract_feat(waveform_list[beg_idx:end_idx])
                # 进行推理，获取 CTC 概率和编码器输出长度
                ctc_logits, encoder_out_lens = self.infer(feats, 
                                     feats_len, 
                                     np.array(language, dtype=np.int32), 
                                     np.array(textnorm, dtype=np.int32)
                                     )
                # 将 CTC 概率转换为 PyTorch 张量
                ctc_logits = torch.from_numpy(ctc_logits).float()
                # 仅支持批量大小为 1
                x = ctc_logits[0, : encoder_out_lens[0].item(), :]
                # 通过 argmax 获取预测序列
                yseq = x.argmax(dim=-1)
                # 去除连续重复的令牌
                yseq = torch.unique_consecutive(yseq, dim=-1)
    
                # 创建掩码以去除空白标记
                mask = yseq != self.blank_id
                # 将有效令牌转换为整数列表
                token_int = yseq[mask].tolist()
                
                # 根据是否提供分词器生成结果
                if tokenizer is not None:
                    asr_res.append(tokenizer.tokens2text(token_int))
                else:
                    asr_res.append(token_int)
            # 返回最终的 ASR 结果
            return asr_res
    
        # 加载音频数据，支持多种输入类型
        def load_data(self, wav_content: Union[str, np.ndarray, List[str]], fs: int = None) -> List:
            # 加载单个 WAV 文件
            def load_wav(path: str) -> np.ndarray:
                waveform, _ = librosa.load(path, sr=fs)
                return waveform
    
            # 处理输入为 numpy 数组的情况
            if isinstance(wav_content, np.ndarray):
                return [wav_content]
    
            # 处理输入为字符串的情况
            if isinstance(wav_content, str):
                return [load_wav(wav_content)]
    
            # 处理输入为列表的情况
            if isinstance(wav_content, list):
                return [load_wav(path) for path in wav_content]
    
            # 抛出类型错误异常
            raise TypeError(f"The type of {wav_content} is not in [str, np.ndarray, list]")
    
        # 提取音频特征
        def extract_feat(self, waveform_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
            # 初始化特征和特征长度的列表
            feats, feats_len = [], []
            # 遍历每个波形以提取特征
            for waveform in waveform_list:
                # 提取梅尔频谱特征
                speech, _ = self.frontend.fbank(waveform)
                # 应用 LFR 和 CMVN 操作
                feat, feat_len = self.frontend.lfr_cmvn(speech)
                # 将提取的特征和长度添加到列表中
                feats.append(feat)
                feats_len.append(feat_len)
    
            # 将特征进行填充以统一长度
            feats = self.pad_feats(feats, np.max(feats_len))
            # 将特征长度转换为 numpy 数组
            feats_len = np.array(feats_len).astype(np.int32)
            # 返回填充后的特征和特征长度
            return feats, feats_len
    
        # 静态方法用于填充特征矩阵
        @staticmethod
        def pad_feats(feats: List[np.ndarray], max_feat_len: int) -> np.ndarray:
            # 定义填充单个特征的内部函数
            def pad_feat(feat: np.ndarray, cur_len: int) -> np.ndarray:
                pad_width = ((0, max_feat_len - cur_len), (0, 0))
                # 使用常量值进行填充
                return np.pad(feat, pad_width, "constant", constant_values=0)
    
            # 对所有特征进行填充
            feat_res = [pad_feat(feat, feat.shape[0]) for feat in feats]
            # 转换为 numpy 数组并返回
            feats = np.array(feat_res).astype(np.float32)
            return feats
    # 定义一个推断方法，接收特征数组及其他参数，返回输出元组
        def infer(self, 
                  feats: np.ndarray,  # 特征数组，表示输入数据
                  feats_len: np.ndarray,  # 特征长度数组，表示每个输入的长度
                  language: np.ndarray,  # 语言数组，表示输入的语言信息
                  textnorm: np.ndarray,) -> Tuple[np.ndarray, np.ndarray]:  # 文本规范化数组，返回输出为元组
            # 调用自定义的推断方法，传入所有必要的参数，获得输出
            outputs = self.ort_infer([feats, feats_len, language, textnorm])
            # 返回推断的输出结果
            return outputs
```