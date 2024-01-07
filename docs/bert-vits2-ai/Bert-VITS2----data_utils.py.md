# `Bert-VITS2\data_utils.py`

```

# 导入所需的库
import os
import random
import torch
import torch.utils.data
from tqdm import tqdm
from tools.log import logger
import commons
from mel_processing import spectrogram_torch, mel_spectrogram_torch
from utils import load_wav_to_torch, load_filepaths_and_text
from text import cleaned_text_to_sequence
from config import config

"""Multi speaker version"""

# 定义一个类，用于对模型输入和目标进行零填充
class TextAudioSpeakerCollate:
    """Zero-pads model inputs and targets"""

    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text, audio and speaker identities
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized, sid]
        """
        # 对文本序列进行右侧零填充，使其长度等于最大输入长度
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]), dim=0, descending=True
        )

        # 计算最大文本长度、最大频谱长度和最大音频长度
        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])

        # 创建存储长度信息的张量
        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        sid = torch.LongTensor(len(batch))

        # 创建存储零填充后数据的张量
        text_padded = torch.LongTensor(len(batch), max_text_len)
        tone_padded = torch.LongTensor(len(batch), max_text_len)
        language_padded = torch.LongTensor(len(batch), max_text_len)
        bert_padded = torch.FloatTensor(len(batch), 1024, max_text_len)
        ja_bert_padded = torch.FloatTensor(len(batch), 1024, max_text_len)
        en_bert_padded = torch.FloatTensor(len(batch), 1024, max_text_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        
        # 将零填充后的张量初始化为零
        text_padded.zero_()
        tone_padded.zero_()
        language_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        bert_padded.zero_()
        ja_bert_padded.zero_()
        en_bert_padded.zero_()

        # 遍历排序后的批次
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            # 处理文本数据
            text = row[0]
            text_padded[i, : text.size(0)] = text
            text_lengths[i] = text.size(0)

            # 处理频谱数据
            spec = row[1]
            spec_padded[i, :, : spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            # 处理音频数据
            wav = row[2]
            wav_padded[i, :, : wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            # 处理说话者身份数据
            sid[i] = row[3]

            # 处理音调数据
            tone = row[4]
            tone_padded[i, : tone.size(0)] = tone

            # 处理语言数据
            language = row[5]
            language_padded[i, : language.size(0)] = language

            # 处理BERT数据
            bert = row[6]
            bert_padded[i, :, : bert.size(1)] = bert

            # 处理日语BERT数据
            ja_bert = row[7]
            ja_bert_padded[i, :, : ja_bert.size(1)] = ja_bert

            # 处理英语BERT数据
            en_bert = row[8]
            en_bert_padded[i, :, : en_bert.size(1)] = en_bert

        # 返回零填充后的数据
        return (
            text_padded,
            text_lengths,
            spec_padded,
            spec_lengths,
            wav_padded,
            wav_lengths,
            sid,
            tone_padded,
            language_padded,
            bert_padded,
            ja_bert_padded,
            en_bert_padded,
        )

```