# `.\diffusers\pipelines\deprecated\spectrogram_diffusion\midi_utils.py`

```py
# 版权声明，标明文件的版权信息
# Copyright 2022 The Music Spectrogram Diffusion Authors.
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# 许可证信息，指出文件使用的许可证类型
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# 可以通过以下链接获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 在未适用或未书面同意的情况下，软件在“按现状”基础上分发，
# 不提供任何明示或暗示的保证或条件
# 查看许可证以获取特定的权限和限制

# 导入 dataclasses 模块，用于创建数据类
import dataclasses
# 导入 math 模块，提供数学函数
import math
# 导入 os 模块，用于操作系统功能
import os
# 从 typing 模块导入各种类型提示
from typing import Any, Callable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

# 导入 numpy 库，常用于数值计算
import numpy as np
# 导入 PyTorch 库
import torch
# 从 PyTorch 中导入功能模块，提供各种函数
import torch.nn.functional as F

# 从 utils 模块导入 is_note_seq_available 函数，检查 note-seq 是否可用
from ....utils import is_note_seq_available
# 从 pipeline_spectrogram_diffusion 模块导入 TARGET_FEATURE_LENGTH 常量
from .pipeline_spectrogram_diffusion import TARGET_FEATURE_LENGTH

# 检查 note-seq 库是否可用，如果可用则导入，否则抛出导入错误
if is_note_seq_available():
    import note_seq
else:
    raise ImportError("Please install note-seq via `pip install note-seq`")

# 定义输入特征的长度
INPUT_FEATURE_LENGTH = 2048

# 定义音频的采样率
SAMPLE_RATE = 16000
# 定义每帧的跳跃大小
HOP_SIZE = 320
# 根据采样率和跳跃大小计算帧率
FRAME_RATE = int(SAMPLE_RATE // HOP_SIZE)

# 定义每秒的默认步骤数
DEFAULT_STEPS_PER_SECOND = 100
# 定义默认的最大偏移时间（秒）
DEFAULT_MAX_SHIFT_SECONDS = 10
# 定义默认的速度区间数量
DEFAULT_NUM_VELOCITY_BINS = 1

# 定义一个字典，映射乐器名称到其对应的程序编号
SLAKH_CLASS_PROGRAMS = {
    "Acoustic Piano": 0,
    "Electric Piano": 4,
    "Chromatic Percussion": 8,
    "Organ": 16,
    "Acoustic Guitar": 24,
    "Clean Electric Guitar": 26,
    "Distorted Electric Guitar": 29,
    "Acoustic Bass": 32,
    "Electric Bass": 33,
    "Violin": 40,
    "Viola": 41,
    "Cello": 42,
    "Contrabass": 43,
    "Orchestral Harp": 46,
    "Timpani": 47,
    "String Ensemble": 48,
    "Synth Strings": 50,
    "Choir and Voice": 52,
    "Orchestral Hit": 55,
    "Trumpet": 56,
    "Trombone": 57,
    "Tuba": 58,
    "French Horn": 60,
    "Brass Section": 61,
    "Soprano/Alto Sax": 64,
    "Tenor Sax": 66,
    "Baritone Sax": 67,
    "Oboe": 68,
    "English Horn": 69,
    "Bassoon": 70,
    "Clarinet": 71,
    "Pipe": 73,
    "Synth Lead": 80,
    "Synth Pad": 88,
}

# 定义一个数据类，用于配置音符表示
@dataclasses.dataclass
class NoteRepresentationConfig:
    """Configuration note representations."""
    # 是否仅包含音符的开始时间
    onsets_only: bool
    # 是否包含连音符
    include_ties: bool

# 定义一个数据类，表示音符事件数据
@dataclasses.dataclass
class NoteEventData:
    pitch: int  # 音高
    velocity: Optional[int] = None  # 音量，可选
    program: Optional[int] = None  # 乐器程序编号，可选
    is_drum: Optional[bool] = None  # 是否为鼓乐器，可选
    instrument: Optional[int] = None  # 乐器类型编号，可选

# 定义一个数据类，表示音符编码状态
@dataclasses.dataclass
class NoteEncodingState:
    """Encoding state for note transcription, keeping track of active pitches."""
    # 存储活动音高及其对应的速度区间
    active_pitches: MutableMapping[Tuple[int, int], int] = dataclasses.field(default_factory=dict)

# 定义一个数据类，表示事件范围
@dataclasses.dataclass
class EventRange:
    type: str  # 事件类型
    min_value: int  # 最小值
    max_value: int  # 最大值

# 定义一个数据类，表示事件
@dataclasses.dataclass
class Event:
    type: str  # 事件类型
    value: int  # 事件值

# 定义一个名为 Tokenizer 的类
class Tokenizer:
    # 初始化方法，用于创建对象时传入正则 ID 的数量
    def __init__(self, regular_ids: int):
        # 特殊标记的数量：0=PAD，1=EOS，2=UNK
        self._num_special_tokens = 3
        # 存储正则 ID 的数量
        self._num_regular_tokens = regular_ids
    
    # 编码方法，接收一组 token ID
    def encode(self, token_ids):
        # 初始化一个空列表，用于存储编码后的 token
        encoded = []
        # 遍历传入的每个 token ID
        for token_id in token_ids:
            # 检查 token ID 是否在有效范围内
            if not 0 <= token_id < self._num_regular_tokens:
                # 如果不在范围内，抛出值错误异常
                raise ValueError(
                    f"token_id {token_id} does not fall within valid range of [0, {self._num_regular_tokens})"
                )
            # 将有效的 token ID 加上特殊标记数量，并添加到编码列表中
            encoded.append(token_id + self._num_special_tokens)
    
        # 在编码列表末尾添加 EOS 标记
        encoded.append(1)
    
        # 将编码列表填充至 INPUT_FEATURE_LENGTH 长度，使用 PAD 标记
        encoded = encoded + [0] * (INPUT_FEATURE_LENGTH - len(encoded))
    
        # 返回最终编码后的列表
        return encoded
# 编码和解码事件的类
class Codec:
    """Encode and decode events.

    用于声明词汇的特定范围。打算在编码前或解码后使用GenericTokenVocabulary。此类更轻量，不包含诸如EOS或UNK令牌处理等内容。

    为确保“shift”事件始终是词汇的第一个块并从0开始，该事件类型是必需的并单独指定。
    """

    # 初始化方法，定义Codec
    def __init__(self, max_shift_steps: int, steps_per_second: float, event_ranges: List[EventRange]):
        """定义Codec。

        参数：
          max_shift_steps: 可编码的最大移位步数。
          steps_per_second: 移位步数将被解释为持续时间为1 / steps_per_second。
          event_ranges: 其他支持的事件类型及其范围。
        """
        # 设置每秒的步数
        self.steps_per_second = steps_per_second
        # 创建移位范围，最小值为0，最大值为max_shift_steps
        self._shift_range = EventRange(type="shift", min_value=0, max_value=max_shift_steps)
        # 将移位范围与其他事件范围合并
        self._event_ranges = [self._shift_range] + event_ranges
        # 确保所有事件类型具有唯一名称
        assert len(self._event_ranges) == len({er.type for er in self._event_ranges})

    # 计算事件类别的总数
    @property
    def num_classes(self) -> int:
        return sum(er.max_value - er.min_value + 1 for er in self._event_ranges)

    # 下面的方法是仅针对移位事件的简化特例方法，打算在autograph函数中使用

    # 检查给定索引是否为移位事件索引
    def is_shift_event_index(self, index: int) -> bool:
        return (self._shift_range.min_value <= index) and (index <= self._shift_range.max_value)

    # 获取最大移位步数
    @property
    def max_shift_steps(self) -> int:
        return self._shift_range.max_value

    # 将事件编码为索引
    def encode_event(self, event: Event) -> int:
        """将事件编码为索引。"""
        offset = 0  # 初始化偏移量
        # 遍历事件范围
        for er in self._event_ranges:
            # 检查事件类型是否匹配
            if event.type == er.type:
                # 验证事件值是否在有效范围内
                if not er.min_value <= event.value <= er.max_value:
                    raise ValueError(
                        f"Event value {event.value} is not within valid range "
                        f"[{er.min_value}, {er.max_value}] for type {event.type}"
                    )
                # 返回编码后的索引
                return offset + event.value - er.min_value
            # 更新偏移量
            offset += er.max_value - er.min_value + 1

        # 抛出未识别事件类型的错误
        raise ValueError(f"Unknown event type: {event.type}")

    # 返回事件类型的范围
    def event_type_range(self, event_type: str) -> Tuple[int, int]:
        """返回事件类型的[min_id, max_id]。"""
        offset = 0  # 初始化偏移量
        # 遍历事件范围
        for er in self._event_ranges:
            # 检查事件类型是否匹配
            if event_type == er.type:
                return offset, offset + (er.max_value - er.min_value)
            # 更新偏移量
            offset += er.max_value - er.min_value + 1

        # 抛出未识别事件类型的错误
        raise ValueError(f"Unknown event type: {event_type}")
    # 解码事件索引，返回对应的 Event 对象
    def decode_event_index(self, index: int) -> Event:
        # 初始化偏移量为 0
        offset = 0
        # 遍历事件范围列表
        for er in self._event_ranges:
            # 检查索引是否在当前事件范围内
            if offset <= index <= offset + er.max_value - er.min_value:
                # 返回相应类型和解码后的值的 Event 对象
                return Event(type=er.type, value=er.min_value + index - offset)
            # 更新偏移量以包含当前事件范围
            offset += er.max_value - er.min_value + 1
    
        # 如果没有找到匹配的事件范围，则抛出值错误
        raise ValueError(f"Unknown event index: {index}")
# 定义一个数据类 ProgramGranularity，用于描述程序粒度的相关函数
@dataclasses.dataclass
class ProgramGranularity:
    # 声明两个函数，要求它们是幂等的（同样输入返回相同输出）
    tokens_map_fn: Callable[[Sequence[int], Codec], Sequence[int]]
    program_map_fn: Callable[[int], int]


# 定义一个函数 drop_programs，用于从令牌序列中删除程序变化事件
def drop_programs(tokens, codec: Codec):
    """从令牌序列中删除程序变化事件。"""
    # 获取程序事件的最小和最大ID
    min_program_id, max_program_id = codec.event_type_range("program")
    # 返回所有不在程序ID范围内的令牌
    return tokens[(tokens < min_program_id) | (tokens > max_program_id)]


# 定义一个函数 programs_to_midi_classes，用于将程序事件修改为 MIDI 类中的第一个程序
def programs_to_midi_classes(tokens, codec):
    """将程序事件修改为 MIDI 类中的第一个程序。"""
    # 获取程序事件的最小和最大ID
    min_program_id, max_program_id = codec.event_type_range("program")
    # 确定哪些令牌是程序事件
    is_program = (tokens >= min_program_id) & (tokens <= max_program_id)
    # 根据程序事件的类别，将其映射到相应的 MIDI 类
    return np.where(is_program, min_program_id + 8 * ((tokens - min_program_id) // 8), tokens)


# 定义一个字典 PROGRAM_GRANULARITIES，用于存储不同程序粒度的映射
PROGRAM_GRANULARITIES = {
    # “平坦”粒度；删除程序变化令牌并将 NoteSequence 的程序设置为零
    "flat": ProgramGranularity(tokens_map_fn=drop_programs, program_map_fn=lambda program: 0),
    # 将每个程序映射到其 MIDI 类中的第一个程序
    "midi_class": ProgramGranularity(
        tokens_map_fn=programs_to_midi_classes, program_map_fn=lambda program: 8 * (program // 8)
    ),
    # 保留程序不变
    "full": ProgramGranularity(tokens_map_fn=lambda tokens, codec: tokens, program_map_fn=lambda program: program),
}


# 定义一个函数 frame，用于将信号分帧
def frame(signal, frame_length, frame_step, pad_end=False, pad_value=0, axis=-1):
    """
    相当于 tf.signal.frame
    """
    # 获取信号在指定轴的长度
    signal_length = signal.shape[axis]
    # 如果需要填充信号的结尾
    if pad_end:
        # 计算帧重叠的大小
        frames_overlap = frame_length - frame_step
        # 计算剩余样本数以确定需要填充的大小
        rest_samples = np.abs(signal_length - frames_overlap) % np.abs(frame_length - frames_overlap)
        pad_size = int(frame_length - rest_samples)

        # 如果需要填充
        if pad_size != 0:
            # 创建填充轴的大小列表
            pad_axis = [0] * signal.ndim
            pad_axis[axis] = pad_size
            # 在信号的指定轴上进行常量填充
            signal = F.pad(signal, pad_axis, "constant", pad_value)
    # 在指定轴上进行信号分帧
    frames = signal.unfold(axis, frame_length, frame_step)
    # 返回分帧后的信号
    return frames


# 定义一个函数 program_to_slakh_program，用于将程序转换为 Slakh 程序
def program_to_slakh_program(program):
    # 这个实现方法很黑客，可能应该使用自定义映射
    # 遍历所有 Slakh 类程序，按照从大到小的顺序
    for slakh_program in sorted(SLAKH_CLASS_PROGRAMS.values(), reverse=True):
        # 如果输入程序大于或等于 Slakh 程序，返回该 Slakh 程序
        if program >= slakh_program:
            return slakh_program


# 定义一个函数 audio_to_frames，用于将音频样本转换为非重叠的帧和帧时间
def audio_to_frames(
    samples,
    hop_size: int,
    frame_rate: int,
) -> Tuple[Sequence[Sequence[int]], torch.Tensor]:
    """将音频样本转换为非重叠的帧和帧时间。"""
    # 将帧大小设置为跳跃大小
    frame_size = hop_size
    # 填充样本以确保长度为帧大小的整数倍
    samples = np.pad(samples, [0, frame_size - len(samples) % frame_size], mode="constant")

    # 将音频分割成帧
    frames = frame(
        torch.Tensor(samples).unsqueeze(0),
        frame_length=frame_size,
        frame_step=frame_size,
        pad_end=False,  # TODO 检查为什么在这里填充为 True 时会偏差 1
    )

    # 计算帧的数量
    num_frames = len(samples) // frame_size

    # 生成每帧的时间
    times = np.arange(num_frames) / frame_rate
    # 返回分帧和对应的时间
    return frames, times
# 将音符序列转换为音符的起始和结束时间及程序信息
def note_sequence_to_onsets_and_offsets_and_programs(
    ns: note_seq.NoteSequence,
) -> Tuple[Sequence[float], Sequence[NoteEventData]]:
    """从 NoteSequence 提取起始和结束时间以及音高和程序。

    起始和结束时间不一定按顺序排列。

    参数:
      ns: 要提取起始和结束时间的 NoteSequence。

    返回:
      times: 音符的起始和结束时间列表。values: 包含音符数据的 NoteEventData 对象列表，其中音符结束的速度为零。
    """
    # 按程序和音高排序，将结束时间放在起始时间之前，以便后续稳定排序
    notes = sorted(ns.notes, key=lambda note: (note.is_drum, note.program, note.pitch))
    # 创建一个列表，包含音符的结束时间和起始时间
    times = [note.end_time for note in notes if not note.is_drum] + [note.start_time for note in notes]
    # 创建一个包含音符数据的列表，其中音符结束的速度为零
    values = [
        NoteEventData(pitch=note.pitch, velocity=0, program=note.program, is_drum=False)
        for note in notes
        if not note.is_drum
    ] + [
        NoteEventData(pitch=note.pitch, velocity=note.velocity, program=note.program, is_drum=note.is_drum)
        for note in notes
    ]
    # 返回起始和结束时间列表及音符数据列表
    return times, values


# 从事件编码器获取速度分箱的数量
def num_velocity_bins_from_codec(codec: Codec):
    """从事件编码器获取速度分箱数量。"""
    # 获取速度的最小和最大事件类型范围
    lo, hi = codec.event_type_range("velocity")
    # 计算并返回速度分箱的数量
    return hi - lo


# 将数组分段为长度为 n 的段
def segment(a, n):
    # 通过列表解析将数组分段
    return [a[i : i + n] for i in range(0, len(a), n)]


# 将速度转换为相应的分箱
def velocity_to_bin(velocity, num_velocity_bins):
    # 如果速度为零，返回第一个分箱
    if velocity == 0:
        return 0
    else:
        # 否则计算并返回相应的速度分箱
        return math.ceil(num_velocity_bins * velocity / note_seq.MAX_MIDI_VELOCITY)


# 将音符事件数据转换为事件序列
def note_event_data_to_events(
    state: Optional[NoteEncodingState],
    value: NoteEventData,
    codec: Codec,
) -> Sequence[Event]:
    """将音符事件数据转换为事件序列。"""
    # 如果速度为 None，仅返回起始事件
    if value.velocity is None:
        return [Event("pitch", value.pitch)]
    else:
        # 从编码器获取速度分箱的数量
        num_velocity_bins = num_velocity_bins_from_codec(codec)
        # 将速度转换为相应的分箱
        velocity_bin = velocity_to_bin(value.velocity, num_velocity_bins)
        if value.program is None:
            # 仅返回起始、结束和速度事件，不包含程序
            if state is not None:
                state.active_pitches[(value.pitch, 0)] = velocity_bin
            return [Event("velocity", velocity_bin), Event("pitch", value.pitch)]
        else:
            if value.is_drum:
                # 对于打击乐事件，使用单独的词汇
                return [Event("velocity", velocity_bin), Event("drum", value.pitch)]
            else:
                # 返回程序、速度和音高的事件
                if state is not None:
                    state.active_pitches[(value.pitch, value.program)] = velocity_bin
                return [
                    Event("program", value.program),
                    Event("velocity", velocity_bin),
                    Event("pitch", value.pitch),
                ]
# 将音符编码状态转换为事件序列，输出活动音符的程序和音高事件，以及最后的延续事件
def note_encoding_state_to_events(state: NoteEncodingState) -> Sequence[Event]:
    # 创建一个空列表以存储事件
    events = []
    # 遍历所有活动音高及其对应的程序，按程序排序
    for pitch, program in sorted(state.active_pitches.keys(), key=lambda k: k[::-1]):
        # 如果当前音高和程序处于活动状态
        if state.active_pitches[(pitch, program)]:
            # 将程序和音高事件添加到事件列表中
            events += [Event("program", program), Event("pitch", pitch)]
    # 在事件列表中添加一个延续事件，值为 0
    events.append(Event("tie", 0))
    # 返回生成的事件列表
    return events


# 编码并索引事件到音频帧时间
def encode_and_index_events(
    state, event_times, event_values, codec, frame_times, encode_event_fn, encoding_state_to_events_fn=None
):
    """编码一个定时事件序列并索引到音频帧时间。

    将时间偏移编码为重复的单步偏移，以便后续运行长度编码。

    可选地，还可以编码一个“状态事件”序列，跟踪每个音频帧的当前编码状态。
    这可以用于在目标段前添加表示当前状态的事件。

    参数：
      state: 初始事件编码状态。
      event_times: 事件时间序列。
      event_values: 事件值序列。
      encode_event_fn: 将事件值转换为一个或多个事件对象的函数。
      codec: 将事件对象映射到索引的 Codec 对象。
      frame_times: 每个音频帧的时间。
      encoding_state_to_events_fn: 将编码状态转换为一个或多个事件对象序列的函数。

    返回：
      events: 编码事件和偏移量。event_start_indices: 每个音频帧对应的起始事件索引。
          注意：由于采样率的差异，一个事件可能对应多个音频索引。这使得拆分序列变得棘手，因为同一个事件可能出现在一个序列的结尾和另一个序列的开头。
      event_end_indices: 每个音频帧对应的结束事件索引。用于确保切片时一个块结束于下一个块的开始。应始终满足 event_end_indices[i] = event_start_indices[i + 1]。
      state_events: 编码的“状态”事件，表示每个事件之前的编码状态。
      state_event_indices: 每个音频帧对应的状态事件索引。
    """
    # 对事件时间进行排序，返回排序后的索引
    indices = np.argsort(event_times, kind="stable")
    # 将事件时间转换为与 codec 的步进对应的步进值
    event_steps = [round(event_times[i] * codec.steps_per_second) for i in indices]
    # 根据排序的索引重新排列事件值
    event_values = [event_values[i] for i in indices]

    # 创建空列表以存储事件、状态事件和索引
    events = []
    state_events = []
    event_start_indices = []
    state_event_indices = []

    # 当前步进、事件索引和状态事件索引初始化为 0
    cur_step = 0
    cur_event_idx = 0
    cur_state_event_idx = 0

    # 定义填充当前步进对应的事件起始索引的函数
    def fill_event_start_indices_to_cur_step():
        # 当事件起始索引列表的长度小于帧时间的长度且当前帧时间小于当前步进转换的时间
        while (
            len(event_start_indices) < len(frame_times)
            and frame_times[len(event_start_indices)] < cur_step / codec.steps_per_second
        ):
            # 将当前事件索引添加到事件起始索引列表中
            event_start_indices.append(cur_event_idx)
            # 将当前状态事件索引添加到状态事件索引列表中
            state_event_indices.append(cur_state_event_idx)
    # 遍历事件步骤和事件值的组合
        for event_step, event_value in zip(event_steps, event_values):
            # 当当前步骤小于事件步骤时，持续添加“shift”事件
            while event_step > cur_step:
                events.append(codec.encode_event(Event(type="shift", value=1)))
                # 当前步骤增加
                cur_step += 1
                # 填充当前步骤的事件开始索引
                fill_event_start_indices_to_cur_step()
                # 记录当前事件的索引
                cur_event_idx = len(events)
                # 记录当前状态事件的索引
                cur_state_event_idx = len(state_events)
            # 如果有编码状态到事件的函数
            if encoding_state_to_events_fn:
                # 在处理下一个事件前，转储状态到状态事件
                for e in encoding_state_to_events_fn(state):
                    state_events.append(codec.encode_event(e))
    
            # 编码事件并添加到事件列表中
            for e in encode_event_fn(state, event_value, codec):
                events.append(codec.encode_event(e))
    
        # 在最后一个事件后，继续填充事件开始索引数组
        # 不严格的比较是因为当前步骤与音频帧的起始重合时需要额外的“shift”事件
        while cur_step / codec.steps_per_second <= frame_times[-1]:
            events.append(codec.encode_event(Event(type="shift", value=1)))
            # 当前步骤增加
            cur_step += 1
            # 填充当前步骤的事件开始索引
            fill_event_start_indices_to_cur_step()
            # 记录当前事件的索引
            cur_event_idx = len(events)
    
        # 填充事件结束索引，确保每个切片结束于下一个切片的开始
        event_end_indices = event_start_indices[1:] + [len(events)]
    
        # 将事件转换为整型数组
        events = np.array(events).astype(np.int32)
        # 将状态事件转换为整型数组
        state_events = np.array(state_events).astype(np.int32)
        # 将事件开始索引分段
        event_start_indices = segment(np.array(event_start_indices).astype(np.int32), TARGET_FEATURE_LENGTH)
        # 将事件结束索引分段
        event_end_indices = segment(np.array(event_end_indices).astype(np.int32), TARGET_FEATURE_LENGTH)
        # 将状态事件索引分段
        state_event_indices = segment(np.array(state_event_indices).astype(np.int32), TARGET_FEATURE_LENGTH)
    
        # 初始化输出列表
        outputs = []
        # 遍历每组开始索引、结束索引和状态事件索引
        for start_indices, end_indices, event_indices in zip(event_start_indices, event_end_indices, state_event_indices):
            # 将输入和索引信息添加到输出中
            outputs.append(
                {
                    "inputs": events,
                    "event_start_indices": start_indices,
                    "event_end_indices": end_indices,
                    "state_events": state_events,
                    "state_event_indices": event_indices,
                }
            )
    
        # 返回最终的输出列表
        return outputs
# 从特征中提取与音频令牌段对应的目标序列
def extract_sequence_with_indices(features, state_events_end_token=None, feature_key="inputs"):
    # 创建特征的副本以避免修改原始数据
    features = features.copy()
    # 获取事件开始索引的第一个值
    start_idx = features["event_start_indices"][0]
    # 获取事件结束索引的最后一个值
    end_idx = features["event_end_indices"][-1]
    
    # 根据开始和结束索引截取特征中的目标序列
    features[feature_key] = features[feature_key][start_idx:end_idx]

    # 如果给定了状态事件结束标记
    if state_events_end_token is not None:
        # 获取与音频开始令牌对应的状态事件开始索引
        state_event_start_idx = features["state_event_indices"][0]
        # 状态事件结束索引初始为开始索引加一
        state_event_end_idx = state_event_start_idx + 1
        # 遍历状态事件，直到遇到结束标记
        while features["state_events"][state_event_end_idx - 1] != state_events_end_token:
            state_event_end_idx += 1
        # 将状态事件与目标序列连接在一起
        features[feature_key] = np.concatenate(
            [
                features["state_events"][state_event_start_idx:state_event_end_idx],
                features[feature_key],
            ],
            axis=0,
        )

    # 返回包含处理后特征的字典
    return features


# 将 MIDI 程序映射应用于令牌序列
def map_midi_programs(
    feature, codec: Codec, granularity_type: str = "full", feature_key: str = "inputs"
) -> Mapping[str, Any]:
    # 获取与给定粒度类型对应的粒度映射
    granularity = PROGRAM_GRANULARITIES[granularity_type]

    # 使用粒度映射函数处理特征的令牌序列
    feature[feature_key] = granularity.tokens_map_fn(feature[feature_key], codec)
    # 返回处理后的特征
    return feature


# 返回一个函数，用于对给定编码器的变化进行行程编码
def run_length_encode_shifts_fn(
    features,
    codec: Codec,
    feature_key: str = "inputs",
    state_change_event_types: Sequence[str] = (),
) -> Callable[[Mapping[str, Any]], Mapping[str, Any]]:
    """返回一个函数，用于对给定编码器的变化进行行程编码。

    参数：
      codec: 用于变化事件的编码器。
      feature_key: 要进行行程编码的特征键。
      state_change_event_types: 表示状态变化的事件类型列表；
          对应这些事件类型的令牌将被解释为状态变化，冗余的将被移除。

    返回：
      一个预处理函数，用于对单步变化进行行程编码。
    """
    # 为每种状态变化事件类型获取事件类型范围
    state_change_event_ranges = [codec.event_type_range(event_type) for event_type in state_change_event_types]
    # 定义一个函数，进行运行长度编码处理
    def run_length_encode_shifts(features: MutableMapping[str, Any]) -> Mapping[str, Any]:
        """合并前导/内部移位，修剪尾部移位。
    
        Args:
          features: 要处理的特征字典。
    
        Returns:
          特征字典。
        """
        # 从特征字典中提取事件
        events = features[feature_key]
    
        # 初始化移位步骤计数器
        shift_steps = 0
        # 初始化总移位步骤计数器
        total_shift_steps = 0
        # 创建空的输出数组，用于存储结果
        output = np.array([], dtype=np.int32)
    
        # 初始化当前状态数组，大小与状态变化事件范围相同
        current_state = np.zeros(len(state_change_event_ranges), dtype=np.int32)
    
        # 遍历所有事件
        for event in events:
            # 检查当前事件是否为移位事件
            if codec.is_shift_event_index(event):
                # 增加当前移位步骤计数
                shift_steps += 1
                # 增加总移位步骤计数
                total_shift_steps += 1
    
            else:
                # 如果当前事件是状态变化且与当前状态相同，标记为冗余
                is_redundant = False
                # 遍历状态变化事件范围
                for i, (min_index, max_index) in enumerate(state_change_event_ranges):
                    # 检查当前事件是否在范围内
                    if (min_index <= event) and (event <= max_index):
                        # 如果当前状态与事件相同，标记为冗余
                        if current_state[i] == event:
                            is_redundant = True
                        # 更新当前状态
                        current_state[i] = event
                # 如果标记为冗余，跳过该事件
                if is_redundant:
                    continue
    
                # 一旦遇到非移位事件，进行之前移位事件的 RLE 编码
                if shift_steps > 0:
                    # 设置当前移位步骤为总移位步骤
                    shift_steps = total_shift_steps
                    # 处理所有移位步骤
                    while shift_steps > 0:
                        # 计算当前可输出的步骤数，不能超过最大移位步骤
                        output_steps = np.minimum(codec.max_shift_steps, shift_steps)
                        # 将当前输出步骤添加到结果数组
                        output = np.concatenate([output, [output_steps]], axis=0)
                        # 减少剩余移位步骤
                        shift_steps -= output_steps
                    # 添加当前事件到输出数组
                    output = np.concatenate([output, [event]], axis=0)
    
        # 将处理后的输出数组存回特征字典
        features[feature_key] = output
        # 返回更新后的特征字典
        return features
    
    # 调用函数并返回结果
    return run_length_encode_shifts(features)
# 定义一个处理音符表示的处理链函数，接受特征、编解码器和音符表示配置
def note_representation_processor_chain(features, codec: Codec, note_representation_config: NoteRepresentationConfig):
    # 编码一个 "tie" 事件，值为 0
    tie_token = codec.encode_event(Event("tie", 0))
    # 如果配置包含连音符，则将其作为状态事件结束的标记，否则为 None
    state_events_end_token = tie_token if note_representation_config.include_ties else None

    # 从特征中提取序列，指定状态事件结束标记和特征键
    features = extract_sequence_with_indices(
        features, state_events_end_token=state_events_end_token, feature_key="inputs"
    )

    # 映射 MIDI 程序到特征
    features = map_midi_programs(features, codec)

    # 对特征进行游程编码，指定状态变化事件类型
    features = run_length_encode_shifts_fn(features, codec, state_change_event_types=["velocity", "program"])

    # 返回处理后的特征
    return features


# 定义一个 MIDI 处理类
class MidiProcessor:
    # 初始化 MIDI 处理器
    def __init__(self):
        # 创建一个编解码器，设置最大移动步数、每秒步数和事件范围
        self.codec = Codec(
            max_shift_steps=DEFAULT_MAX_SHIFT_SECONDS * DEFAULT_STEPS_PER_SECOND,
            steps_per_second=DEFAULT_STEPS_PER_SECOND,
            event_ranges=[
                EventRange("pitch", note_seq.MIN_MIDI_PITCH, note_seq.MAX_MIDI_PITCH),
                EventRange("velocity", 0, DEFAULT_NUM_VELOCITY_BINS),
                EventRange("tie", 0, 0),
                EventRange("program", note_seq.MIN_MIDI_PROGRAM, note_seq.MAX_MIDI_PROGRAM),
                EventRange("drum", note_seq.MIN_MIDI_PITCH, note_seq.MAX_MIDI_PITCH),
            ],
        )
        # 创建一个标记器，传入编解码器的类别数量
        self.tokenizer = Tokenizer(self.codec.num_classes)
        # 配置音符表示，设置为仅包含起始音符并包括连音符
        self.note_representation_config = NoteRepresentationConfig(onsets_only=False, include_ties=True)

    # 定义调用函数，接受 MIDI 数据
    def __call__(self, midi: Union[bytes, os.PathLike, str]):
        # 检查输入是否为字节，如果不是则读取文件内容
        if not isinstance(midi, bytes):
            with open(midi, "rb") as f:
                midi = f.read()

        # 将 MIDI 数据转换为音符序列
        ns = note_seq.midi_to_note_sequence(midi)
        # 应用延音控制变化到音符序列
        ns_sus = note_seq.apply_sustain_control_changes(ns)

        # 遍历音符序列中的音符
        for note in ns_sus.notes:
            # 如果音符不是打击乐器，则将程序号转换为相应的程序
            if not note.is_drum:
                note.program = program_to_slakh_program(note.program)

        # 创建一个零数组以存储样本，长度为总时间乘以采样率
        samples = np.zeros(int(ns_sus.total_time * SAMPLE_RATE))

        # 将音频样本转换为帧，获取帧时间
        _, frame_times = audio_to_frames(samples, HOP_SIZE, FRAME_RATE)
        # 从音符序列提取开始和结束时间及程序号
        times, values = note_sequence_to_onsets_and_offsets_and_programs(ns_sus)

        # 编码和索引事件
        events = encode_and_index_events(
            state=NoteEncodingState(),
            event_times=times,
            event_values=values,
            frame_times=frame_times,
            codec=self.codec,
            encode_event_fn=note_event_data_to_events,
            encoding_state_to_events_fn=note_encoding_state_to_events,
        )

        # 对每个事件应用音符表示处理链
        events = [
            note_representation_processor_chain(event, self.codec, self.note_representation_config) for event in events
        ]
        # 对处理后的事件进行编码，生成输入标记
        input_tokens = [self.tokenizer.encode(event["inputs"]) for event in events]

        # 返回输入标记
        return input_tokens
```