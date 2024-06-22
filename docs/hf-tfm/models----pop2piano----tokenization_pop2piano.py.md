# `.\transformers\models\pop2piano\tokenization_pop2piano.py`

```py
# 设置代码文件的编码格式为 UTF-8

# 版权声明，版权归 The Pop2Piano Authors 和 The HuggingFace Inc. team 所有
# 根据 Apache 许可证 2.0 版本进行许可
# 除非符合许可证要求或经书面同意，否则不得使用此文件
# 可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 根据许可证要求，在"原样"的基础上分发软件，不提供任何担保或条件，无论是明示的还是暗示的 
# 详细了解特定语言对权限的限制和限制条件，请参见许可证
"""Pop2Piano 的分词类"""

# 导入各种模块
import json
import os
from typing import List, Optional, Tuple, Union
import numpy as np
from ...feature_extraction_utils import BatchFeature
from ...tokenization_utils import AddedToken, BatchEncoding, PaddingStrategy, PreTrainedTokenizer, TruncationStrategy
from ...utils import TensorType, is_pretty_midi_available, logging, requires_backends, to_numpy

# 如果 pretty_midi 模块可用，则导入
if is_pretty_midi_available():
    import pretty_midi

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 设置文件名常量 
VOCAB_FILES_NAMES = {
    "vocab": "vocab.json",
}

# 预训练模型文件名映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab": {
        "sweetcocoa/pop2piano": "https://huggingface.co/sweetcocoa/pop2piano/blob/main/vocab.json",
    },
}

# 定义一个函数，将时间标记转换为音符标记
def token_time_to_note(number, cutoff_time_idx, current_idx):
    current_idx += number
    # 如果指定了截止时间索引，则将当前索引与之进行比较
    if cutoff_time_idx is not None:
        current_idx = min(current_idx, cutoff_time_idx)
    return current_idx

# 定义一个函数，将音符标记转换为音符对象
def token_note_to_note(number, current_velocity, default_velocity, note_onsets_ready, current_idx, notes):
    if note_onsets_ready[number] is not None:
        # 带有起始时间的偏移
        onset_idx = note_onsets_ready[number]
        if onset_idx < current_idx:
            # 在前一个音符开始后进行时间转移
            offset_idx = current_idx
            notes.append([onset_idx, offset_idx, number, default_velocity])
            onsets_ready = None if current_velocity == 0 else current_idx
            note_onsets_ready[number] = onsets_ready
    else:
        note_onsets_ready[number] = current_idx
    return notes

# 定义 Pop2PianoTokenizer 类，继承自 PreTrainedTokenizer
class Pop2PianoTokenizer(PreTrainedTokenizer):
    """
    构造一个 Pop2Piano 分词器。此分词器不需要训练。

    这个分词器继承自[`PreTrainedTokenizer`]，其中包含大多数主要方法。用户应参考
    这个超类以获取有关这些方法���更多信息。

    Args:
        vocab (`str`):
            包含词汇表的词汇文件的路径。
        default_velocity (`int`, *optional*, defaults to 77):
            确定在创建 MIDI 音符时要使用的默认速度。
        num_bars (`int`, *optional*, defaults to 2):
            确定每个标记的截止时间索引。
    """

    model_input_names = ["token_ids", "attention_mask"]
    vocab_files_names = VOCAB_FILES_NAMES
    # 加载预训练词汇文件映射
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP

    # 初始化方法
    def __init__(
        self,
        vocab,
        default_velocity=77,
        num_bars=2,
        unk_token="-1",
        eos_token="1",
        pad_token="0",
        bos_token="2",
        **kwargs,
    ):
        # 根据传入的未知标记初始化 AddedToken 对象，确保左右两边不会被删除
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        # 根据传入的结束标记初始化 AddedToken 对象，确保左右两边不会被删除
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        # 根据传入的填充标记初始化 AddedToken 对象，确保左右两边不会被删除
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token
        # 根据传入的起始标记初始化 AddedToken 对象，确保左右两边不会被删除
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token

        # 设置默认速度和小节数
        self.default_velocity = default_velocity
        self.num_bars = num_bars

        # 加载词汇表
        with open(vocab, "rb") as file:
            self.encoder = json.load(file)

        # 创建编码器的反向映射
        self.decoder = {v: k for k, v in self.encoder.items()}

        # 调用父类的初始化方法
        super().__init__(
            unk_token=unk_token,
            eos_token=eos_token,
            pad_token=pad_token,
            bos_token=bos_token,
            **kwargs,
        )

    @property
    def vocab_size(self):
        """返回标记器的词汇量大小。"""
        return len(self.encoder)

    def get_vocab(self):
        """返回标记器的词汇表。"""
        return dict(self.encoder, **self.added_tokens_encoder)

    def _convert_id_to_token(self, token_id: int) -> list:
        """
        将由转换器生成的标记 ID 解码为音符。

        Args:
            token_id (`int`):
                表示由转换器生成要转换为 Midi 标记的 ID。

        Returns:
            `List`: 包含标记类型 (`str`) 和值 (`int`) 的列表。
        """

        # 获取标记 ID 对应的类型和值
        token_type_value = self.decoder.get(token_id, f"{self.unk_token}_TOKEN_TIME")
        # 分割类型和值
        token_type_value = token_type_value.split("_")
        # 提取类型和值
        token_type, value = "_".join(token_type_value[1:]), int(token_type_value[0])

        return [token_type, value]

    def _convert_token_to_id(self, token, token_type="TOKEN_TIME") -> int:
        """
        将 Midi 标记编码为转换器生成的标记 ID。

        Args:
            token (`int`):
                表示标记值。
            token_type (`str`):
                表示标记的类型。有四种类型的 Midi 标记，如 "TOKEN_TIME"、"TOKEN_VELOCITY"、"TOKEN_NOTE" 和 "TOKEN_SPECIAL"。

        Returns:
            `int`: 返回标记的 ID。
        """
        # 获取标记和类型对应的 ID
        return self.encoder.get(f"{token}_{token_type}", int(self.unk_token))

    def relative_batch_tokens_ids_to_notes(
        self,
        tokens: np.ndarray,
        beat_offset_idx: int,
        bars_per_batch: int,
        cutoff_time_idx: int,
    ):
        """
        Converts relative tokens to notes which are then used to generate pretty midi object.

        Args:
            tokens (`numpy.ndarray`):
                Tokens to be converted to notes.
            beat_offset_idx (`int`):
                Denotes beat offset index for each note in generated Midi.
            bars_per_batch (`int`):
                A parameter to control the Midi output generation.
            cutoff_time_idx (`int`):
                Denotes the cutoff time index for each note in generated Midi.
        """

        notes = None

        for index in range(len(tokens)):
            _tokens = tokens[index]
            _start_idx = beat_offset_idx + index * bars_per_batch * 4
            _cutoff_time_idx = cutoff_time_idx + _start_idx
            _notes = self.relative_tokens_ids_to_notes(
                _tokens,
                start_idx=_start_idx,
                cutoff_time_idx=_cutoff_time_idx,
            )

            if len(_notes) == 0:
                pass
            elif notes is None:
                notes = _notes
            else:
                notes = np.concatenate((notes, _notes), axis=0)

        if notes is None:
            return []
        return notes

    def relative_batch_tokens_ids_to_midi(
        self,
        tokens: np.ndarray,
        beatstep: np.ndarray,
        beat_offset_idx: int = 0,
        bars_per_batch: int = 2,
        cutoff_time_idx: int = 12,
    ):
        """
        Converts tokens to Midi. This method calls `relative_batch_tokens_ids_to_notes` method to convert batch tokens
        to notes then uses `notes_to_midi` method to convert them to Midi.

        Args:
            tokens (`numpy.ndarray`):
                Denotes tokens which alongside beatstep will be converted to Midi.
            beatstep (`np.ndarray`):
                We get beatstep from feature extractor which is also used to get Midi.
            beat_offset_idx (`int`, *optional*, defaults to 0):
                Denotes beat offset index for each note in generated Midi.
            bars_per_batch (`int`, *optional*, defaults to 2):
                A parameter to control the Midi output generation.
            cutoff_time_idx (`int`, *optional*, defaults to 12):
                Denotes the cutoff time index for each note in generated Midi.
        """
        beat_offset_idx = 0 if beat_offset_idx is None else beat_offset_idx
        notes = self.relative_batch_tokens_ids_to_notes(
            tokens=tokens,
            beat_offset_idx=beat_offset_idx,
            bars_per_batch=bars_per_batch,
            cutoff_time_idx=cutoff_time_idx,
        )
        midi = self.notes_to_midi(notes, beatstep, offset_sec=beatstep[beat_offset_idx])
        return midi

    # Taken from the original code
    # Please see https://github.com/sweetcocoa/pop2piano/blob/fac11e8dcfc73487513f4588e8d0c22a22f2fdc5/midi_tokenizer.py#L257
    # 将相对 tokens 转换为 notes，以便用于创建 Pretty Midi 对象
    def relative_tokens_ids_to_notes(self, tokens: np.ndarray, start_idx: float, cutoff_time_idx: float = None):
        """
        Converts relative tokens to notes which will then be used to create Pretty Midi objects.

        Args:
            tokens (`numpy.ndarray`):
                Relative Tokens which will be converted to notes.
            start_idx (`float`):
                A parameter which denotes the starting index.
            cutoff_time_idx (`float`, *optional*):
                A parameter used while converting tokens to notes.
        """
        # 将 tokens 转换为对应的单词列表
        words = [self._convert_id_to_token(token) for token in tokens]

        # 初始化当前索引和当前速度
        current_idx = start_idx
        current_velocity = 0
        # 初始化用于记录音符开始的列表
        note_onsets_ready = [None for i in range(sum([k.endswith("NOTE") for k in self.encoder.keys()]) + 1)]
        notes = []
        # 遍历单词列表
        for token_type, number in words:
            if token_type == "TOKEN_SPECIAL":
                if number == 1:
                    break
            elif token_type == "TOKEN_TIME":
                # 根据时间 token 转换为音符
                current_idx = token_time_to_note(
                    number=number, cutoff_time_idx=cutoff_time_idx, current_idx=current_idx
                )
            elif token_type == "TOKEN_VELOCITY":
                current_velocity = number
            elif token_type == "TOKEN_NOTE":
                # 根据音符 token 转换为音符
                notes = token_note_to_note(
                    number=number,
                    current_velocity=current_velocity,
                    default_velocity=self.default_velocity,
                    note_onsets_ready=note_onsets_ready,
                    current_idx=current_idx,
                    notes=notes,
                )
            else:
                raise ValueError("Token type not understood!")

        # 对于每个音高，强制偏移（offset）如果没有偏移
        for pitch, note_onset in enumerate(note_onsets_ready):
            if note_onset is not None:
                if cutoff_time_idx is None:
                    cutoff = note_onset + 1
                else:
                    cutoff = max(cutoff_time_idx, note_onset + 1)

                offset_idx = max(current_idx, cutoff)
                notes.append([note_onset, offset_idx, pitch, self.default_velocity])

        # 如果 notes 长度为 0，则返回空列表，否则对 notes 进行排序后返回
        if len(notes) == 0:
            return []
        else:
            notes = np.array(notes)
            note_order = notes[:, 0] * 128 + notes[:, 1]
            notes = notes[note_order.argsort()]
            return notes
    def notes_to_midi(self, notes: np.ndarray, beatstep: np.ndarray, offset_sec: int = 0.0):
        """
        Converts notes to Midi.

        Args:
            notes (`numpy.ndarray`):
                This is used to create Pretty Midi objects.
            beatstep (`numpy.ndarray`):
                This is the extrapolated beatstep that we get from feature extractor.
            offset_sec (`int`, *optional*, defaults to 0.0):
                This represents the offset seconds which is used while creating each Pretty Midi Note.
        """

        # 确保依赖库已安装
        requires_backends(self, ["pretty_midi"])

        # 创建一个新的 PrettyMIDI 对象，设置分辨率和初始速度
        new_pm = pretty_midi.PrettyMIDI(resolution=384, initial_tempo=120.0)
        # 创建一个新的 Instrument 对象，默认程序为 0
        new_inst = pretty_midi.Instrument(program=0)
        # 创建一个空的 Note 列表
        new_notes = []

        # 遍历输入的音符
        for onset_idx, offset_idx, pitch, velocity in notes:
            # 创建一个新的 Note 对象，设置音高、起始时间和结束时间
            new_note = pretty_midi.Note(
                velocity=velocity,
                pitch=pitch,
                start=beatstep[onset_idx] - offset_sec,
                end=beatstep[offset_idx] - offset_sec,
            )
            # 将新的 Note 对象添加到 Note 列表中
            new_notes.append(new_note)
        
        # 将 Note 列表赋值给新的 Instrument 对象
        new_inst.notes = new_notes
        # 将新的 Instrument 对象添加到新的 PrettyMIDI 对象中
        new_pm.instruments.append(new_inst)
        # 移除无效的音符
        new_pm.remove_invalid_notes()
        # 返回新的 PrettyMIDI 对象
        return new_pm

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Saves the tokenizer's vocabulary dictionary to the provided save_directory.

        Args:
            save_directory (`str`):
                A path to the directory where to saved. It will be created if it doesn't exist.
            filename_prefix (`Optional[str]`, *optional*):
                A prefix to add to the names of the files saved by the tokenizer.
        """
        # 检查保存目录是否存在
        if not os.path.isdir(save_directory):
            # 如果不存在，则记录错误并返回
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return

        # 保存编码器的词汇表字典到指定目录下
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab"]
        )
        with open(out_vocab_file, "w") as file:
            file.write(json.dumps(self.encoder))

        # 返回保存的文件路径的元组
        return (out_vocab_file,)

    def encode_plus(
        self,
        notes: Union[np.ndarray, List[pretty_midi.Note]],
        truncation_strategy: Optional[TruncationStrategy] = None,
        max_length: Optional[int] = None,
        **kwargs,
    ):
        """
        Encodes a single set of notes into a dictionary containing the input_ids, attention_mask, and token_type_ids.

        Args:
            notes (Union[np.ndarray, List[pretty_midi.Note]]):
                The input notes to be encoded.
            truncation_strategy (Optional[TruncationStrategy], *optional*):
                The truncation strategy to apply. Defaults to None.
            max_length (Optional[int], *optional*):
                The maximum length of the sequence. Defaults to None.
            **kwargs:
                Additional keyword arguments passed to the tokenizer.
        """
        # 编码单个音符集合为包含 input_ids、attention_mask 和 token_type_ids 的字典
        # 省略部分代码

    def batch_encode_plus(
        self,
        notes: Union[np.ndarray, List[pretty_midi.Note]],
        truncation_strategy: Optional[TruncationStrategy] = None,
        max_length: Optional[int] = None,
        **kwargs,
    ):
        """
        Encodes a batch of notes into a list of dictionaries, each containing the input_ids, attention_mask, and token_type_ids.

        Args:
            notes (Union[np.ndarray, List[pretty_midi.Note]]):
                The batch of notes to be encoded.
            truncation_strategy (Optional[TruncationStrategy], *optional*):
                The truncation strategy to apply. Defaults to None.
            max_length (Optional[int], *optional*):
                The maximum length of the sequences. Defaults to None.
            **kwargs:
                Additional keyword arguments passed to the tokenizer.
        """
        # 编码批量音符集合为包含 input_ids、attention_mask 和 token_type_ids 的字典列表
        # 省略部分代码
    ) -> BatchEncoding:
        r"""
        This is the `batch_encode_plus` method for `Pop2PianoTokenizer`. It converts the midi notes to the transformer
        generated token ids. It works on multiple batches by calling `encode_plus` multiple times in a loop.

        Args:
            notes (`numpy.ndarray` of shape `[batch_size, sequence_length, 4]` or `list` of `pretty_midi.Note` objects):
                This represents the midi notes. If `notes` is a `numpy.ndarray`:
                    - Each sequence must have 4 values, they are `onset idx`, `offset idx`, `pitch` and `velocity`.
                If `notes` is a `list` containing `pretty_midi.Note` objects:
                    - Each sequence must have 4 attributes, they are `start`, `end`, `pitch` and `velocity`.
            truncation_strategy ([`~tokenization_utils_base.TruncationStrategy`], *optional*):
                Indicates the truncation strategy that is going to be used during truncation.
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).

        Returns:
            `BatchEncoding` containing the tokens ids.
        """

        encoded_batch_token_ids = []
        for i in range(len(notes)):
            encoded_batch_token_ids.append(
                self.encode_plus(
                    notes[i],
                    truncation_strategy=truncation_strategy,
                    max_length=max_length,
                    **kwargs,
                )["token_ids"]
            )

        return BatchEncoding({"token_ids": encoded_batch_token_ids})

    def __call__(
        self,
        notes: Union[
            np.ndarray,
            List[pretty_midi.Note],
            List[List[pretty_midi.Note]],
        ],
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        verbose: bool = True,
        **kwargs,
    ):
        """
        This method is used to tokenizes a set of midi notes into token ids.

        Args:
            notes (Union[np.ndarray, List[pretty_midi.Note], List[List[pretty_midi.Note]]]):
                The midi notes to be tokenized. It can be either a numpy array of shape `[batch_size, sequence_length, 4]`
                or a list of pretty_midi.Note objects.
            padding (Union[bool, str, PaddingStrategy], optional):
                Controls padding. See `BatchEncoding` for more options.
            truncation (Union[bool, str, TruncationStrategy], optional):
                Controls truncation. See `BatchEncoding` for more options.
            max_length (int, optional):
                Maximum length of the returned list and optionally padding length.
            pad_to_multiple_of (int, optional):
                Pad to a multiple of this length if provided and padding is turned on.
            return_attention_mask (bool, optional):
                Whether to return the attention mask.
            return_tensors (Optional[Union[str, TensorType]], optional):
                The type of tensors to return. See `BatchEncoding` for more options.
            verbose (bool, optional):
                Whether to print status information during tokenization.
            **kwargs: Additional keyword arguments passed to `encode_plus`.

        Returns:
            BatchEncoding: A `BatchEncoding` containing the token ids.
        """

    def batch_decode(
        self,
        token_ids,
        feature_extractor_output: BatchFeature,
        return_midi: bool = True,
    ):
        """
        Decodes a batch of token ids into midi notes.

        Args:
            token_ids (List[List[int]]): The token ids to decode.
            feature_extractor_output (BatchFeature): The feature extractor output.
            return_midi (bool, optional): Whether to return midi notes.

        Returns:
            Union[List[pretty_midi.Note], List[List[pretty_midi.Note]]]: The decoded midi notes.
        """
```