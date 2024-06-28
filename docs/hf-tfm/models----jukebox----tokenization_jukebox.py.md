# `.\models\jukebox\tokenization_jukebox.py`

```py
# 引入所需的库和模块
import json  # 导入处理 JSON 格式的模块
import os    # 导入操作系统相关功能的模块
import re    # 导入正则表达式模块
import unicodedata  # 导入 Unicode 数据库模块
from json.encoder import INFINITY  # 从 JSON 库中导入 INFINITY 常量
from typing import Any, Dict, List, Optional, Tuple, Union  # 导入类型提示相关的功能

import numpy as np  # 导入 NumPy 库，用于数值计算
import regex       # 导入 regex 库，支持更强大的正则表达式功能

# 从 tokenization_utils 模块导入 AddedToken 和 PreTrainedTokenizer 类
from ...tokenization_utils import AddedToken, PreTrainedTokenizer
# 从 tokenization_utils_base 模块导入 BatchEncoding 类
from ...tokenization_utils_base import BatchEncoding
# 从 utils 模块导入 TensorType, is_flax_available, is_tf_available, is_torch_available, logging 等功能
from ...utils import TensorType, is_flax_available, is_tf_available, is_torch_available, logging
# 从 utils.generic 模块导入 _is_jax 和 _is_numpy 函数
from ...utils.generic import _is_jax, _is_numpy

# 获取 logger 对象，用于记录日志
logger = logging.get_logger(__name__)

# 定义各种文件名与其对应的词汇表文件名
VOCAB_FILES_NAMES = {
    "artists_file": "artists.json",   # 艺术家信息的 JSON 文件名
    "lyrics_file": "lyrics.json",     # 歌词信息的 JSON 文件名
    "genres_file": "genres.json",     # 音乐流派信息的 JSON 文件名
}

# 预训练词汇文件映射表
PRETRAINED_VOCAB_FILES_MAP = {
    "artists_file": {
        "jukebox": "https://huggingface.co/ArthurZ/jukebox/blob/main/artists.json",  # 艺术家信息的预训练 URL
    },
    "genres_file": {
        "jukebox": "https://huggingface.co/ArthurZ/jukebox/blob/main/genres.json",   # 音乐流派信息的预训练 URL
    },
    "lyrics_file": {
        "jukebox": "https://huggingface.co/ArthurZ/jukebox/blob/main/lyrics.json",   # 歌词信息的预训练 URL
    },
}

# 预训练歌词 token 大小
PRETRAINED_LYRIC_TOKENS_SIZES = {
    "jukebox": 512,   # Jukebox 模型的歌词 token 大小为 512
}

# JukeboxTokenizer 类，继承自 PreTrainedTokenizer
class JukeboxTokenizer(PreTrainedTokenizer):
    """
    构造 Jukebox 分词器。Jukebox 可以根据三种不同的输入条件进行条件化：
        - 艺术家：每个艺术家关联的唯一 ID 存储在提供的字典中。
        - 音乐流派：每种流派关联的唯一 ID 存储在提供的字典中。
        - 歌词：基于字符的分词。必须初始化使用词汇表中包含的字符列表。

    该分词器不需要训练。它应该能够处理不同数量的输入：
    因为模型的条件化可以在三种不同的查询上完成。如果未提供任何值，则将使用默认值。

    根据应该条件化模型的流派数量（`n_genres`）而定。

    参数：
        - PreTrainedTokenizer：继承自父类 PreTrainedTokenizer 的构造函数。

    示例用法：
    ```
    >>> from transformers import JukeboxTokenizer

    >>> tokenizer = JukeboxTokenizer.from_pretrained("openai/jukebox-1b-lyrics")
    >>> tokenizer("Alan Jackson", "Country Rock", "old town road")["input_ids"]
    [tensor([[   0,    0,    0, 6785,  546,   41,   38,   30,   76,   46,   41,   49,
               40,   76,   44,   41,   27,   30]]), tensor([[  0,   0,   0, 145,   0]]), tensor([[  0,   0,   0, 145,   0]])]
    ```
    ```
    # 你可以通过在实例化这个分词器时或在调用它处理文本时传递 `add_prefix_space=True` 来避免这种行为，但由于模型不是以这种方式预训练的，可能会导致性能下降。
    
    # 提示信息

    # 如果未提供任何内容，流派和艺术家将随机选择或设置为 None。

    # 这个分词器继承自 [`PreTrainedTokenizer`]，其中包含大多数主要方法。用户应参考该超类以获取有关这些方法的更多信息。

    # 然而，代码不允许这样做，只支持从各种流派组成。

    # 参数说明:
    # artists_file (`str`):
    #     包含艺术家与其ID映射的词汇文件的路径。默认文件支持 "v2" 和 "v3"。
    # genres_file (`str`):
    #     包含流派与其ID映射的词汇文件的路径。
    # lyrics_file (`str`):
    #     包含歌词分词接受字符的词汇文件的路径。
    # version (`List[str]`, 可选, 默认为 `["v3", "v2", "v2"]`) :
    #     分词器版本列表。`5b-lyrics` 的顶级优先模型使用 `v3` 而不是 `v2` 进行训练。
    # n_genres (`int`, 可选, 默认为 5):
    #     用于组合的最大流派数。
    # max_n_lyric_tokens (`int`, 可选, 默认为 512):
    #     保留的最大歌词分词数量。
    # unk_token (`str`, 可选, 默认为 `"<|endoftext|>"`):
    #     未知标记。词汇表中没有的标记将无法转换为ID，并被设置为此标记。
    """

    # 定义类级别的属性

    # 词汇文件名列表
    vocab_files_names = VOCAB_FILES_NAMES
    # 预训练词汇文件映射
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 预训练歌词分词器的最大输入尺寸
    max_lyric_input_size = PRETRAINED_LYRIC_TOKENS_SIZES
    # 模型的输入名称列表
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        artists_file,
        genres_file,
        lyrics_file,
        version=["v3", "v2", "v2"],
        max_n_lyric_tokens=512,
        n_genres=5,
        unk_token="<|endoftext|>",
        **kwargs,
    ):
    ):
        # 如果 unk_token 是字符串，则创建一个 AddedToken 对象，保留字符串两侧空白字符
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        # 设置模型版本号
        self.version = version
        # 设置歌词最大 token 数量
        self.max_n_lyric_tokens = max_n_lyric_tokens
        # 设置流派数量
        self.n_genres = n_genres
        # 初始化未知 token 的解码器
        self._added_tokens_decoder = {0: unk_token}

        # 读取并加载艺术家编码器（JSON 格式）
        with open(artists_file, encoding="utf-8") as vocab_handle:
            self.artists_encoder = json.load(vocab_handle)

        # 读取并加载流派编码器（JSON 格式）
        with open(genres_file, encoding="utf-8") as vocab_handle:
            self.genres_encoder = json.load(vocab_handle)

        # 读取并加载歌词编码器（JSON 格式）
        with open(lyrics_file, encoding="utf-8") as vocab_handle:
            self.lyrics_encoder = json.load(vocab_handle)

        # 正则表达式模式，用于识别词汇表中的未知字符
        oov = r"[^A-Za-z0-9.,:;!?\-'\"()\[\] \t\n]+"
        # 在 v2 版本中，我们的 n_vocab=80，但在 v3 中我们遗漏了 +，所以现在 n_vocab=79 个字符。
        # 如果歌词编码器长度为 79，则更新正则表达式以包括额外的字符 '-'
        if len(self.lyrics_encoder) == 79:
            oov = oov.replace(r"\-'", r"\-+'")

        # 编译正则表达式模式，用于匹配词汇表中的未知字符
        self.out_of_vocab = regex.compile(oov)
        # 创建艺术家的解码器，将编码器的键值对反转
        self.artists_decoder = {v: k for k, v in self.artists_encoder.items()}
        # 创建流派的解码器，将编码器的键值对反转
        self.genres_decoder = {v: k for k, v in self.genres_encoder.items()}
        # 创建歌词的解码器，将编码器的键值对反转
        self.lyrics_decoder = {v: k for k, v in self.lyrics_encoder.items()}
        # 调用父类的初始化方法，传递参数给父类
        super().__init__(
            unk_token=unk_token,
            n_genres=n_genres,
            version=version,
            max_n_lyric_tokens=max_n_lyric_tokens,
            **kwargs,
        )

    @property
    def vocab_size(self):
        # 返回总的词汇量大小，包括艺术家、流派和歌词的编码器的长度之和
        return len(self.artists_encoder) + len(self.genres_encoder) + len(self.lyrics_encoder)

    def get_vocab(self):
        # 返回包含艺术家、流派和歌词编码器的字典
        return {
            "artists_encoder": self.artists_encoder,
            "genres_encoder": self.genres_encoder,
            "lyrics_encoder": self.lyrics_encoder,
        }

    def _convert_token_to_id(self, list_artists, list_genres, list_lyrics):
        """Converts the artist, genre and lyrics tokens to their index using the vocabulary.
        The total_length, offset and duration have to be provided in order to select relevant lyrics and add padding to
        the lyrics token sequence.
        """
        # 将艺术家标签转换为它们在编码器中的索引
        artists_id = [self.artists_encoder.get(artist, 0) for artist in list_artists]
        # 将流派标签转换为它们在编码器中的索引，并在需要时添加填充标记
        for genres in range(len(list_genres)):
            list_genres[genres] = [self.genres_encoder.get(genre, 0) for genre in list_genres[genres]]
            list_genres[genres] = list_genres[genres] + [-1] * (self.n_genres - len(list_genres[genres]))

        # 将歌词字符转换为它们在编码器中的索引，每个歌词位置（如 total_length、offset、duration）提供相应的歌词
        lyric_ids = [[self.lyrics_encoder.get(character, 0) for character in list_lyrics[0]], [], []]
        return artists_id, list_genres, lyric_ids
    # 将字符串 lyrics 转换为标记序列（字符串），使用指定的标记器。
    # 如果是基于词汇的，按单词拆分；如果是基于子词（如BPE/SentencePieces/WordPieces），则按子词拆分。
    # 对于基于字符的词汇表，仅将歌词拆分成字符。
    def _tokenize(self, lyrics):
        """
        Converts a string into a sequence of tokens (string), using the tokenizer. Split in words for word-based
        vocabulary or sub-words for sub-word-based vocabularies (BPE/SentencePieces/WordPieces).

        Do NOT take care of added tokens. Only the lyrics are split into character for the character-based vocabulary.
        """
        # 仅对歌词进行拆分，如果是基于字符的词汇表，这很容易处理
        return list(lyrics)

    # 使用标记器将艺术家、流派和歌词转换为标记序列的三元组
    def tokenize(self, artist, genre, lyrics, **kwargs):
        """
        Converts three strings in a 3 sequence of tokens using the tokenizer
        """
        # 准备艺术家、流派和歌词以进行标记化
        artist, genre, lyrics = self.prepare_for_tokenization(artist, genre, lyrics)
        # 将歌词转换为标记序列
        lyrics = self._tokenize(lyrics)
        return artist, genre, lyrics

    # 准备艺术家、流派和歌词以进行标记化
    def prepare_for_tokenization(
        self, artists: str, genres: str, lyrics: str, is_split_into_words: bool = False
    ):
    ) -> Tuple[str, str, str, Dict[str, Any]]:
        """
        Performs any necessary transformations before tokenization.

        Args:
            artist (`str`):
                The artist name to prepare. This will mostly lower the string
            genres (`str`):
                The genre name to prepare. This will mostly lower the string.
            lyrics (`str`):
                The lyrics to prepare.
            is_split_into_words (`bool`, *optional*, defaults to `False`):
                Whether or not the input is already pre-tokenized (e.g., split into words). If set to `True`, the
                tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
                which it will tokenize. This is useful for NER or token classification.
        """
        # 循环遍历版本列表，进行必要的转换操作
        for idx in range(len(self.version)):
            # 如果版本为 "v3"，将艺术家和流派名称转换为小写
            if self.version[idx] == "v3":
                artists[idx] = artists[idx].lower()
                genres[idx] = [genres[idx].lower()]
            else:
                # 如果版本不为 "v3"，对艺术家名称进行标准化处理并添加后缀 ".v2"，对流派名称进行拆分并添加后缀 ".v2"
                artists[idx] = self._normalize(artists[idx]) + ".v2"
                genres[idx] = [
                    self._normalize(genre) + ".v2" for genre in genres[idx].split("_")
                ]  # split is for the full dictionary with combined genres

        # 如果版本为 "v2"，设置处理非词汇表外字符的正则表达式和词汇表
        if self.version[0] == "v2":
            self.out_of_vocab = regex.compile(r"[^A-Za-z0-9.,:;!?\-'\"()\[\] \t\n]+")
            vocab = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,:;!?-+'\"()[] \t\n"
            # 创建词汇表和词汇表的索引
            self.vocab = {vocab[index]: index + 1 for index in range(len(vocab))}
            self.vocab["<unk>"] = 0
            self.n_vocab = len(vocab) + 1
            self.lyrics_encoder = self.vocab
            self.lyrics_decoder = {v: k for k, v in self.vocab.items()}
            self.lyrics_decoder[0] = ""
        else:
            # 如果版本不为 "v2"，设置处理非词汇表外字符的正则表达式
            self.out_of_vocab = regex.compile(r"[^A-Za-z0-9.,:;!?\-+'\"()\[\] \t\n]+")

        # 运行去除文本中重音符号的函数
        lyrics = self._run_strip_accents(lyrics)
        # 替换文本中的 "\\" 为换行符 "\n"
        lyrics = lyrics.replace("\\", "\n")
        # 使用正则表达式去除文本中的非词汇表外字符，并初始化两个空列表
        lyrics = self.out_of_vocab.sub("", lyrics), [], []
        # 返回处理后的艺术家名称、流派名称和歌词
        return artists, genres, lyrics

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        # 使用 unicodedata 库规范化文本中的 Unicode 字符，去除重音符号
        text = unicodedata.normalize("NFD", text)
        output = []
        # 遍历文本中的每个字符
        for char in text:
            # 获取字符的 Unicode 分类
            cat = unicodedata.category(char)
            # 如果字符的分类为 "Mn"（非重音符号），跳过该字符
            if cat == "Mn":
                continue
            # 将符合条件的字符添加到输出列表中
            output.append(char)
        # 将输出列表中的字符连接成字符串并返回
        return "".join(output)
    # 定义一个方法，用于规范化输入的文本。这个过程适用于音乐流派和艺术家名称。

    def _normalize(self, text: str) -> str:
        """
        Normalizes the input text. This process is for the genres and the artist

        Args:
            text (`str`):
                Artist or Genre string to normalize
        """
        # 定义可接受的字符集，包括小写字母、大写字母、数字和点号
        accepted = (
            [chr(i) for i in range(ord("a"), ord("z") + 1)]
            + [chr(i) for i in range(ord("A"), ord("Z") + 1)]
            + [chr(i) for i in range(ord("0"), ord("9") + 1)]
            + ["."]
        )
        accepted = frozenset(accepted)  # 将字符集转换为不可变集合以提高性能
        pattern = re.compile(r"_+")  # 编译用于匹配多个下划线的正则表达式模式
        # 将文本转换为小写，并替换不在接受字符集中的字符为下划线
        text = "".join([c if c in accepted else "_" for c in text.lower()])
        text = pattern.sub("_", text).strip("_")  # 将多个连续的下划线替换为单个下划线，并去除首尾的下划线
        return text  # 返回规范化后的文本字符串

    # 定义一个方法，将歌词令牌列表转换为一个字符串
    def convert_lyric_tokens_to_string(self, lyrics: List[str]) -> str:
        return " ".join(lyrics)

    # 定义一个方法，用于将输入转换为张量（Tensor），可以选择添加批次轴
        """
        Convert the inner content to tensors.

        Args:
            tensor_type (`str` or [`~utils.TensorType`], *optional*):
                The type of tensors to use. If `str`, should be one of the values of the enum [`~utils.TensorType`]. If
                unset, no modification is done.
            prepend_batch_axis (`int`, *optional*, defaults to `False`):
                Whether or not to add the batch dimension during the conversion.
        """
        # Convert to TensorType
        if not isinstance(tensor_type, TensorType):
            # 如果 `tensor_type` 不是 `TensorType` 类型的实例，则转换为 `TensorType`
            tensor_type = TensorType(tensor_type)

        # Get a function reference for the correct framework
        if tensor_type == TensorType.TENSORFLOW:
            # 如果 `tensor_type` 是 `TensorType.TENSORFLOW`
            if not is_tf_available():
                # 检查 TensorFlow 是否可用，若不可用则抛出 ImportError 异常
                raise ImportError(
                    "Unable to convert output to TensorFlow tensors format, TensorFlow is not installed."
                )
            import tensorflow as tf

            # 使用 TensorFlow 的 constant 函数
            as_tensor = tf.constant
            # 使用 TensorFlow 的 is_tensor 函数
            is_tensor = tf.is_tensor
        elif tensor_type == TensorType.PYTORCH:
            # 如果 `tensor_type` 是 `TensorType.PYTORCH`
            if not is_torch_available():
                # 检查 PyTorch 是否可用，若不可用则抛出 ImportError 异常
                raise ImportError("Unable to convert output to PyTorch tensors format, PyTorch is not installed.")
            import torch

            # 使用 PyTorch 的 tensor 函数
            as_tensor = torch.tensor
            # 使用 PyTorch 的 is_tensor 函数
            is_tensor = torch.is_tensor
        elif tensor_type == TensorType.JAX:
            # 如果 `tensor_type` 是 `TensorType.JAX`
            if not is_flax_available():
                # 检查 JAX 是否可用，若不可用则抛出 ImportError 异常
                raise ImportError("Unable to convert output to JAX tensors format, JAX is not installed.")
            import jax.numpy as jnp  # noqa: F811

            # 使用 JAX 的 array 函数
            as_tensor = jnp.array
            # 使用自定义的 `_is_jax` 函数
            is_tensor = _is_jax
        else:
            # 默认情况下使用 NumPy 的 asarray 函数
            as_tensor = np.asarray
            # 使用自定义的 `_is_numpy` 函数
            is_tensor = _is_numpy

        # Do the tensor conversion in batch
        # 在批处理中进行张量转换

        try:
            if prepend_batch_axis:
                # 如果 `prepend_batch_axis` 为真，则在 `inputs` 前面添加一个批次维度
                inputs = [inputs]

            # 如果 `inputs` 不是张量，则使用 `as_tensor` 将其转换为张量
            if not is_tensor(inputs):
                inputs = as_tensor(inputs)
        except:  # noqa E722
            # 捕获所有异常，通常用于处理可能的数值或类型转换问题
            raise ValueError(
                "Unable to create tensor, you should probably activate truncation and/or padding "
                "with 'padding=True' 'truncation=True' to have batched tensors with the same length."
            )

        return inputs
    def __call__(self, artist, genres, lyrics="", return_tensors="pt") -> BatchEncoding:
        """Convert the raw string to a list of token ids

        Args:
            artist (`str`):
                Name of the artist.
            genres (`str`):
                List of genres that will be mixed to condition the audio
            lyrics (`str`, *optional*, defaults to `""`):
                Lyrics used to condition the generation
        """
        # 初始化输入的 token ids
        input_ids = [0, 0, 0]
        # 将 artist 复制多份，以匹配 self.version 的长度
        artist = [artist] * len(self.version)
        # 将 genres 复制多份，以匹配 self.version 的长度
        genres = [genres] * len(self.version)

        # 使用 tokenize 方法将 artist、genres 和 lyrics 转换为 tokens
        artists_tokens, genres_tokens, lyrics_tokens = self.tokenize(artist, genres, lyrics)
        # 将 tokens 转换为对应的 ids
        artists_id, genres_ids, full_tokens = self._convert_token_to_id(artists_tokens, genres_tokens, lyrics_tokens)

        # 初始化 attention_masks 为负无穷大
        attention_masks = [-INFINITY] * len(full_tokens[-1])
        # 根据每个版本的要求，将各个 ids 组合成 input_ids，并转换为 tensors
        input_ids = [
            self.convert_to_tensors(
                [input_ids + [artists_id[i]] + genres_ids[i] + full_tokens[i]], tensor_type=return_tensors
            )
            for i in range(len(self.version))
        ]
        # 返回 BatchEncoding 对象，包含 input_ids 和 attention_masks
        return BatchEncoding({"input_ids": input_ids, "attention_masks": attention_masks})

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Saves the tokenizer's vocabulary dictionary to the provided save_directory.

        Args:
            save_directory (`str`):
                A path to the directory where to saved. It will be created if it doesn't exist.

            filename_prefix (`Optional[str]`, *optional*):
                A prefix to add to the names of the files saved by the tokenizer.

        """
        # 检查 save_directory 是否存在，若不存在则记录错误信息并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return

        # 将 artists_encoder 转换为 JSON 格式并保存到指定路径
        artists_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["artists_file"]
        )
        with open(artists_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.artists_encoder, ensure_ascii=False))

        # 将 genres_encoder 转换为 JSON 格式并保存到指定路径
        genres_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["genres_file"]
        )
        with open(genres_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.genres_encoder, ensure_ascii=False))

        # 将 lyrics_encoder 转换为 JSON 格式并保存到指定路径
        lyrics_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["lyrics_file"]
        )
        with open(lyrics_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.lyrics_encoder, ensure_ascii=False))

        # 返回保存的文件路径元组
        return (artists_file, genres_file, lyrics_file)
    def _convert_id_to_token(self, artists_index, genres_index, lyric_index):
        """
        Converts an index (integer) in a token (str) using the vocab.

        Args:
            artists_index (`int`):
                Index of the artist in its corresponding dictionary.
            genres_index (`Union[List[int], int]`):
               Index of the genre in its corresponding dictionary. Can be a single index or a list of indices.
            lyric_index (`List[int]`):
                List of character indices, each corresponding to a character.

        Returns:
            artist (`Optional[str]`):
                Decoded artist name corresponding to artists_index.
            genres (`List[Optional[str]]`):
                List of decoded genre names corresponding to genres_index.
            lyrics (`List[Optional[str]]`):
                List of decoded characters corresponding to lyric_index.
        """
        # Retrieve artist name from artists_decoder using artists_index
        artist = self.artists_decoder.get(artists_index)
        
        # Retrieve genre names from genres_decoder for each genre index in genres_index
        genres = [self.genres_decoder.get(genre) for genre in genres_index]
        
        # Retrieve character representations from lyrics_decoder for each character index in lyric_index
        lyrics = [self.lyrics_decoder.get(character) for character in lyric_index]
        
        # Return the decoded artist name, list of decoded genres, and list of decoded characters
        return artist, genres, lyrics
```