# `.\models\jukebox\tokenization_jukebox.py`

```
# 设置文件编码为 utf-8
# 版权声明，声明代码作者和版权信息
# 根据 Apache 许可证 2.0 版本，使用此文件需要遵守许可证规定
# 可以在 http://www.apache.org/licenses/LICENSE-2.0 获取许可证的副本
# 除非法律要求或书面同意，否则不得使用此文件
# 根据许可证规定，本软件是基于"AS IS"的基础分发的，没有任何明示或暗示的担保或条件
# 请查看许可证以了解特定语言的权限和限制
# OpenAI Jukebox 的标记类

# 导入所需的库
import json
import os
import re
import unicodedata
from json.encoder import INFINITY
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import regex

# 导入所需的模块和函数
from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...tokenization_utils_base import BatchEncoding
from ...utils import TensorType, is_flax_available, is_tf_available, is_torch_available, logging
from ...utils.generic import _is_jax, _is_numpy

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义文件名和对应的词汇文件
VOCAB_FILES_NAMES = {
    "artists_file": "artists.json",
    "lyrics_file": "lyrics.json",
    "genres_file": "genres.json",
}

# 预训练词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "artists_file": {
        "jukebox": "https://huggingface.co/ArthurZ/jukebox/blob/main/artists.json",
    },
    "genres_file": {
        "jukebox": "https://huggingface.co/ArthurZ/jukebox/blob/main/genres.json",
    },
    "lyrics_file": {
        "jukebox": "https://huggingface.co/ArthurZ/jukebox/blob/main/lyrics.json",
    },
}

# 预训练歌词标记大小
PRETRAINED_LYRIC_TOKENS_SIZES = {
    "jukebox": 512,
}

# JukeboxTokenizer 类，继承自 PreTrainedTokenizer
class JukeboxTokenizer(PreTrainedTokenizer):
    """
    构建一个 Jukebox 分词器。Jukebox 可以根据 3 种不同的输入进行条件化：
        - 艺术家，每个艺术家都与提供的字典中的唯一 ID 相关联。
        - 风格，每个风格都与提供的字典中的唯一 ID 相关联。
        - 歌词，基于字符的分词。必须使用词汇表中存在的字符列表进行初始化。

    此分词器不需要训练。它应该能够处理不同数量的输入：
    因为模型的条件化可以在三个不同的查询上完成。如果未提供任何值，则将使用默认值。

    根据模型应该在哪些风格上进行条件化的数量 (`n_genres`)。
    ```python
    >>> from transformers import JukeboxTokenizer

    >>> tokenizer = JukeboxTokenizer.from_pretrained("openai/jukebox-1b-lyrics")
    >>> tokenizer("Alan Jackson", "Country Rock", "old town road")["input_ids"]
    [tensor([[   0,    0,    0, 6785,  546,   41,   38,   30,   76,   46,   41,   49,
               40,   76,   44,   41,   27,   30]]), tensor([[  0,   0,   0, 145,   0]]), tensor([[  0,   0,   0, 145,   0]])]
    ```
    # 通过在实例化此分词器时或在对某些文本调用时传递 `add_prefix_space=True` 来绕过这种行为，但由于模型不是以这种方式进行预训练的，可能会导致性能下降。
    
    <Tip>
    
    # 如果未提供任何内容，则流派和艺术家将随机选择或设置为 None
    
    </Tip>
    
    # 这个分词器继承自 [`PreTrainedTokenizer`]，其中包含大多数主要方法。用户应参考：
    # 这个超类以获取有关这些方法的更多信息。
    
    # 但是代码不允许这样做，只支持从各种流派中组合。
    
    Args:
        artists_file (`str`):
            # 包含艺术家和 ID 之间映射的词汇文件的路径。默认文件支持 "v2" 和 "v3"
        genres_file (`str`):
            # 包含流派和 ID 之间映射的词汇文件的路径。
        lyrics_file (`str`):
            # 包含歌词分词的接受字符的词汇文件的路径。
        version (`List[str]`, `optional`, 默认为 `["v3", "v2", "v2"]`) :
            # 分词器版本的列表。`5b-lyrics` 的顶级先前模型是使用 `v3` 而不是 `v2` 进行训练的。
        n_genres (`int`, `optional`, 默认为 1):
            # 用于组合的最大流派数量。
        max_n_lyric_tokens (`int`, `optional`, 默认为 512):
            # 要保留的最大歌词标记数量。
        unk_token (`str`, *optional*, 默认为 `"<|endoftext|>"`):
            # 未知标记。词汇表中不存在的标记无法转换为 ID，并将设置为此标记。
    """

    # 词汇文件名称
    vocab_files_names = VOCAB_FILES_NAMES
    # 预训练词汇文件映射
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 预训练歌词标记大小
    max_lyric_input_size = PRETRAINED_LYRIC_TOKENS_SIZES
    # 模型输入名称
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
        # 如果unk_token是字符串，则创建一个AddedToken对象，否则直接使用unk_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        # 初始化对象的属性
        self.version = version
        self.max_n_lyric_tokens = max_n_lyric_tokens
        self.n_genres = n_genres
        self._added_tokens_decoder = {0: unk_token}

        # 读取并加载艺术家编码器的JSON文件
        with open(artists_file, encoding="utf-8") as vocab_handle:
            self.artists_encoder = json.load(vocab_handle)

        # 读取并加载流派编码器的JSON文件
        with open(genres_file, encoding="utf-8") as vocab_handle:
            self.genres_encoder = json.load(vocab_handle)

        # 读取并加载歌词编码器的JSON文件
        with open(lyrics_file, encoding="utf-8") as vocab_handle:
            self.lyrics_encoder = json.load(vocab_handle)

        # 定义正则表达式用于匹配不在词汇表中的字符
        oov = r"[^A-Za-z0-9.,:;!?\-'\"()\[\] \t\n]+"
        # 如果歌词编码器长度为79，则修改正则表达式
        if len(self.lyrics_encoder) == 79:
            oov = oov.replace(r"\-'", r"\-+'")

        # 编译正则表达式
        self.out_of_vocab = regex.compile(oov)
        # 创建艺术家、流派、歌词的解码器
        self.artists_decoder = {v: k for k, v in self.artists_encoder.items()}
        self.genres_decoder = {v: k for k, v in self.genres_encoder.items()}
        self.lyrics_decoder = {v: k for k, v in self.lyrics_encoder.items()}
        # 调用父类的初始化方法
        super().__init__(
            unk_token=unk_token,
            n_genres=n_genres,
            version=version,
            max_n_lyric_tokens=max_n_lyric_tokens,
            **kwargs,
        )

    @property
    def vocab_size(self):
        # 返回词汇表的大小
        return len(self.artists_encoder) + len(self.genres_encoder) + len(self.lyrics_encoder)

    def get_vocab(self):
        # 返回包含艺术家、流派、歌词编码器的字典
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
        # 将艺术家、流派、歌词的token转换为它们在词汇表中的索引
        artists_id = [self.artists_encoder.get(artist, 0) for artist in list_artists]
        for genres in range(len(list_genres)):
            list_genres[genres] = [self.genres_encoder.get(genre, 0) for genre in list_genres[genres]]
            list_genres[genres] = list_genres[genres] + [-1] * (self.n_genres - len(list_genres[genres]))

        lyric_ids = [[self.lyrics_encoder.get(character, 0) for character in list_lyrics[0]], [], []]
        return artists_id, list_genres, lyric_ids
    # 将字符串转换为标记序列（字符串），使用标记器。对于基于单词的词汇表，将其拆分为单词；对于基于子词的词汇表（BPE/SentencePieces/WordPieces），将其拆分为子词。

    def _tokenize(self, lyrics):
        """
        Converts a string into a sequence of tokens (string), using the tokenizer. Split in words for word-based
        vocabulary or sub-words for sub-word-based vocabularies (BPE/SentencePieces/WordPieces).

        Do NOT take care of added tokens. Only the lyrics are split into character for the character-based vocabulary.
        """
        # 只有歌词没有被标记化，但基于字符的标记化很容易处理
        return list(lyrics)

    def tokenize(self, artist, genre, lyrics, **kwargs):
        """
        Converts three strings in a 3 sequence of tokens using the tokenizer
        """
        # 准备将艺术家、流派和歌词转换为标记序列
        artist, genre, lyrics = self.prepare_for_tokenization(artist, genre, lyrics)
        # 对歌词进行标记化
        lyrics = self._tokenize(lyrics)
        return artist, genre, lyrics

    def prepare_for_tokenization(
        self, artists: str, genres: str, lyrics: str, is_split_into_words: bool = False
    def prepare_for_tokenization(
        self,
        artist: str,
        genres: str,
        lyrics: str,
        is_split_into_words: bool = False
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
        # Loop through the versions of the tokenizer
        for idx in range(len(self.version)):
            # Check if the version is v3
            if self.version[idx] == "v3":
                # Lowercase the artist name
                artists[idx] = artists[idx].lower()
                # Lowercase the genre name and store it in a list
                genres[idx] = [genres[idx].lower()]
            else:
                # Normalize the artist name and append ".v2"
                artists[idx] = self._normalize(artists[idx]) + ".v2"
                # Normalize each genre and append ".v2", split genres if necessary
                genres[idx] = [
                    self._normalize(genre) + ".v2" for genre in genres[idx].split("_")
                ]  # split is for the full dictionary with combined genres

        # Check if the version is v2
        if self.version[0] == "v2":
            # Compile regex pattern for characters out of vocabulary
            self.out_of_vocab = regex.compile(r"[^A-Za-z0-9.,:;!?\-'\"()\[\] \t\n]+")
            # Define vocabulary characters
            vocab = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,:;!?-+'\"()[] \t\n"
            # Create vocabulary dictionary
            self.vocab = {vocab[index]: index + 1 for index in range(len(vocab))}
            # Add unknown token to vocabulary
            self.vocab["
    # 对输入文本进行标准化处理，用于流派和艺术家
    def _normalize(self, text: str) -> str:
        # 定义允许的字符集合，包括小写字母、大写字母、数字和句号
        accepted = (
            [chr(i) for i in range(ord("a"), ord("z") + 1)]
            + [chr(i) for i in range(ord("A"), ord("Z") + 1)]
            + [chr(i) for i in range(ord("0"), ord("9") + 1)]
            + ["."]
        )
        # 将字符集合转换为不可变集合
        accepted = frozenset(accepted)
        # 编译正则表达式模式，用于替换连续多个下划线为单个下划线
        pattern = re.compile(r"_+")
        # 将输入文本转换为小写，并根据允许的字符集合进行过滤，同时替换连续多个下划线为单个下划线并去除两端的下划线
        text = "".join([c if c in accepted else "_" for c in text.lower()])
        text = pattern.sub("_", text).strip("_")
        return text

    # 将歌词标记转换为一个字符串
    def convert_lyric_tokens_to_string(self, lyrics: List[str]) -> str:
        return " ".join(lyrics)

    # 将输入转换为张量
    def convert_to_tensors(
        self, inputs, tensor_type: Optional[Union[str, TensorType]] = None, prepend_batch_axis: bool = False
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
        if not isinstance(tensor_type, TensorType):  # 检查 tensor_type 是否为 TensorType 类型
            tensor_type = TensorType(tensor_type)  # 如果不是，将其转换为 TensorType 类型

        # Get a function reference for the correct framework
        if tensor_type == TensorType.TENSORFLOW:
            if not is_tf_available():  # 检查是否安装了 TensorFlow
                raise ImportError(
                    "Unable to convert output to TensorFlow tensors format, TensorFlow is not installed."
                )
            import tensorflow as tf  # 导入 TensorFlow 库

            as_tensor = tf.constant  # 设置 as_tensor 为 TensorFlow 的常量函数
            is_tensor = tf.is_tensor  # 设置 is_tensor 为 TensorFlow 的是否为张量函数
        elif tensor_type == TensorType.PYTORCH:
            if not is_torch_available():  # 检查是否安装了 PyTorch
                raise ImportError("Unable to convert output to PyTorch tensors format, PyTorch is not installed.")
            import torch  # 导入 PyTorch 库

            as_tensor = torch.tensor  # 设置 as_tensor 为 PyTorch 的张量函数
            is_tensor = torch.is_tensor  # 设置 is_tensor 为 PyTorch 的是否为张量函数
        elif tensor_type == TensorType.JAX:
            if not is_flax_available():  # 检查是否安装了 JAX
                raise ImportError("Unable to convert output to JAX tensors format, JAX is not installed.")
            import jax.numpy as jnp  # 导入 JAX 的 numpy 库  # noqa: F811

            as_tensor = jnp.array  # 设置 as_tensor 为 JAX 的数组函数
            is_tensor = _is_jax
        else:
            as_tensor = np.asarray  # 设置 as_tensor 为 numpy 的数组函数
            is_tensor = _is_numpy

        # Do the tensor conversion in batch

        try:
            if prepend_batch_axis:  # 检查是否需要在转换中添加批量维度
                inputs = [inputs]  # 如果需要，在输入数据外面包装一个数组

            if not is_tensor(inputs):  # 检查输入数据是否为张量
                inputs = as_tensor(inputs)  # 如果不是，将输入数据转换为张量
        except:  # noqa E722
            raise ValueError(
                "Unable to create tensor, you should probably activate truncation and/or padding "
                "with 'padding=True' 'truncation=True' to have batched tensors with the same length."
            )  # 捕获可能的异常，并抛出 ValueError

        return inputs  # 返回转换后的张量数据
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
        # 初始化输入的 token ids 列表
        input_ids = [0, 0, 0]
        # 将艺术家的名称扩展至与版本数相同的列表
        artist = [artist] * len(self.version)
        # 将音乐流派列表扩展至与版本数相同的列表
        genres = [genres] * len(self.version)

        # 将艺术家、音乐流派和歌词进行分词处理
        artists_tokens, genres_tokens, lyrics_tokens = self.tokenize(artist, genres, lyrics)
        # 将分词后的艺术家、音乐流派和歌词转换为 token ids
        artists_id, genres_ids, full_tokens = self._convert_token_to_id(artists_tokens, genres_tokens, lyrics_tokens)

        # 初始化注意力掩码为负无穷，长度为当前 token 列表的最后一个元素长度
        attention_masks = [-INFINITY] * len(full_tokens[-1])
        input_ids = [
            # 将 input_ids、艺术家 token ids、音乐流派 token ids、所有 token ids 转换为张量
            self.convert_to_tensors(
                [input_ids + [artists_id[i]] + genres_ids[i] + full_tokens[i]], tensor_type=return_tensors
            )
            for i in range(len(self.version))
        ]
        # 返回张量化后的结果
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
        # 如果保存目录不存在，则记录错误信息并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return

        # 将艺术家的词典存储为 JSON 文件
        artists_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["artists_file"]
        )
        with open(artists_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.artists_encoder, ensure_ascii=False))

        # 将音乐流派的词典存储为 JSON 文件
        genres_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["genres_file"]
        )
        with open(genres_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.genres_encoder, ensure_ascii=False))

        # 将歌词的词典存储为 JSON 文件
        lyrics_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["lyrics_file"]
        )
        with open(lyrics_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.lyrics_encoder, ensure_ascii=False))

        # 返回保存的文件路径元组
        return (artists_file, genres_file, lyrics_file)
    # 将艺术家、流派和歌词的索引转换为对应的字符串，使用相应的解码器
    def _convert_id_to_token(self, artists_index, genres_index, lyric_index):
        """
        Converts an index (integer) in a token (str) using the vocab.

        Args:
            artists_index (`int`):
                Index of the artist in its corresponding dictionary.
            genres_index (`Union[List[int], int]`):
               Index of the genre in its corresponding dictionary.
            lyric_index (`List[int]`):
                List of character indices, which each correspond to a character.
        """
        # 使用艺术家的解码器将索引转换为艺术家名称
        artist = self.artists_decoder.get(artists_index)
        # 使用流派的解码器将索引列表转换为流派名称列表
        genres = [self.genres_decoder.get(genre) for genre in genres_index]
        # 使用歌词的解码器将索引列表转换为对应的字符列表
        lyrics = [self.lyrics_decoder.get(character) for character in lyric_index]
        # 返回转换后的艺术家、流派和歌词名称
        return artist, genres, lyrics
```