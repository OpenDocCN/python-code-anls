# `.\transformers\models\bark\processing_bark.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，告知代码的版权和许可
"""
Bark 的 Processor 类
"""
# 导入所需模块
import json  # 导入用于 JSON 数据处理的模块
import os  # 导入用于操作系统相关功能的模块
from typing import Optional  # 导入用于类型提示的模块

import numpy as np  # 导入 NumPy 库，用于数值计算

from ...feature_extraction_utils import BatchFeature  # 导入 BatchFeature 类
from ...processing_utils import ProcessorMixin  # 导入 ProcessorMixin 类，用于处理相关功能
from ...utils import logging  # 导入日志记录模块
from ...utils.hub import get_file_from_repo  # 从 hub 导入从代码库获取文件的函数
from ..auto import AutoTokenizer  # 导入自动 Tokenizer 类

logger = logging.get_logger(__name__)  # 获取 logger 对象


class BarkProcessor(ProcessorMixin):
    r"""
    构建一个 Bark 处理器，将文本 tokenizer 和可选的 Bark 语音预设封装成一个单一的处理器。

    Args:
        tokenizer ([`PreTrainedTokenizer`]):
            一个 [`PreTrainedTokenizer`] 的实例。
        speaker_embeddings (`Dict[Dict[str]]`, *optional*):
            可选的嵌套的说话人嵌入字典。第一级包含声音预设名称（例如 `"en_speaker_4"`）。第二级包含 `"semantic_prompt"`、`"coarse_prompt"` 和 `"fine_prompt"` 嵌入。值对应相应 `np.ndarray` 的路径。
            参见[这里](https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c)以获取 `voice_preset_names` 列表。

    """

    tokenizer_class = "AutoTokenizer"  # 设置 tokenizer 类为 "AutoTokenizer"
    attributes = ["tokenizer"]  # 设置属性列表

    preset_shape = {
        "semantic_prompt": 1,
        "coarse_prompt": 2,
        "fine_prompt": 2,
    }  # 预设嵌入的形状

    def __init__(self, tokenizer, speaker_embeddings=None):
        """
        初始化函数。

        Args:
            tokenizer: 文本 tokenizer 实例。
            speaker_embeddings: 可选的说话人嵌入字典。
        """
        super().__init__(tokenizer)  # 调用父类的初始化函数

        self.speaker_embeddings = speaker_embeddings  # 设置说话人嵌入属性

    @classmethod
    def from_pretrained(
        cls, pretrained_processor_name_or_path, speaker_embeddings_dict_path="speaker_embeddings_path.json", **kwargs
    ):
        """
        从预训练获取处理器。

        Args:
            pretrained_processor_name_or_path: 预训练处理器的名称或路径。
            speaker_embeddings_dict_path: 说话人嵌入字典的路径，默认为 "speaker_embeddings_path.json"。
        """
        ):
        r"""
        Instantiate a Bark processor associated with a pretrained model.

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                This can be either:

                - a string, the *model id* of a pretrained [`BarkProcessor`] hosted inside a model repo on
                  huggingface.co. Valid model ids can be located at the root-level, like `bert-base-uncased`, or
                  namespaced under a user or organization name, like `dbmdz/bert-base-german-cased`.
                - a path to a *directory* containing a processor saved using the [`~BarkProcessor.save_pretrained`]
                  method, e.g., `./my_model_directory/`.
            speaker_embeddings_dict_path (`str`, *optional*, defaults to `"speaker_embeddings_path.json"`):
                The name of the `.json` file containing the speaker_embeddings dictionary located in
                `pretrained_model_name_or_path`. If `None`, no speaker_embeddings is loaded.
            **kwargs
                Additional keyword arguments passed along to both
                [`~tokenization_utils_base.PreTrainedTokenizer.from_pretrained`].
        """

        # 如果提供了说话者嵌入字典路径
        if speaker_embeddings_dict_path is not None:
            # 从预训练处理器名称或路径中获取说话者嵌入字典路径
            speaker_embeddings_path = get_file_from_repo(
                pretrained_processor_name_or_path,
                speaker_embeddings_dict_path,
                subfolder=kwargs.pop("subfolder", None),
                cache_dir=kwargs.pop("cache_dir", None),
                force_download=kwargs.pop("force_download", False),
                proxies=kwargs.pop("proxies", None),
                resume_download=kwargs.pop("resume_download", False),
                local_files_only=kwargs.pop("local_files_only", False),
                token=kwargs.pop("use_auth_token", None),
                revision=kwargs.pop("revision", None),
            )
            # 如果说话者嵌入字典路径为空
            if speaker_embeddings_path is None:
                # 发出警告，说明没有预加载的说话者嵌入将被使用
                logger.warning(
                    f"""`{os.path.join(pretrained_processor_name_or_path,speaker_embeddings_dict_path)}` does not exists
                    , no preloaded speaker embeddings will be used - Make sure to provide a correct path to the json
                    dictionary if wanted, otherwise set `speaker_embeddings_dict_path=None`."""
                )
                # 将说话者嵌入设置为None
                speaker_embeddings = None
            else:
                # 从说话者嵌入路径中加载说话者嵌入字典
                with open(speaker_embeddings_path) as speaker_embeddings_json:
                    speaker_embeddings = json.load(speaker_embeddings_json)
        else:
            # 如果没有提供说话者嵌入字典路径，则将说话者嵌入设置为None
            speaker_embeddings = None

        # 从预训练处理器名称或路径中实例化自动分词器
        tokenizer = AutoTokenizer.from_pretrained(pretrained_processor_name_or_path, **kwargs)

        # 返回实例化的BarkProcessor对象，包括分词器和说话者嵌入
        return cls(tokenizer=tokenizer, speaker_embeddings=speaker_embeddings)
    def save_pretrained(
        self,
        save_directory,
        speaker_embeddings_dict_path="speaker_embeddings_path.json",
        speaker_embeddings_directory="speaker_embeddings",
        push_to_hub: bool = False,
        **kwargs,
    ):
        """
        Saves the attributes of this processor (tokenizer...) in the specified directory so that it can be reloaded
        using the [`~BarkProcessor.from_pretrained`] method.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the tokenizer files and the speaker embeddings will be saved (directory will be created
                if it does not exist).
            speaker_embeddings_dict_path (`str`, *optional*, defaults to `"speaker_embeddings_path.json"`):
                The name of the `.json` file that will contains the speaker_embeddings nested path dictionnary, if it
                exists, and that will be located in `pretrained_model_name_or_path/speaker_embeddings_directory`.
            speaker_embeddings_directory (`str`, *optional*, defaults to `"speaker_embeddings/"`):
                The name of the folder in which the speaker_embeddings arrays will be saved.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            kwargs:
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        # 如果存在说话者嵌入，则创建文件夹保存说话者嵌入
        if self.speaker_embeddings is not None:
            os.makedirs(os.path.join(save_directory, speaker_embeddings_directory, "v2"), exist_ok=True)

            embeddings_dict = {}

            embeddings_dict["repo_or_path"] = save_directory

            # 遍历说话者嵌入字典
            for prompt_key in self.speaker_embeddings:
                if prompt_key != "repo_or_path":
                    # 加载声音预设
                    voice_preset = self._load_voice_preset(prompt_key)

                    tmp_dict = {}
                    # 遍历说话者嵌入的键值对
                    for key in self.speaker_embeddings[prompt_key]:
                        # 保存说话者嵌入数组为.npy文件
                        np.save(
                            os.path.join(
                                embeddings_dict["repo_or_path"], speaker_embeddings_directory, f"{prompt_key}_{key}"
                            ),
                            voice_preset[key],
                            allow_pickle=False,
                        )
                        tmp_dict[key] = os.path.join(speaker_embeddings_directory, f"{prompt_key}_{key}.npy")

                    embeddings_dict[prompt_key] = tmp_dict

            # 将说话者嵌入字典保存为.json文件
            with open(os.path.join(save_directory, speaker_embeddings_dict_path), "w") as fp:
                json.dump(embeddings_dict, fp)

        # 调用父类的保存预训练方法
        super().save_pretrained(save_directory, push_to_hub, **kwargs)
    # 加载语音预设数据，根据给定的语音预设名称获取对应路径
    def _load_voice_preset(self, voice_preset: str = None, **kwargs):
        # 获取语音预设路径字典
        voice_preset_paths = self.speaker_embeddings[voice_preset]

        # 创建空字典用于存储语音预设数据
        voice_preset_dict = {}

        # 遍历语音预设路径字典的关键字列表
        for key in ["semantic_prompt", "coarse_prompt", "fine_prompt"]:
            # 检查关键字是否存在于语音预设路径字典中
            if key not in voice_preset_paths:
                # 如果不存在，则抛出异常
                raise ValueError(
                    f"Voice preset unrecognized, missing {key} as a key in self.speaker_embeddings[{voice_preset}]."
                )

            # 获取语音预设数据的路径
            path = get_file_from_repo(
                # 从语音预设路径字典中获取路径所在的仓库或路径
                self.speaker_embeddings.get("repo_or_path", "/"),
                # 从语音预设路径字典中获取具体文件路径
                voice_preset_paths[key],
                # 设置子文件夹路径
                subfolder=kwargs.pop("subfolder", None),
                # 设置缓存目录
                cache_dir=kwargs.pop("cache_dir", None),
                # 设置是否强制下载
                force_download=kwargs.pop("force_download", False),
                # 设置代理
                proxies=kwargs.pop("proxies", None),
                # 设置是否恢复下载
                resume_download=kwargs.pop("resume_download", False),
                # 设置是否仅使用本地文件
                local_files_only=kwargs.pop("local_files_only", False),
                # 设置身份验证令牌
                token=kwargs.pop("use_auth_token", None),
                # 设置版本号
                revision=kwargs.pop("revision", None),
            )
            # 检查路径是否为空
            if path is None:
                # 如果为空，则抛出异常
                raise ValueError(
                    f"""`{os.path.join(self.speaker_embeddings.get("repo_or_path", "/"),voice_preset_paths[key])}` does not exists
                    , no preloaded voice preset will be used - Make sure to provide correct paths to the {voice_preset}
                    embeddings."""
                )

            # 使用 numpy 加载语音预设数据
            voice_preset_dict[key] = np.load(path)

        # 返回语音预设数据字典
        return voice_preset_dict

    # 验证语音预设数据字典的有效性
    def _validate_voice_preset_dict(self, voice_preset: Optional[dict] = None):
        # 遍历语音预设数据字典的关键字列表
        for key in ["semantic_prompt", "coarse_prompt", "fine_prompt"]:
            # 检查关键字是否存在于语音预设数据字典中
            if key not in voice_preset:
                # 如果不存在，则抛出异常
                raise ValueError(f"Voice preset unrecognized, missing {key} as a key.")

            # 检查语音预设数据是否为 numpy 数组
            if not isinstance(voice_preset[key], np.ndarray):
                # 如果不是，则抛出异常
                raise ValueError(f"{key} voice preset must be a {str(self.preset_shape[key])}D ndarray.")

            # 检查语音预设数据的维度是否正确
            if len(voice_preset[key].shape) != self.preset_shape[key]:
                # 如果维度不正确，则抛出异常
                raise ValueError(f"{key} voice preset must be a {str(self.preset_shape[key])}D ndarray.")

    # 调用函数，生成文本对应的音频
    def __call__(
        self,
        text=None,
        voice_preset=None,
        return_tensors="pt",
        max_length=256,
        add_special_tokens=False,
        return_attention_mask=True,
        return_token_type_ids=False,
        **kwargs,
```