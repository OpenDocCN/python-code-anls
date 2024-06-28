# `.\models\bark\processing_bark.py`

```
"""
Processor class for Bark
"""
# 引入必要的库和模块
import json  # 导入处理 JSON 的模块
import os  # 导入操作系统相关功能的模块
from typing import Optional  # 导入类型提示中的 Optional 类型

import numpy as np  # 导入 NumPy 库

# 导入所需的自定义模块和函数
from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessorMixin
from ...utils import logging
from ...utils.hub import get_file_from_repo
from ..auto import AutoTokenizer  # 导入自动化的 Tokenizer

# 获取日志记录器
logger = logging.get_logger(__name__)


class BarkProcessor(ProcessorMixin):
    r"""
    Constructs a Bark processor which wraps a text tokenizer and optional Bark voice presets into a single processor.

    Args:
        tokenizer ([`PreTrainedTokenizer`]):
            An instance of [`PreTrainedTokenizer`].
        speaker_embeddings (`Dict[Dict[str]]`, *optional*):
            Optional nested speaker embeddings dictionary. The first level contains voice preset names (e.g
            `"en_speaker_4"`). The second level contains `"semantic_prompt"`, `"coarse_prompt"` and `"fine_prompt"`
            embeddings. The values correspond to the path of the corresponding `np.ndarray`. See
            [here](https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c) for
            a list of `voice_preset_names`.

    """

    # 类属性：指定使用的 Tokenizer 类型
    tokenizer_class = "AutoTokenizer"
    # 类属性：指定必须具备的属性列表
    attributes = ["tokenizer"]

    # 类属性：定义不同类型的预设形状
    preset_shape = {
        "semantic_prompt": 1,
        "coarse_prompt": 2,
        "fine_prompt": 2,
    }

    def __init__(self, tokenizer, speaker_embeddings=None):
        # 构造函数：初始化 BarkProcessor 实例
        super().__init__(tokenizer)
        # 初始化属性：说话者嵌入（可选）
        self.speaker_embeddings = speaker_embeddings

    @classmethod
    def from_pretrained(
        cls, pretrained_processor_name_or_path, speaker_embeddings_dict_path="speaker_embeddings_path.json", **kwargs
    ):
        # 类方法：从预训练的处理器名称或路径创建 BarkProcessor 实例
        ):
            r"""
            Instantiate a Bark processor associated with a pretrained model.

            Args:
                pretrained_model_name_or_path (`str` or `os.PathLike`):
                    This can be either:

                    - a string, the *model id* of a pretrained [`BarkProcessor`] hosted inside a model repo on
                      huggingface.co.
                    - a path to a *directory* containing a processor saved using the [`~BarkProcessor.save_pretrained`]
                      method, e.g., `./my_model_directory/`.
                speaker_embeddings_dict_path (`str`, *optional*, defaults to `"speaker_embeddings_path.json"`):
                    The name of the `.json` file containing the speaker_embeddings dictionary located in
                    `pretrained_model_name_or_path`. If `None`, no speaker_embeddings is loaded.
                **kwargs
                    Additional keyword arguments passed along to both
                    [`~tokenization_utils_base.PreTrainedTokenizer.from_pretrained`].
            """

            if speaker_embeddings_dict_path is not None:
                # 获取存储在指定路径的说话者嵌入字典文件路径
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
                if speaker_embeddings_path is None:
                    # 若找不到指定的说话者嵌入字典文件路径，则警告并设置为 None
                    logger.warning(
                        f"""`{os.path.join(pretrained_processor_name_or_path,speaker_embeddings_dict_path)}` does not exists
                        , no preloaded speaker embeddings will be used - Make sure to provide a correct path to the json
                        dictionnary if wanted, otherwise set `speaker_embeddings_dict_path=None`."""
                    )
                    speaker_embeddings = None
                else:
                    # 若找到了指定的说话者嵌入字典文件路径，则读取其中的内容为 JSON 格式的字典数据
                    with open(speaker_embeddings_path) as speaker_embeddings_json:
                        speaker_embeddings = json.load(speaker_embeddings_json)
            else:
                # 如果未提供说话者嵌入字典文件路径，则设置为 None
                speaker_embeddings = None

            # 使用预训练的模型名或路径加载 tokenizer
            tokenizer = AutoTokenizer.from_pretrained(pretrained_processor_name_or_path, **kwargs)

            # 返回实例化后的 BarkProcessor 对象，包括 tokenizer 和 speaker_embeddings
            return cls(tokenizer=tokenizer, speaker_embeddings=speaker_embeddings)

        def save_pretrained(
            self,
            save_directory,
            speaker_embeddings_dict_path="speaker_embeddings_path.json",
            speaker_embeddings_directory="speaker_embeddings",
            push_to_hub: bool = False,
            **kwargs,
        """
        Saves the attributes of this processor (tokenizer...) in the specified directory so that it can be reloaded
        using the [`~BarkProcessor.from_pretrained`] method.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the tokenizer files and the speaker embeddings will be saved (directory will be created
                if it does not exist).
            speaker_embeddings_dict_path (`str`, *optional*, defaults to `"speaker_embeddings_path.json"`):
                The name of the `.json` file that will contains the speaker_embeddings nested path dictionary, if it
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
        # 如果存在说话者嵌入，则创建保存目录
        if self.speaker_embeddings is not None:
            # 创建目录结构，确保路径存在
            os.makedirs(os.path.join(save_directory, speaker_embeddings_directory, "v2"), exist_ok=True)

            # 创建一个空的嵌入字典
            embeddings_dict = {}

            # 设置存储路径为保存目录
            embeddings_dict["repo_or_path"] = save_directory

            # 遍历每个提示键（prompt_key）
            for prompt_key in self.speaker_embeddings:
                # 排除键名为 "repo_or_path" 的情况
                if prompt_key != "repo_or_path":
                    # 载入语音预设
                    voice_preset = self._load_voice_preset(prompt_key)

                    # 创建临时字典
                    tmp_dict = {}
                    # 遍历每个键（key）
                    for key in self.speaker_embeddings[prompt_key]:
                        # 将语音预设保存为 .npy 文件
                        np.save(
                            os.path.join(
                                embeddings_dict["repo_or_path"], speaker_embeddings_directory, f"{prompt_key}_{key}"
                            ),
                            voice_preset[key],
                            allow_pickle=False,
                        )
                        # 更新临时字典
                        tmp_dict[key] = os.path.join(speaker_embeddings_directory, f"{prompt_key}_{key}.npy")

                    # 将临时字典添加到嵌入字典中
                    embeddings_dict[prompt_key] = tmp_dict

            # 将嵌入字典保存为 JSON 文件
            with open(os.path.join(save_directory, speaker_embeddings_dict_path), "w") as fp:
                json.dump(embeddings_dict, fp)

        # 调用父类方法保存预训练模型到指定目录，并可选择推送到 Hugging Face 模型中心
        super().save_pretrained(save_directory, push_to_hub, **kwargs)
    # 加载指定的语音预设数据，支持的关键字包括语义提示、粗略提示和细致提示
    def _load_voice_preset(self, voice_preset: str = None, **kwargs):
        # 从self.speaker_embeddings中获取指定voice_preset的路径信息
        voice_preset_paths = self.speaker_embeddings[voice_preset]

        # 初始化空字典用于存储语音预设数据
        voice_preset_dict = {}

        # 遍历语音预设的三个关键字：语义提示、粗略提示、细致提示
        for key in ["semantic_prompt", "coarse_prompt", "fine_prompt"]:
            # 检查路径信息中是否包含当前关键字，若不存在则抛出数值错误异常
            if key not in voice_preset_paths:
                raise ValueError(
                    f"Voice preset unrecognized, missing {key} as a key in self.speaker_embeddings[{voice_preset}]."
                )

            # 根据路径信息获取预设文件的路径
            path = get_file_from_repo(
                # 从self.speaker_embeddings获取存储库路径或基础路径
                self.speaker_embeddings.get("repo_or_path", "/"),
                # 获取指定key的文件路径
                voice_preset_paths[key],
                subfolder=kwargs.pop("subfolder", None),  # 子文件夹（可选）
                cache_dir=kwargs.pop("cache_dir", None),  # 缓存目录（可选）
                force_download=kwargs.pop("force_download", False),  # 是否强制下载（可选）
                proxies=kwargs.pop("proxies", None),  # 代理设置（可选）
                resume_download=kwargs.pop("resume_download", False),  # 是否恢复下载（可选）
                local_files_only=kwargs.pop("local_files_only", False),  # 仅使用本地文件（可选）
                token=kwargs.pop("use_auth_token", None),  # 认证令牌（可选）
                revision=kwargs.pop("revision", None),  # 版本号（可选）
            )
            
            # 若路径为None，则抛出数值错误异常，说明找不到指定路径的预设数据
            if path is None:
                raise ValueError(
                    f"""`{os.path.join(self.speaker_embeddings.get("repo_or_path", "/"), voice_preset_paths[key])}` does not exists
                    , no preloaded voice preset will be used - Make sure to provide correct paths to the {voice_preset}
                    embeddings."""
                )

            # 使用numpy加载指定路径的数据，存储到语音预设字典中的当前关键字位置
            voice_preset_dict[key] = np.load(path)

        # 返回加载后的语音预设字典
        return voice_preset_dict

    # 验证语音预设字典的有效性，确保包含必须的关键字和正确的数据类型和形状
    def _validate_voice_preset_dict(self, voice_preset: Optional[dict] = None):
        # 遍历语音预设的三个关键字：语义提示、粗略提示、细致提示
        for key in ["semantic_prompt", "coarse_prompt", "fine_prompt"]:
            # 检查语音预设字典是否缺少当前关键字，若是则抛出数值错误异常
            if key not in voice_preset:
                raise ValueError(f"Voice preset unrecognized, missing {key} as a key.")

            # 检查当前关键字的值是否为numpy数组类型，若不是则抛出数值错误异常
            if not isinstance(voice_preset[key], np.ndarray):
                raise ValueError(f"{key} voice preset must be a {str(self.preset_shape[key])}D ndarray.")

            # 检查当前关键字的值的维度是否与预期维度一致，若不是则抛出数值错误异常
            if len(voice_preset[key].shape) != self.preset_shape[key]:
                raise ValueError(f"{key} voice preset must be a {str(self.preset_shape[key])}D ndarray.")

    # 对象的可调用方法，用于执行模型的推理或生成任务
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