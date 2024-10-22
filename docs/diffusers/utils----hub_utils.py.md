# `.\diffusers\utils\hub_utils.py`

```
# coding=utf-8  # 指定文件的编码为 UTF-8
# Copyright 2024 The HuggingFace Inc. team.  # 文件版权信息
#
# Licensed under the Apache License, Version 2.0 (the "License");  # 许可证声明
# you may not use this file except in compliance with the License.  # 使用许可证的条件说明
# You may obtain a copy of the License at  # 许可证获取链接
#
#     http://www.apache.org/licenses/LICENSE-2.0  # 许可证链接
#
# Unless required by applicable law or agreed to in writing, software  # 免责条款
# distributed under the License is distributed on an "AS IS" BASIS,  # 免责条款的进一步说明
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  # 不提供任何明示或暗示的保证
# See the License for the specific language governing permissions and  # 查看许可证以获取特定权限
# limitations under the License.  # 以及在许可证下的限制


import json  # 导入 json 模块，用于处理 JSON 数据
import os  # 导入 os 模块，用于与操作系统交互
import re  # 导入 re 模块，用于正则表达式操作
import sys  # 导入 sys 模块，用于访问与 Python 解释器相关的信息
import tempfile  # 导入 tempfile 模块，用于创建临时文件
import traceback  # 导入 traceback 模块，用于处理异常的跟踪信息
import warnings  # 导入 warnings 模块，用于发出警告
from pathlib import Path  # 从 pathlib 导入 Path 类，用于路径操作
from typing import Dict, List, Optional, Union  # 导入类型提示相关的类
from uuid import uuid4  # 从 uuid 导入 uuid4 函数，用于生成唯一标识符

# 从 huggingface_hub 导入所需的模块和函数
from huggingface_hub import (
    ModelCard,  # 导入 ModelCard 类，用于模型卡片管理
    ModelCardData,  # 导入 ModelCardData 类，用于处理模型卡片数据
    create_repo,  # 导入 create_repo 函数，用于创建模型仓库
    hf_hub_download,  # 导入 hf_hub_download 函数，用于下载模型
    model_info,  # 导入 model_info 函数，用于获取模型信息
    snapshot_download,  # 导入 snapshot_download 函数，用于下载快照
    upload_folder,  # 导入 upload_folder 函数，用于上传文件夹
)
# 导入 huggingface_hub 的常量
from huggingface_hub.constants import HF_HUB_CACHE, HF_HUB_DISABLE_TELEMETRY, HF_HUB_OFFLINE
# 从 huggingface_hub.file_download 导入正则表达式相关内容
from huggingface_hub.file_download import REGEX_COMMIT_HASH
# 导入 huggingface_hub.utils 的多个异常处理和实用函数
from huggingface_hub.utils import (
    EntryNotFoundError,  # 导入找不到条目的异常
    RepositoryNotFoundError,  # 导入找不到仓库的异常
    RevisionNotFoundError,  # 导入找不到修订版本的异常
    is_jinja_available,  # 导入检查 Jinja 模板是否可用的函数
    validate_hf_hub_args,  # 导入验证 Hugging Face Hub 参数的函数
)
from packaging import version  # 导入 version 模块，用于版本处理
from requests import HTTPError  # 从 requests 导入 HTTPError 异常，用于处理 HTTP 错误

from .. import __version__  # 导入当前包的版本信息
from .constants import (
    DEPRECATED_REVISION_ARGS,  # 导入已弃用的修订参数常量
    HUGGINGFACE_CO_RESOLVE_ENDPOINT,  # 导入 Hugging Face 解析端点常量
    SAFETENSORS_WEIGHTS_NAME,  # 导入安全张量权重名称常量
    WEIGHTS_NAME,  # 导入权重名称常量
)
from .import_utils import (
    ENV_VARS_TRUE_VALUES,  # 导入环境变量真值集合
    _flax_version,  # 导入 Flax 版本
    _jax_version,  # 导入 JAX 版本
    _onnxruntime_version,  # 导入 ONNX 运行时版本
    _torch_version,  # 导入 PyTorch 版本
    is_flax_available,  # 导入检查 Flax 是否可用的函数
    is_onnx_available,  # 导入检查 ONNX 是否可用的函数
    is_torch_available,  # 导入检查 PyTorch 是否可用的函数
)
from .logging import get_logger  # 从 logging 模块导入获取日志记录器的函数

logger = get_logger(__name__)  # 获取当前模块的日志记录器实例

MODEL_CARD_TEMPLATE_PATH = Path(__file__).parent / "model_card_template.md"  # 设置模型卡片模板文件的路径
SESSION_ID = uuid4().hex  # 生成一个唯一的会话 ID

def http_user_agent(user_agent: Union[Dict, str, None] = None) -> str:  # 定义一个格式化用户代理字符串的函数
    """
    Formats a user-agent string with basic info about a request.  # 函数说明，格式化用户代理字符串
    """
    ua = f"diffusers/{__version__}; python/{sys.version.split()[0]}; session_id/{SESSION_ID}"  # 构建基本用户代理字符串
    if HF_HUB_DISABLE_TELEMETRY or HF_HUB_OFFLINE:  # 检查是否禁用遥测或处于离线状态
        return ua + "; telemetry/off"  # 返回禁用遥测的用户代理字符串
    if is_torch_available():  # 检查 PyTorch 是否可用
        ua += f"; torch/{_torch_version}"  # 将 PyTorch 版本信息添加到用户代理字符串
    if is_flax_available():  # 检查 Flax 是否可用
        ua += f"; jax/{_jax_version}"  # 将 JAX 版本信息添加到用户代理字符串
        ua += f"; flax/{_flax_version}"  # 将 Flax 版本信息添加到用户代理字符串
    if is_onnx_available():  # 检查 ONNX 是否可用
        ua += f"; onnxruntime/{_onnxruntime_version}"  # 将 ONNX 运行时版本信息添加到用户代理字符串
    # CI will set this value to True  # CI 会将此值设置为 True
    if os.environ.get("DIFFUSERS_IS_CI", "").upper() in ENV_VARS_TRUE_VALUES:  # 检查环境变量是否指示在 CI 中运行
        ua += "; is_ci/true"  # 如果是 CI，添加相关信息到用户代理字符串
    if isinstance(user_agent, dict):  # 检查用户代理是否为字典类型
        ua += "; " + "; ".join(f"{k}/{v}" for k, v in user_agent.items())  # 将字典项格式化为字符串并添加到用户代理
    elif isinstance(user_agent, str):  # 检查用户代理是否为字符串类型
        ua += "; " + user_agent  # 直接添加用户代理字符串
    return ua  # 返回最终的用户代理字符串


def load_or_create_model_card(  # 定义加载或创建模型卡片的函数
    repo_id_or_path: str = None,  # 仓库 ID 或路径，默认为 None
    token: Optional[str] = None,  # 访问令牌，默认为 None
    is_pipeline: bool = False,  # 是否为管道模型，默认为 False
    from_training: bool = False,  # 是否从训练中加载，默认为 False
    # 定义模型描述，类型为可选的字符串，默认值为 None
        model_description: Optional[str] = None,
        # 定义基础模型，类型为字符串，默认值为 None
        base_model: str = None,
        # 定义提示信息，类型为可选的字符串，默认值为 None
        prompt: Optional[str] = None,
        # 定义许可证信息，类型为可选的字符串，默认值为 None
        license: Optional[str] = None,
        # 定义小部件列表，类型为可选的字典列表，默认值为 None
        widget: Optional[List[dict]] = None,
        # 定义推理标志，类型为可选的布尔值，默认值为 None
        inference: Optional[bool] = None,
# 定义一个函数，返回类型为 ModelCard
) -> ModelCard:
    """
    加载或创建模型卡片。

    参数:
        repo_id_or_path (`str`):
            仓库 ID（例如 "runwayml/stable-diffusion-v1-5"）或查找模型卡片的本地路径。
        token (`str`, *可选*):
            认证令牌。默认为存储的令牌。详细信息见 https://huggingface.co/settings/token。
        is_pipeline (`bool`):
            布尔值，指示是否为 [`DiffusionPipeline`] 添加标签。
        from_training: (`bool`): 布尔标志，表示模型卡片是否是从训练脚本创建的。
        model_description (`str`, *可选*): 要添加到模型卡片的模型描述。在从训练脚本使用 `load_or_create_model_card` 时有用。
        base_model (`str`): 基础模型标识符（例如 "stabilityai/stable-diffusion-xl-base-1.0"）。对类似 DreamBooth 的训练有用。
        prompt (`str`, *可选*): 用于训练的提示。对类似 DreamBooth 的训练有用。
        license: (`str`, *可选*): 输出工件的许可证。在从训练脚本使用 `load_or_create_model_card` 时有用。
        widget (`List[dict]`, *可选*): 附带画廊模板的部件。
        inference: (`bool`, *可选*): 是否开启推理部件。在从训练脚本使用 `load_or_create_model_card` 时有用。
    """
    # 检查是否安装了 Jinja 模板引擎
    if not is_jinja_available():
        # 如果未安装，抛出一个值错误，并提供安装建议
        raise ValueError(
            "Modelcard 渲染基于 Jinja 模板。"
            " 请确保在使用 `load_or_create_model_card` 之前安装了 `jinja`."
            " 要安装它，请运行 `pip install Jinja2`."
        )

    try:
        # 检查远程仓库中是否存在模型卡片
        model_card = ModelCard.load(repo_id_or_path, token=token)
    except (EntryNotFoundError, RepositoryNotFoundError):
        # 如果模型卡片不存在，则根据模板创建一个模型卡片
        if from_training:
            # 从模板创建模型卡片，并使用卡片数据作为 YAML 块
            model_card = ModelCard.from_template(
                card_data=ModelCardData(  # 卡片元数据对象
                    license=license,
                    library_name="diffusers",  # 指定库名
                    inference=inference,  # 指定推理设置
                    base_model=base_model,  # 指定基础模型
                    instance_prompt=prompt,  # 指定实例提示
                    widget=widget,  # 指定部件
                ),
                template_path=MODEL_CARD_TEMPLATE_PATH,  # 模板路径
                model_description=model_description,  # 模型描述
            )
        else:
            # 创建一个空的模型卡片数据对象
            card_data = ModelCardData()
            # 根据 is_pipeline 变量确定组件类型
            component = "pipeline" if is_pipeline else "model"
            # 如果没有提供模型描述，则生成默认描述
            if model_description is None:
                model_description = f"This is the model card of a 🧨 diffusers {component} that has been pushed on the Hub. This model card has been automatically generated."
            # 从模板创建模型卡片
            model_card = ModelCard.from_template(card_data, model_description=model_description)
    # 返回模型卡片的内容
        return model_card
# 定义一个函数，用于填充模型卡片的库名称和可选标签
def populate_model_card(model_card: ModelCard, tags: Union[str, List[str]] = None) -> ModelCard:
    # 如果模型卡片的库名称为空，则设置为 "diffusers"
    if model_card.data.library_name is None:
        model_card.data.library_name = "diffusers"

    # 如果标签不为空
    if tags is not None:
        # 如果标签是字符串，则转换为列表
        if isinstance(tags, str):
            tags = [tags]
        # 如果模型卡片的标签为空，则初始化为空列表
        if model_card.data.tags is None:
            model_card.data.tags = []
        # 遍历所有标签，将它们添加到模型卡片的标签中
        for tag in tags:
            model_card.data.tags.append(tag)

    # 返回更新后的模型卡片
    return model_card


# 定义一个函数，从已解析的文件名中提取提交哈希
def extract_commit_hash(resolved_file: Optional[str], commit_hash: Optional[str] = None):
    # 提取提交哈希，优先使用提供的提交哈希
    if resolved_file is None or commit_hash is not None:
        return commit_hash
    # 将解析后的文件路径转换为 POSIX 格式
    resolved_file = str(Path(resolved_file).as_posix())
    # 在文件路径中搜索提交哈希的模式
    search = re.search(r"snapshots/([^/]+)/", resolved_file)
    # 如果未找到模式，则返回 None
    if search is None:
        return None
    # 从搜索结果中提取提交哈希
    commit_hash = search.groups()[0]
    # 如果提交哈希符合规定格式，则返回它，否则返回 None
    return commit_hash if REGEX_COMMIT_HASH.match(commit_hash) else None


# 定义旧的默认缓存路径，可能需要迁移
# 该逻辑大体来源于 `transformers`，并有如下不同之处：
# - Diffusers 不使用自定义环境变量来指定缓存路径。
# - 无需迁移缓存格式，只需将文件移动到新位置。
hf_cache_home = os.path.expanduser(
    # 获取环境变量 HF_HOME，默认路径为 ~/.cache/huggingface
    os.getenv("HF_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "huggingface"))
)
# 定义旧的 diffusers 缓存路径
old_diffusers_cache = os.path.join(hf_cache_home, "diffusers")


# 定义一个函数，用于移动缓存目录
def move_cache(old_cache_dir: Optional[str] = None, new_cache_dir: Optional[str] = None) -> None:
    # 如果新缓存目录为空，则设置为 HF_HUB_CACHE
    if new_cache_dir is None:
        new_cache_dir = HF_HUB_CACHE
    # 如果旧缓存目录为空，则使用旧的 diffusers 缓存路径
    if old_cache_dir is None:
        old_cache_dir = old_diffusers_cache

    # 扩展用户目录路径
    old_cache_dir = Path(old_cache_dir).expanduser()
    new_cache_dir = Path(new_cache_dir).expanduser()
    # 遍历旧缓存目录中的所有 blob 文件
    for old_blob_path in old_cache_dir.glob("**/blobs/*"):
        # 如果路径是文件且不是符号链接
        if old_blob_path.is_file() and not old_blob_path.is_symlink():
            # 计算新 blob 文件的路径
            new_blob_path = new_cache_dir / old_blob_path.relative_to(old_cache_dir)
            # 创建新路径的父目录
            new_blob_path.parent.mkdir(parents=True, exist_ok=True)
            # 替换旧的 blob 文件为新的 blob 文件
            os.replace(old_blob_path, new_blob_path)
            # 尝试在旧路径和新路径之间创建符号链接
            try:
                os.symlink(new_blob_path, old_blob_path)
            except OSError:
                # 如果无法创建符号链接，发出警告
                logger.warning(
                    "Could not create symlink between old cache and new cache. If you use an older version of diffusers again, files will be re-downloaded."
                )
    # 现在，old_cache_dir 包含指向新缓存的符号链接（仍然可以使用）


# 定义缓存版本文件的路径
cache_version_file = os.path.join(HF_HUB_CACHE, "version_diffusers_cache.txt")
# 如果缓存版本文件不存在，则设置缓存版本为 0
if not os.path.isfile(cache_version_file):
    cache_version = 0
else:
    # 打开文件以读取缓存版本
    with open(cache_version_file) as f:
        try:
            # 尝试将读取内容转换为整数
            cache_version = int(f.read())
        except ValueError:
            # 如果转换失败，则设置缓存版本为 0
            cache_version = 0

# 如果缓存版本小于 1
if cache_version < 1:
    # 检查旧的缓存目录是否存在且非空
        old_cache_is_not_empty = os.path.isdir(old_diffusers_cache) and len(os.listdir(old_diffusers_cache)) > 0
        # 如果旧缓存不为空，则记录警告信息
        if old_cache_is_not_empty:
            logger.warning(
                "The cache for model files in Diffusers v0.14.0 has moved to a new location. Moving your "
                "existing cached models. This is a one-time operation, you can interrupt it or run it "
                "later by calling `diffusers.utils.hub_utils.move_cache()`."
            )
            # 尝试移动缓存
            try:
                move_cache()
            # 捕获任何异常并处理
            except Exception as e:
                # 获取异常的追踪信息并格式化为字符串
                trace = "\n".join(traceback.format_tb(e.__traceback__))
                # 记录错误信息，建议用户在 GitHub 提交问题
                logger.error(
                    f"There was a problem when trying to move your cache:\n\n{trace}\n{e.__class__.__name__}: {e}\n\nPlease "
                    "file an issue at https://github.com/huggingface/diffusers/issues/new/choose, copy paste this whole "
                    "message and we will do our best to help."
                )
# 检查缓存版本是否小于1
if cache_version < 1:
    # 尝试创建缓存目录
    try:
        os.makedirs(HF_HUB_CACHE, exist_ok=True)  # 创建目录，如果已存在则不报错
        # 打开缓存版本文件以写入版本号
        with open(cache_version_file, "w") as f:
            f.write("1")  # 写入版本号1
    except Exception:  # 捕获异常
        # 记录警告信息，提示用户可能存在的问题
        logger.warning(
            f"There was a problem when trying to write in your cache folder ({HF_HUB_CACHE}). Please, ensure "
            "the directory exists and can be written to."
        )

# 定义函数以添加变体到权重名称
def _add_variant(weights_name: str, variant: Optional[str] = None) -> str:
    # 如果变体不为 None
    if variant is not None:
        # 按 '.' 分割权重名称
        splits = weights_name.split(".")
        # 确定分割索引
        split_index = -2 if weights_name.endswith(".index.json") else -1
        # 更新权重名称的分割部分，插入变体
        splits = splits[:-split_index] + [variant] + splits[-split_index:]
        # 重新连接分割部分为完整的权重名称
        weights_name = ".".join(splits)

    # 返回更新后的权重名称
    return weights_name

# 装饰器用于验证 HF Hub 的参数
@validate_hf_hub_args
def _get_model_file(
    pretrained_model_name_or_path: Union[str, Path],  # 预训练模型的名称或路径
    *,
    weights_name: str,  # 权重文件的名称
    subfolder: Optional[str] = None,  # 子文件夹，默认为 None
    cache_dir: Optional[str] = None,  # 缓存目录，默认为 None
    force_download: bool = False,  # 强制下载标志，默认为 False
    proxies: Optional[Dict] = None,  # 代理设置，默认为 None
    local_files_only: bool = False,  # 仅使用本地文件的标志，默认为 False
    token: Optional[str] = None,  # 访问令牌，默认为 None
    user_agent: Optional[Union[Dict, str]] = None,  # 用户代理设置，默认为 None
    revision: Optional[str] = None,  # 修订版本，默认为 None
    commit_hash: Optional[str] = None,  # 提交哈希值，默认为 None
):
    # 将预训练模型路径转换为字符串
    pretrained_model_name_or_path = str(pretrained_model_name_or_path)
    # 如果路径指向一个文件
    if os.path.isfile(pretrained_model_name_or_path):
        return pretrained_model_name_or_path  # 直接返回文件路径
    # 如果路径指向一个目录
    elif os.path.isdir(pretrained_model_name_or_path):
        # 检查目录中是否存在权重文件
        if os.path.isfile(os.path.join(pretrained_model_name_or_path, weights_name)):
            # 从 PyTorch 检查点加载模型文件
            model_file = os.path.join(pretrained_model_name_or_path, weights_name)
            return model_file  # 返回模型文件路径
        # 如果有子文件夹且子文件夹中存在权重文件
        elif subfolder is not None and os.path.isfile(
            os.path.join(pretrained_model_name_or_path, subfolder, weights_name)
        ):
            model_file = os.path.join(pretrained_model_name_or_path, subfolder, weights_name)
            return model_file  # 返回子文件夹中的模型文件路径
        else:
            # 抛出环境错误，指示未找到权重文件
            raise EnvironmentError(
                f"Error no file named {weights_name} found in directory {pretrained_model_name_or_path}."
            )

# 检查本地是否存在分片文件的函数
def _check_if_shards_exist_locally(local_dir, subfolder, original_shard_filenames):
    # 构造分片文件的路径
    shards_path = os.path.join(local_dir, subfolder)
    # 获取所有分片文件的完整路径
    shard_filenames = [os.path.join(shards_path, f) for f in original_shard_filenames]
    # 遍历每个分片文件
    for shard_file in shard_filenames:
        # 检查分片文件是否存在
        if not os.path.exists(shard_file):
            # 如果不存在，抛出错误提示
            raise ValueError(
                f"{shards_path} does not appear to have a file named {shard_file} which is "
                "required according to the checkpoint index."
            )

# 获取检查点分片文件的函数定义
def _get_checkpoint_shard_files(
    pretrained_model_name_or_path,  # 预训练模型的名称或路径
    index_filename,  # 索引文件名
    cache_dir=None,  # 缓存目录，默认为 None
    proxies=None,  # 代理设置，默认为 None
    # 设置是否仅使用本地文件，默认为 False
    local_files_only=False,
    # 设置访问令牌，默认为 None，表示不使用令牌
    token=None,
    # 设置用户代理字符串，默认为 None
    user_agent=None,
    # 设置修订版号，默认为 None
    revision=None,
    # 设置子文件夹路径，默认为空字符串
    subfolder="",
):
    """
    对于给定的模型：

    - 如果 `pretrained_model_name_or_path` 是 Hub 上的模型 ID，则下载并缓存所有分片的检查点
    - 返回所有分片的路径列表，以及一些元数据。

    有关每个参数的描述，请参见 [`PreTrainedModel.from_pretrained`]。 `index_filename` 是索引的完整路径
    （如果 `pretrained_model_name_or_path` 是 Hub 上的模型 ID，则下载并缓存）。
    """
    # 检查索引文件是否存在，如果不存在则抛出错误
    if not os.path.isfile(index_filename):
        raise ValueError(f"Can't find a checkpoint index ({index_filename}) in {pretrained_model_name_or_path}.")

    # 打开索引文件并读取内容，解析为 JSON 格式
    with open(index_filename, "r") as f:
        index = json.loads(f.read())

    # 获取权重映射中的所有原始分片文件名，并去重后排序
    original_shard_filenames = sorted(set(index["weight_map"].values()))
    # 获取分片元数据
    sharded_metadata = index["metadata"]
    # 将所有检查点键的列表添加到元数据中
    sharded_metadata["all_checkpoint_keys"] = list(index["weight_map"].keys())
    # 复制权重映射到元数据中
    sharded_metadata["weight_map"] = index["weight_map"].copy()
    # 构建分片的路径
    shards_path = os.path.join(pretrained_model_name_or_path, subfolder)

    # 首先处理本地文件夹
    if os.path.isdir(pretrained_model_name_or_path):
        # 检查本地是否存在所需的分片
        _check_if_shards_exist_locally(
            pretrained_model_name_or_path, subfolder=subfolder, original_shard_filenames=original_shard_filenames
        )
        # 返回分片路径和分片元数据
        return shards_path, sharded_metadata

    # 此时 pretrained_model_name_or_path 是 Hub 上的模型标识符
    # 设置允许的文件模式为原始分片文件名
    allow_patterns = original_shard_filenames
    # 如果提供了子文件夹，则更新允许的文件模式
    if subfolder is not None:
        allow_patterns = [os.path.join(subfolder, p) for p in allow_patterns]

    # 定义需要忽略的文件模式
    ignore_patterns = ["*.json", "*.md"]
    # 如果不是仅使用本地文件
    if not local_files_only:
        # `model_info` 调用必须受到上述条件的保护
        model_files_info = model_info(pretrained_model_name_or_path, revision=revision)
        # 遍历原始分片文件名
        for shard_file in original_shard_filenames:
            # 检查当前分片文件是否在模型文件信息中存在
            shard_file_present = any(shard_file in k.rfilename for k in model_files_info.siblings)
            # 如果分片文件不存在，则抛出环境错误
            if not shard_file_present:
                raise EnvironmentError(
                    f"{shards_path} 不存在名为 {shard_file} 的文件，这是根据检查点索引所需的。"
                )

        try:
            # 从 URL 加载
            cached_folder = snapshot_download(
                pretrained_model_name_or_path,  # 要下载的模型路径
                cache_dir=cache_dir,  # 缓存目录
                proxies=proxies,  # 代理设置
                local_files_only=local_files_only,  # 是否仅使用本地文件
                token=token,  # 授权令牌
                revision=revision,  # 版本信息
                allow_patterns=allow_patterns,  # 允许的文件模式
                ignore_patterns=ignore_patterns,  # 忽略的文件模式
                user_agent=user_agent,  # 用户代理信息
            )
            # 如果指定了子文件夹，则更新缓存文件夹路径
            if subfolder is not None:
                cached_folder = os.path.join(cached_folder, subfolder)

        # 已经在获取索引时处理了 RepositoryNotFoundError 和 RevisionNotFoundError，
        # 所以这里不需要捕获它们。也处理了 EntryNotFoundError。
        except HTTPError as e:
            # 如果无法连接到指定的端点，则抛出环境错误
            raise EnvironmentError(
                f"我们无法连接到 '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' 来加载 {pretrained_model_name_or_path}。请检查您的互联网连接后重试。"
            ) from e

    # 如果 `local_files_only=True`，则 `cached_folder` 可能不包含所有分片文件
    elif local_files_only:
        # 检查本地是否存在所有分片
        _check_if_shards_exist_locally(
            local_dir=cache_dir,  # 本地目录
            subfolder=subfolder,  # 子文件夹
            original_shard_filenames=original_shard_filenames  # 原始分片文件名列表
        )
        # 如果指定了子文件夹，则更新缓存文件夹路径
        if subfolder is not None:
            cached_folder = os.path.join(cached_folder, subfolder)

    # 返回缓存文件夹和分片元数据
    return cached_folder, sharded_metadata
# 定义一个混合类，用于将模型、调度器或管道推送到 Hugging Face Hub
class PushToHubMixin:
    """
    A Mixin to push a model, scheduler, or pipeline to the Hugging Face Hub.
    """

    # 定义一个私有方法，用于上传指定文件夹中的所有文件
    def _upload_folder(
        self,
        working_dir: Union[str, os.PathLike],  # 工作目录，包含待上传的文件
        repo_id: str,                           # 目标仓库的 ID
        token: Optional[str] = None,           # 可选的认证令牌
        commit_message: Optional[str] = None,  # 可选的提交信息
        create_pr: bool = False,                # 是否创建拉取请求
    ):
        """
        Uploads all files in `working_dir` to `repo_id`.
        """
        # 如果未提供提交信息，则根据类名生成默认提交信息
        if commit_message is None:
            if "Model" in self.__class__.__name__:
                commit_message = "Upload model"  # 如果是模型类，设置默认信息
            elif "Scheduler" in self.__class__.__name__:
                commit_message = "Upload scheduler"  # 如果是调度器类，设置默认信息
            else:
                commit_message = f"Upload {self.__class__.__name__}"  # 否则，使用类名作为提交信息

        # 记录上传文件的日志信息
        logger.info(f"Uploading the files of {working_dir} to {repo_id}.")
        # 调用 upload_folder 函数上传文件，并返回其结果
        return upload_folder(
            repo_id=repo_id,                    # 目标仓库 ID
            folder_path=working_dir,            # 待上传的文件夹路径
            token=token,                        # 认证令牌
            commit_message=commit_message,      # 提交信息
            create_pr=create_pr                 # 是否创建拉取请求
        )

    # 定义一个公共方法，用于将文件推送到 Hugging Face Hub
    def push_to_hub(
        self,
        repo_id: str,                           # 目标仓库的 ID
        commit_message: Optional[str] = None,  # 可选的提交信息
        private: Optional[bool] = None,        # 可选，是否将仓库设置为私有
        token: Optional[str] = None,           # 可选的认证令牌
        create_pr: bool = False,                # 是否创建拉取请求
        safe_serialization: bool = True,        # 是否安全序列化
        variant: Optional[str] = None,          # 可选的变体参数
    ) -> str:  # 定义函数返回值类型为字符串
        """
        Upload model, scheduler, or pipeline files to the 🤗 Hugging Face Hub.  # 函数文档字符串，说明功能

        Parameters:  # 参数说明部分
            repo_id (`str`):  # 仓库 ID，类型为字符串
                The name of the repository you want to push your model, scheduler, or pipeline files to. It should
                contain your organization name when pushing to an organization. `repo_id` can also be a path to a local
                directory.  # 描述 repo_id 的用途和格式
            commit_message (`str`, *optional*):  # 可选参数，类型为字符串
                Message to commit while pushing. Default to `"Upload {object}".`  # 提交消息的默认值
            private (`bool`, *optional*):  # 可选参数，类型为布尔值
                Whether or not the repository created should be private.  # 是否创建私有仓库
            token (`str`, *optional*):  # 可选参数，类型为字符串
                The token to use as HTTP bearer authorization for remote files. The token generated when running
                `huggingface-cli login` (stored in `~/.huggingface`).  # 说明 token 的用途
            create_pr (`bool`, *optional*, defaults to `False`):  # 可选参数，类型为布尔值，默认值为 False
                Whether or not to create a PR with the uploaded files or directly commit.  # 是否创建 PR
            safe_serialization (`bool`, *optional*, defaults to `True`):  # 可选参数，类型为布尔值，默认值为 True
                Whether or not to convert the model weights to the `safetensors` format.  # 是否使用安全序列化格式
            variant (`str`, *optional*):  # 可选参数，类型为字符串
                If specified, weights are saved in the format `pytorch_model.<variant>.bin`.  # 权重保存格式

        Examples:  # 示例说明部分

        ```py
        from diffusers import UNet2DConditionModel  # 从 diffusers 导入 UNet2DConditionModel

        unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2", subfolder="unet")  # 从预训练模型加载 UNet

        # Push the `unet` to your namespace with the name "my-finetuned-unet".  # 推送到个人命名空间
        unet.push_to_hub("my-finetuned-unet")  # 将 unet 推送到指定名称的仓库

        # Push the `unet` to an organization with the name "my-finetuned-unet".  # 推送到组织
        unet.push_to_hub("your-org/my-finetuned-unet")  # 将 unet 推送到指定组织的仓库
        ```
        """  # 结束文档字符串
        repo_id = create_repo(repo_id, private=private, token=token, exist_ok=True).repo_id  # 创建仓库并获取仓库 ID

        # Create a new empty model card and eventually tag it  # 创建新的模型卡片并可能添加标签
        model_card = load_or_create_model_card(repo_id, token=token)  # 加载或创建模型卡片
        model_card = populate_model_card(model_card)  # 填充模型卡片信息

        # Save all files.  # 保存所有文件
        save_kwargs = {"safe_serialization": safe_serialization}  # 设置保存文件的参数
        if "Scheduler" not in self.__class__.__name__:  # 检查当前类名是否包含 "Scheduler"
            save_kwargs.update({"variant": variant})  # 如果不包含，则添加 variant 参数

        with tempfile.TemporaryDirectory() as tmpdir:  # 创建临时目录
            self.save_pretrained(tmpdir, **save_kwargs)  # 将模型保存到临时目录

            # Update model card if needed:  # 如果需要，更新模型卡片
            model_card.save(os.path.join(tmpdir, "README.md"))  # 将模型卡片保存为 README.md 文件

            return self._upload_folder(  # 上传临时目录中的文件
                tmpdir,  # 临时目录路径
                repo_id,  # 仓库 ID
                token=token,  # 认证 token
                commit_message=commit_message,  # 提交消息
                create_pr=create_pr,  # 是否创建 PR
            )  # 返回上传结果
```