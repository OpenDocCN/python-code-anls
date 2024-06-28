# `.\models\realm\configuration_realm.py`

```py
# coding=utf-8
# Copyright 2022 The REALM authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" REALM model configuration."""

from ...configuration_utils import PretrainedConfig  # 导入预训练配置基类
from ...utils import logging  # 导入日志工具


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

REALM_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "google/realm-cc-news-pretrained-embedder": (
        "https://huggingface.co/google/realm-cc-news-pretrained-embedder/resolve/main/config.json"
    ),
    "google/realm-cc-news-pretrained-encoder": (
        "https://huggingface.co/google/realm-cc-news-pretrained-encoder/resolve/main/config.json"
    ),
    "google/realm-cc-news-pretrained-scorer": (
        "https://huggingface.co/google/realm-cc-news-pretrained-scorer/resolve/main/config.json"
    ),
    "google/realm-cc-news-pretrained-openqa": (
        "https://huggingface.co/google/realm-cc-news-pretrained-openqa/aresolve/main/config.json"
    ),
    "google/realm-orqa-nq-openqa": "https://huggingface.co/google/realm-orqa-nq-openqa/resolve/main/config.json",
    "google/realm-orqa-nq-reader": "https://huggingface.co/google/realm-orqa-nq-reader/resolve/main/config.json",
    "google/realm-orqa-wq-openqa": "https://huggingface.co/google/realm-orqa-wq-openqa/resolve/main/config.json",
    "google/realm-orqa-wq-reader": "https://huggingface.co/google/realm-orqa-wq-reader/resolve/main/config.json",
    # See all REALM models at https://huggingface.co/models?filter=realm
}

class RealmConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of

    1. [`RealmEmbedder`]
    2. [`RealmScorer`]
    3. [`RealmKnowledgeAugEncoder`]
    4. [`RealmRetriever`]
    5. [`RealmReader`]
    6. [`RealmForOpenQA`]

    It is used to instantiate an REALM model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the REALM
    [google/realm-cc-news-pretrained-embedder](https://huggingface.co/google/realm-cc-news-pretrained-embedder)
    architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Example:

    ```
    >>> from transformers import RealmConfig, RealmEmbedder

    >>> # Initializing a REALM realm-cc-news-pretrained-* style configuration
    >>> configuration = RealmConfig()
    ```
    # 使用给定的配置初始化一个模型（具有随机权重），使用 google/realm-cc-news-pretrained-embedder 风格的配置
    model = RealmEmbedder(configuration)

    # 访问模型的配置信息
    configuration = model.config
```