# `.\models\rag\retrieval_rag.py`

```py
# coding=utf-8
# Copyright 2020, The RAG Authors and The HuggingFace Inc. team.
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
"""
RAG Retriever model implementation.
"""

import os
import pickle
import time
from typing import Iterable, List, Optional, Tuple

import numpy as np

from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_utils_base import BatchEncoding
from ...utils import cached_file, is_datasets_available, is_faiss_available, logging, requires_backends, strtobool
from .configuration_rag import RagConfig
from .tokenization_rag import RagTokenizer

# 如果datasets可用，则导入相关模块
if is_datasets_available():
    from datasets import Dataset, load_dataset, load_from_disk

# 如果faiss可用，则导入faiss模块
if is_faiss_available():
    import faiss

# 获取日志记录器
logger = logging.get_logger(__name__)

# Legacy索引路径
LEGACY_INDEX_PATH = "https://storage.googleapis.com/huggingface-nlp/datasets/wiki_dpr/"

class Index:
    """
    A base class for the Indices encapsulated by the [`RagRetriever`].
    """

    def get_doc_dicts(self, doc_ids: np.ndarray) -> List[dict]:
        """
        Returns a list of dictionaries, containing titles and text of the retrieved documents.

        Args:
            doc_ids (`np.ndarray` of shape `(batch_size, n_docs)`):
                A tensor of document indices.
        """
        raise NotImplementedError

    def get_top_docs(self, question_hidden_states: np.ndarray, n_docs=5) -> Tuple[np.ndarray, np.ndarray]:
        """
        For each query in the batch, retrieves `n_docs` documents.

        Args:
            question_hidden_states (`np.ndarray` of shape `(batch_size, vector_size)`):
                An array of query vectors.
            n_docs (`int`):
                The number of docs retrieved per query.

        Returns:
            `np.ndarray` of shape `(batch_size, n_docs)`: A tensor of indices of retrieved documents.
            `np.ndarray` of shape `(batch_size, vector_size)`: A tensor of vector representations of retrieved documents.
        """
        raise NotImplementedError

    def is_initialized(self):
        """
        Returns `True` if index is already initialized.
        """
        raise NotImplementedError

    def init_index(self):
        """
        A function responsible for loading the index into memory. Should be called only once per training run of a RAG
        model. E.g. if the model is trained on multiple GPUs in a distributed setup, only one of the workers will load
        the index.
        """
        raise NotImplementedError
    """
    一个可以从使用 https://github.com/facebookresearch/DPR 构建的文件中反序列化的索引。我们使用该仓库中指定的默认 faiss 索引参数。

    Args:
        vector_size (`int`):
            索引向量的维度。
        index_path (`str`):
            包含与 [`~models.rag.retrieval_rag.LegacyIndex`] 兼容的索引文件的 *目录* 路径。
    """

    # 索引文件名
    INDEX_FILENAME = "hf_bert_base.hnswSQ8_correct_phi_128.c_index"
    # 文章段落文件名
    PASSAGE_FILENAME = "psgs_w100.tsv.pkl"

    def __init__(self, vector_size, index_path):
        # 用于映射索引 ID 到数据库 ID 的空列表
        self.index_id_to_db_id = []
        # 索引文件路径
        self.index_path = index_path
        # 加载文章段落数据
        self.passages = self._load_passages()
        # 索引向量的维度
        self.vector_size = vector_size
        # 索引对象
        self.index = None
        # 索引是否已初始化标志
        self._index_initialized = False

    def _resolve_path(self, index_path, filename):
        # 判断索引路径是本地路径还是远程路径
        is_local = os.path.isdir(index_path)
        try:
            # 从 URL 或缓存中加载文件
            resolved_archive_file = cached_file(index_path, filename)
        except EnvironmentError:
            # 抛出加载错误信息
            msg = (
                f"Can't load '{filename}'. Make sure that:\n\n"
                f"- '{index_path}' is a correct remote path to a directory containing a file named {filename}\n\n"
                f"- or '{index_path}' is the correct path to a directory containing a file named {filename}.\n\n"
            )
            raise EnvironmentError(msg)
        # 打印加载信息，如果是本地路径则显示完整路径
        if is_local:
            logger.info(f"loading file {resolved_archive_file}")
        else:
            logger.info(f"loading file {filename} from cache at {resolved_archive_file}")
        # 返回解析后的文件路径
        return resolved_archive_file

    def _load_passages(self):
        # 打印从指定路径加载段落信息的日志
        logger.info(f"Loading passages from {self.index_path}")
        # 解析文章段落文件的路径
        passages_path = self._resolve_path(self.index_path, self.PASSAGE_FILENAME)
        # 如果未设置环境变量 TRUST_REMOTE_CODE 或其值为 False，则抛出安全性错误
        if not strtobool(os.environ.get("TRUST_REMOTE_CODE", "False")):
            raise ValueError(
                "This part uses `pickle.load` which is insecure and will execute arbitrary code that is potentially "
                "malicious. It's recommended to never unpickle data that could have come from an untrusted source, or "
                "that could have been tampered with. If you already verified the pickle data and decided to use it, "
                "you can set the environment variable `TRUST_REMOTE_CODE` to `True` to allow it."
            )
        # 使用 pickle 加载文章段落数据
        with open(passages_path, "rb") as passages_file:
            passages = pickle.load(passages_file)
        # 返回加载的段落数据
        return passages
    # 日志记录，显示正在从指定路径加载索引
    logger.info(f"Loading index from {self.index_path}")
    # 解析索引文件路径，生成完整路径名，包括索引文件名和扩展名
    resolved_index_path = self._resolve_path(self.index_path, self.INDEX_FILENAME + ".index.dpr")
    # 使用 Faiss 库从解析后的索引路径读取索引数据
    self.index = faiss.read_index(resolved_index_path)
    # 解析元数据文件路径，生成完整路径名，包括索引文件名和元数据文件扩展名
    resolved_meta_path = self._resolve_path(self.index_path, self.INDEX_FILENAME + ".index_meta.dpr")
    # 如果环境变量 TRUST_REMOTE_CODE 不为真，则抛出值错误异常，警告使用不安全的 pickle.load
    if not strtobool(os.environ.get("TRUST_REMOTE_CODE", "False")):
        raise ValueError(
            "This part uses `pickle.load` which is insecure and will execute arbitrary code that is potentially "
            "malicious. It's recommended to never unpickle data that could have come from an untrusted source, or "
            "that could have been tampered with. If you already verified the pickle data and decided to use it, "
            "you can set the environment variable `TRUST_REMOTE_CODE` to `True` to allow it."
        )
    # 使用二进制读取模式打开元数据文件，加载 self.index_id_to_db_id 字典数据
    with open(resolved_meta_path, "rb") as metadata_file:
        self.index_id_to_db_id = pickle.load(metadata_file)
    # 断言索引 id 到数据库 id 的映射长度应与 Faiss 索引的总数相同，用于验证数据一致性
    assert (
        len(self.index_id_to_db_id) == self.index.ntotal
    ), "Deserialized index_id_to_db_id should match faiss index size"
class HFIndexBase(Index):
    # HFIndexBase 类，继承自 Index 类，用于处理特定格式的数据集索引

    def __init__(self, vector_size, dataset, index_initialized=False):
        # 初始化方法，接受向量大小、数据集和索引初始化状态作为参数
        self.vector_size = vector_size
        self.dataset = dataset
        self._index_initialized = index_initialized
        # 检查数据集格式是否正确
        self._check_dataset_format(with_index=index_initialized)
        # 设置数据集格式为 numpy 格式，指定列为 embeddings，输出所有列，数据类型为 float32
        dataset.set_format("numpy", columns=["embeddings"], output_all_columns=True, dtype="float32")

    def _check_dataset_format(self, with_index: bool):
        # 检查数据集格式是否符合要求，包括是否为 datasets.Dataset 对象，列是否包含必需的 title、text 和 embeddings
        if not isinstance(self.dataset, Dataset):
            raise ValueError(f"Dataset should be a datasets.Dataset object, but got {type(self.dataset)}")
        if len({"title", "text", "embeddings"} - set(self.dataset.column_names)) > 0:
            raise ValueError(
                "Dataset should be a dataset with the following columns: "
                "title (str), text (str) and embeddings (arrays of dimension vector_size), "
                f"but got columns {self.dataset.column_names}"
            )
        # 如果需要索引但数据集中未包含 embeddings 索引，则引发异常
        if with_index and "embeddings" not in self.dataset.list_indexes():
            raise ValueError(
                "Missing faiss index in the dataset. Make sure you called `dataset.add_faiss_index` to compute it "
                "or `dataset.load_faiss_index` to load one from the disk."
            )

    def init_index(self):
        # 初始化索引的抽象方法，需要在子类中实现
        raise NotImplementedError()

    def is_initialized(self):
        # 返回索引是否已初始化的状态
        return self._index_initialized

    def get_doc_dicts(self, doc_ids: np.ndarray) -> List[dict]:
        # 根据文档的 ID 获取文档信息，返回一个字典列表
        return [self.dataset[doc_ids[i].tolist()] for i in range(doc_ids.shape[0])]

    def get_top_docs(self, question_hidden_states: np.ndarray, n_docs=5) -> Tuple[np.ndarray, np.ndarray]:
        # 根据问题的隐藏状态获取前 n_docs 个最相关的文档
        # 使用数据集的批量搜索功能，根据 embeddings 列和问题的隐藏状态进行搜索
        _, ids = self.dataset.search_batch("embeddings", question_hidden_states, n_docs)
        # 根据搜索结果获取对应的文档数据
        docs = [self.dataset[[i for i in indices if i >= 0]] for indices in ids]
        # 提取文档的 embeddings 向量
        vectors = [doc["embeddings"] for doc in docs]
        # 对于搜索结果数量不足 n_docs 的情况，用零向量填充
        for i in range(len(vectors)):
            if len(vectors[i]) < n_docs:
                vectors[i] = np.vstack([vectors[i], np.zeros((n_docs - len(vectors[i]), self.vector_size))])
        # 返回搜索结果的 IDs 和对应的 embeddings 向量
        return np.array(ids), np.array(vectors)  # shapes (batch_size, n_docs) and (batch_size, n_docs, d)


class CanonicalHFIndex(HFIndexBase):
    """
    A wrapper around an instance of [`~datasets.Datasets`]. If `index_path` is set to `None`, we load the pre-computed
    index available with the [`~datasets.arrow_dataset.Dataset`], otherwise, we load the index from the indicated path
    on disk.
    """
    # CanonicalHFIndex 类，继承自 HFIndexBase 类，是对 datasets.Datasets 的封装，支持加载预先计算的索引或从磁盘加载索引
    """
    Args:
        vector_size (`int`): the dimension of the passages embeddings used by the index
        dataset_name (`str`, optional, defaults to `wiki_dpr`):
            A dataset identifier of the indexed dataset on HuggingFace AWS bucket (list all available datasets and ids
            with `datasets.list_datasets()`).
        dataset_split (`str`, optional, defaults to `train`):
            Which split of the `dataset` to load.
        index_name (`str`, optional, defaults to `train`):
            The index_name of the index associated with the `dataset`. The index loaded from `index_path` will be saved
            under this name.
        index_path (`str`, optional, defaults to `None`):
            The path to the serialized faiss index on disk.
        use_dummy_dataset (`bool`, optional, defaults to `False`):
            If True, use the dummy configuration of the dataset for tests.
    """

    def __init__(
        self,
        vector_size: int,
        dataset_name: str = "wiki_dpr",
        dataset_split: str = "train",
        index_name: Optional[str] = None,
        index_path: Optional[str] = None,
        use_dummy_dataset: bool = False,
        dataset_revision=None,
    ):
        # Validate that either `index_name` or `index_path` is provided
        if int(index_path is None) + int(index_name is None) != 1:
            raise ValueError("Please provide `index_name` or `index_path`.")
        
        # Initialize instance variables with provided parameters
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self.index_name = index_name
        self.index_path = index_path
        self.use_dummy_dataset = use_dummy_dataset
        self.dataset_revision = dataset_revision
        
        # Log information about dataset loading
        logger.info(f"Loading passages from {self.dataset_name}")
        
        # Load the dataset using Hugging Face datasets library
        dataset = load_dataset(
            self.dataset_name,
            with_index=False,
            split=self.dataset_split,
            dummy=self.use_dummy_dataset,
            revision=self.dataset_revision,
        )
        
        # Call superclass initialization with vector size and loaded dataset
        super().__init__(vector_size, dataset, index_initialized=False)

    def init_index(self):
        # Initialize index based on provided `index_path` or `index_name`
        if self.index_path is not None:
            # Load index from specified file path
            logger.info(f"Loading index from {self.index_path}")
            self.dataset.load_faiss_index("embeddings", file=self.index_path)
        else:
            # Load index associated with `index_name` from dataset
            logger.info(f"Loading index from {self.dataset_name} with index name {self.index_name}")
            
            # Load dataset with embeddings and index
            self.dataset = load_dataset(
                self.dataset_name,
                with_embeddings=True,
                with_index=True,
                split=self.dataset_split,
                index_name=self.index_name,
                dummy=self.use_dummy_dataset,
                revision=self.dataset_revision,
            )
            
            # Set dataset format to numpy for compatibility
            self.dataset.set_format("numpy", columns=["embeddings"], output_all_columns=True)
        
        # Mark index initialization as completed
        self._index_initialized = True
class CustomHFIndex(HFIndexBase):
    """
    A wrapper around an instance of [`datasets.Datasets`]. The dataset and the index are both loaded from the
    indicated paths on disk.

    Args:
        vector_size (`int`): the dimension of the passages embeddings used by the index
        dataset_path (`str`):
            The path to the serialized dataset on disk. The dataset should have 3 columns: title (str), text (str) and
            embeddings (arrays of dimension vector_size)
        index_path (`str`)
            The path to the serialized faiss index on disk.
    """

    def __init__(self, vector_size: int, dataset, index_path=None):
        super().__init__(vector_size, dataset, index_initialized=index_path is None)
        self.index_path = index_path
        # 初始化函数，设置向量大小和数据集，并根据 index_path 是否为 None 来初始化索引状态

    @classmethod
    def load_from_disk(cls, vector_size, dataset_path, index_path):
        logger.info(f"Loading passages from {dataset_path}")
        if dataset_path is None or index_path is None:
            raise ValueError(
                "Please provide `dataset_path` and `index_path` after calling `dataset.save_to_disk(dataset_path)` "
                "and `dataset.get_index('embeddings').save(index_path)`."
            )
        dataset = load_from_disk(dataset_path)
        return cls(vector_size=vector_size, dataset=dataset, index_path=index_path)
        # 从磁盘加载数据集和索引，根据提供的路径信息，返回一个 CustomHFIndex 的实例

    def init_index(self):
        if not self.is_initialized():
            logger.info(f"Loading index from {self.index_path}")
            self.dataset.load_faiss_index("embeddings", file=self.index_path)
            self._index_initialized = True
            # 如果索引尚未初始化，则加载 faiss 索引文件到数据集，并设置索引初始化状态为 True
    # 定义一个名为 RagRetriever 的类，用于处理检索相关的功能
    class RagRetriever:

        # 初始化函数，接受配置参数、问题编码器的分词器、生成器的分词器、索引对象和初始化检索标志
        def __init__(self, config, question_encoder_tokenizer, generator_tokenizer, index=None, init_retrieval=True):
            # 标志是否初始化检索
            self._init_retrieval = init_retrieval
            # 要求必需的后端库
            requires_backends(self, ["datasets", "faiss"])
            # 调用父类初始化函数
            super().__init__()
            # 设置索引对象，如果未提供索引则调用内部方法构建索引
            self.index = index or self._build_index(config)
            # 设置生成器的分词器
            self.generator_tokenizer = generator_tokenizer
            # 设置问题编码器的分词器
            self.question_encoder_tokenizer = question_encoder_tokenizer

            # 设置文档数量
            self.n_docs = config.n_docs
            # 设置检索批处理大小
            self.batch_size = config.retrieval_batch_size

            # 存储配置参数
            self.config = config
            # 如果标志允许，则初始化检索
            if self._init_retrieval:
                self.init_retrieval()

            # 上下文编码器的分词器初始化为空
            self.ctx_encoder_tokenizer = None
            # 是否返回标记化的文档标志初始化为假
            self.return_tokenized_docs = False

        @staticmethod
    # 从给定的配置参数构建索引对象
    def _build_index(config):
        # 如果配置指定使用旧版索引，返回 LegacyIndex 实例
        if config.index_name == "legacy":
            return LegacyIndex(
                config.retrieval_vector_size,
                config.index_path or LEGACY_INDEX_PATH,
            )
        # 如果配置指定使用自定义索引，加载自定义索引数据并返回 CustomHFIndex 实例
        elif config.index_name == "custom":
            return CustomHFIndex.load_from_disk(
                vector_size=config.retrieval_vector_size,
                dataset_path=config.passages_path,
                index_path=config.index_path,
            )
        # 否则，返回 CanonicalHFIndex 实例
        else:
            return CanonicalHFIndex(
                vector_size=config.retrieval_vector_size,
                dataset_name=config.dataset,
                dataset_split=config.dataset_split,
                index_name=config.index_name,
                index_path=config.index_path,
                use_dummy_dataset=config.use_dummy_dataset,
                dataset_revision=config.dataset_revision,
            )

    @classmethod
    # 从预训练模型或路径中加载检索器实例
    def from_pretrained(cls, retriever_name_or_path, indexed_dataset=None, **kwargs):
        requires_backends(cls, ["datasets", "faiss"])
        # 加载配置信息，如果未提供则从预训练模型中加载
        config = kwargs.pop("config", None) or RagConfig.from_pretrained(retriever_name_or_path, **kwargs)
        # 加载 RAG 模型的分词器
        rag_tokenizer = RagTokenizer.from_pretrained(retriever_name_or_path, config=config)
        # 获取问题编码器和生成器的分词器
        question_encoder_tokenizer = rag_tokenizer.question_encoder
        generator_tokenizer = rag_tokenizer.generator
        # 如果提供了索引数据集，强制配置使用自定义索引，并创建 CustomHFIndex 实例
        if indexed_dataset is not None:
            config.index_name = "custom"
            index = CustomHFIndex(config.retrieval_vector_size, indexed_dataset)
        # 否则，根据配置构建索引对象
        else:
            index = cls._build_index(config)
        # 返回根据配置构建的检索器实例
        return cls(
            config,
            question_encoder_tokenizer=question_encoder_tokenizer,
            generator_tokenizer=generator_tokenizer,
            index=index,
        )

    # 将当前对象的预训练参数保存到指定目录
    def save_pretrained(self, save_directory):
        # 如果当前索引为 CustomHFIndex 类型
        if isinstance(self.index, CustomHFIndex):
            # 如果配置中索引路径为空，则保存索引数据到默认路径
            if self.config.index_path is None:
                index_path = os.path.join(save_directory, "hf_dataset_index.faiss")
                self.index.dataset.get_index("embeddings").save(index_path)
                self.config.index_path = index_path
            # 如果配置中 passages_path 为空，则保存数据集到默认路径
            if self.config.passages_path is None:
                passages_path = os.path.join(save_directory, "hf_dataset")
                # 由于当前版本的 datasets 不支持带有索引的 save_to_disk 操作，因此需执行此操作
                faiss_index = self.index.dataset._indexes.pop("embeddings")
                self.index.dataset.save_to_disk(passages_path)
                self.index.dataset._indexes["embeddings"] = faiss_index
                self.config.passages_path = passages_path
        # 将当前配置保存到指定目录
        self.config.save_pretrained(save_directory)
        # 初始化 RAG 分词器并保存到指定目录
        rag_tokenizer = RagTokenizer(
            question_encoder=self.question_encoder_tokenizer,
            generator=self.generator_tokenizer,
        )
        rag_tokenizer.save_pretrained(save_directory)
    def init_retrieval(self):
        """
        Retriever initialization function. It loads the index into memory.
        """

        # 记录初始化检索过程的日志信息
        logger.info("initializing retrieval")
        # 调用索引对象的初始化方法，加载索引到内存中
        self.index.init_index()

    def postprocess_docs(self, docs, input_strings, prefix, n_docs, return_tensors=None):
        r"""
        Postprocessing retrieved `docs` and combining them with `input_strings`.

        Args:
            docs  (`dict`):
                Retrieved documents.
            input_strings (`str`):
                Input strings decoded by `preprocess_query`.
            prefix (`str`):
                Prefix added at the beginning of each input, typically used with T5-based models.

        Return:
            `tuple(tensors)`: a tuple consisting of two elements: contextualized `input_ids` and a compatible
            `attention_mask`.
        """

        def cat_input_and_doc(doc_title, doc_text, input_string, prefix):
            # TODO(Patrick): if we train more RAG models, I want to put the input first to take advantage of effortless truncation
            # TODO(piktus): better handling of truncation
            # 如果文档标题以双引号开头，去除开头的双引号
            if doc_title.startswith('"'):
                doc_title = doc_title[1:]
            # 如果文档标题以双引号结尾，去除结尾的双引号
            if doc_title.endswith('"'):
                doc_title = doc_title[:-1]
            # 如果前缀为空，则置为空字符串
            if prefix is None:
                prefix = ""
            # 组装处理后的文本片段，包括标题、文本内容、输入字符串，中间用指定的分隔符分隔，并处理多余的空格
            out = (prefix + doc_title + self.config.title_sep + doc_text + self.config.doc_sep + input_string).replace(
                "  ", " "
            )
            return out

        # 构建 RAG 模型的输入字符串列表
        rag_input_strings = [
            cat_input_and_doc(
                docs[i]["title"][j],
                docs[i]["text"][j],
                input_strings[i],
                prefix,
            )
            for i in range(len(docs))  # 遍历每个文档
            for j in range(n_docs)      # 遍历每个文档的多个版本（如果有的话）
        ]

        # 使用生成器的 tokenizer 对输入字符串列表进行批量编码处理
        contextualized_inputs = self.generator_tokenizer.batch_encode_plus(
            rag_input_strings,
            max_length=self.config.max_combined_length,  # 指定最大长度
            return_tensors=return_tensors,                # 是否返回张量
            padding="max_length",                         # 填充到最大长度
            truncation=True,                              # 是否截断超出最大长度的部分
        )

        # 返回编码后的输入张量
        return contextualized_inputs["input_ids"], contextualized_inputs["attention_mask"]

    def _chunk_tensor(self, t: Iterable, chunk_size: int) -> List[Iterable]:
        # 将输入张量按照指定的块大小进行切片并返回切片后的列表
        return [t[i : i + chunk_size] for i in range(0, len(t), chunk_size)]
    def _main_retrieve(self, question_hidden_states: np.ndarray, n_docs: int) -> Tuple[np.ndarray, np.ndarray]:
        # 将查询向量按照设定的批次大小分块处理
        question_hidden_states_batched = self._chunk_tensor(question_hidden_states, self.batch_size)
        ids_batched = []
        vectors_batched = []
        for question_hidden_states in question_hidden_states_batched:
            # 记录开始时间
            start_time = time.time()
            # 使用索引对象获取每个查询向量的前 n_docs 个文档的 ids 和向量表示
            ids, vectors = self.index.get_top_docs(question_hidden_states, n_docs)
            # 打印索引搜索时间和当前批次大小
            logger.debug(
                f"index search time: {time.time() - start_time} sec, batch size {question_hidden_states.shape}"
            )
            # 将获取的 ids 和 vectors 扩展到批次级别的列表中
            ids_batched.extend(ids)
            vectors_batched.extend(vectors)
        return (
            np.array(ids_batched),
            np.array(vectors_batched),
        )  # shapes (batch_size, n_docs) and (batch_size, n_docs, d)

    def retrieve(self, question_hidden_states: np.ndarray, n_docs: int) -> Tuple[np.ndarray, List[dict]]:
        """
        为指定的 `question_hidden_states` 检索文档。

        Args:
            question_hidden_states (`np.ndarray` of shape `(batch_size, vector_size)`):
                要检索的查询向量的批次。
            n_docs (`int`):
                每个查询检索的文档数量。

        Return:
            `Tuple[np.ndarray, np.ndarray, List[dict]]`: 返回包含以下对象的元组：

            - **retrieved_doc_embeds** (`np.ndarray` of shape `(batch_size, n_docs, dim)`) -- 每个查询的检索嵌入的文档。
            - **doc_ids** (`np.ndarray` of shape `(batch_size, n_docs)`) -- 索引中文档的 ids。
            - **doc_dicts** (`List[dict]`): 每个查询的 `retrieved_doc_embeds` 示例。
        """

        # 使用 _main_retrieve 方法获取文档 ids 和检索嵌入
        doc_ids, retrieved_doc_embeds = self._main_retrieve(question_hidden_states, n_docs)
        # 返回检索嵌入、文档 ids 和获取的文档字典
        return retrieved_doc_embeds, doc_ids, self.index.get_doc_dicts(doc_ids)

    def set_ctx_encoder_tokenizer(self, ctx_encoder_tokenizer: PreTrainedTokenizer):
        # 用于端到端检索器训练中，设置上下文编码器的分词器
        self.ctx_encoder_tokenizer = ctx_encoder_tokenizer
        self.return_tokenized_docs = True

    def __call__(
        self,
        question_input_ids: List[List[int]],
        question_hidden_states: np.ndarray,
        prefix=None,
        n_docs=None,
        return_tensors=None,
):
```