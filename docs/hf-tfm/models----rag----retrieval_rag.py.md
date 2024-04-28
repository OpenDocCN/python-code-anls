# `.\transformers\models\rag\retrieval_rag.py`

```py
# coding=utf-8
# 声明编码格式为 utf-8
# 版权声明
# Copyright 2020, The RAG Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""RAG Retriever model implementation."""

import os
import pickle
import time
from typing import Iterable, List, Optional, Tuple
# 导入必要的库

import numpy as np
# 导入 numpy 库

from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_utils_base import BatchEncoding
from ...utils import cached_file, is_datasets_available, is_faiss_available, logging, requires_backends, strtobool
from .configuration_rag import RagConfig
from .tokenization_rag import RagTokenizer
# 从不同文件和模块中导入函数和类

if is_datasets_available():
    from datasets import Dataset, load_dataset, load_from_disk
# 如果 datasets 可用，导入相关函数和类

if is_faiss_available():
    import faiss
# 如果 faiss 可用，导入 faiss

logger = logging.get_logger(__name__)
# 获取 logger 实例

LEGACY_INDEX_PATH = "https://storage.googleapis.com/huggingface-nlp/datasets/wiki_dpr/"
# 设置 legacy 索引的远程路径

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
    # 获取文档字典的方法，传入文档的索引，返回标题和文本的字典列表

    def get_top_docs(self, question_hidden_states: np.ndarray, n_docs=5) -> Tuple[np.ndarray, np.ndarray]:
        """
        For each query in the batch, retrieves `n_docs` documents.

        Args:
            question_hidden_states (`np.ndarray` of shape `(batch_size, vector_size)`):
                An array of query vectors.
            n_docs (`int`):
                The number of docs retrieved per query.

        Returns:
            `np.ndarray` of shape `(batch_size, n_docs)`: A tensor of indices of retrieved documents. `np.ndarray` of
            shape `(batch_size, vector_size)`: A tensor of vector representations of retrieved documents.
        """
        raise NotImplementedError
    # 获取前 n_docs 个文档的方法，传入查询向量，返回查询到的文档的索引和向量表示

    def is_initialized(self):
        """
        Returns `True` if index is already initialized.
        """
        raise NotImplementedError
    # 检查索引是否已初始化的方法

    def init_index(self):
        """
        A function responsible for loading the index into memory. Should be called only once per training run of a RAG
        model. E.g. if the model is trained on multiple GPUs in a distributed setup, only one of the workers will load
        the index.
        """
        raise NotImplementedError
    # 负责将索引加载到内存中的方法，每次 RAG 模型的训练运行中只能调用一次
class LegacyIndex(Index):
    """
    An index which can be deserialized from the files built using https://github.com/facebookresearch/DPR. We use
    default faiss index parameters as specified in that repository.

    Args:
        vector_size (`int`):
            The dimension of indexed vectors.
        index_path (`str`):
            A path to a *directory* containing index files compatible with [`~models.rag.retrieval_rag.LegacyIndex`]
    """

    # 定义索引文件名常量
    INDEX_FILENAME = "hf_bert_base.hnswSQ8_correct_phi_128.c_index"
    # 定义段落文件名常量
    PASSAGE_FILENAME = "psgs_w100.tsv.pkl"

    def __init__(self, vector_size, index_path):
        self.index_id_to_db_id = []  # 初始化空列表
        self.index_path = index_path  # 初始化索引文件路径
        self.passages = self._load_passages()  # 加载段落数据
        self.vector_size = vector_size  # 初始化向量维度
        self.index = None  # 初始化索引对象
        self._index_initialized = False  # 初始化索引初始化标志为假

    def _resolve_path(self, index_path, filename):
        is_local = os.path.isdir(index_path)  # 判断路径是否本地
        try:
            # 从 URL 或缓存加载文件
            resolved_archive_file = cached_file(index_path, filename)
        except EnvironmentError:
            msg = (
                f"Can't load '{filename}'. Make sure that:\n\n"
                f"- '{index_path}' is a correct remote path to a directory containing a file named {filename}\n\n"
                f"- or '{index_path}' is the correct path to a directory containing a file named {filename}.\n\n"
            )
            raise EnvironmentError(msg)  # 处理加载异常
        if is_local:
            logger.info(f"loading file {resolved_archive_file}")  # 记录日志：加载本地文件
        else:
            logger.info(f"loading file {filename} from cache at {resolved_archive_file}")  # 记录日志：从缓存加载文件
        return resolved_archive_file  # 返回解析后的文件路径

    def _load_passages(self):
        logger.info(f"Loading passages from {self.index_path}")  # 记录日志：从索引路径加载段落
        passages_path = self._resolve_path(self.index_path, self.PASSAGE_FILENAME)  # 解析段落文件路径
        if not strtobool(os.environ.get("TRUST_REMOTE_CODE", "False")):
            raise ValueError(
                "This part uses `pickle.load` which is insecure and will execute arbitrary code that is potentially "
                "malicious. It's recommended to never unpickle data that could have come from an untrusted source, or "
                "that could have been tampered with. If you already verified the pickle data and decided to use it, "
                "you can set the environment variable `TRUST_REMOTE_CODE` to `True` to allow it."
            )  # 抛出异常：加载数据不安全
        with open(passages_path, "rb") as passages_file:
            passages = pickle.load(passages_file)  # 从文件中加载段落数据
        return passages  # 返回加载的段落数据
    # 从索引文件中反序列化索引数据
    def _deserialize_index(self):
        # 记录日志，显示正在从指定路径加载索引
        logger.info(f"Loading index from {self.index_path}")
        # 解析索引文件路径
        resolved_index_path = self._resolve_path(self.index_path, self.INDEX_FILENAME + ".index.dpr")
        # 读取并加载索引数据
        self.index = faiss.read_index(resolved_index_path)
        # 解析元数据文件路径
        resolved_meta_path = self._resolve_path(self.index_path, self.INDEX_FILENAME + ".index_meta.dpr")
        # 如果未设置信任远程代码环境变量，抛出 ValueError 异常提醒用户不安全
        if not strtobool(os.environ.get("TRUST_REMOTE_CODE", "False")):
            raise ValueError(
                "This part uses `pickle.load` which is insecure and will execute arbitrary code that is potentially "
                "malicious. It's recommended to never unpickle data that could have come from an untrusted source, or "
                "that could have been tampered with. If you already verified the pickle data and decided to use it, "
                "you can set the environment variable `TRUST_REMOTE_CODE` to `True` to allow it."
            )
        # 以二进制模式打开元数据文件，加载索引 ID 到数据库 ID 的映射关系
        with open(resolved_meta_path, "rb") as metadata_file:
            self.index_id_to_db_id = pickle.load(metadata_file)
        # 断言反序列化后的索引 ID 到数据库 ID 的映射关系数量与 faiss 索引大小相匹配
        assert (
            len(self.index_id_to_db_id) == self.index.ntotal
        ), "Deserialized index_id_to_db_id should match faiss index size"

    # 检查索引是否已初始化
    def is_initialized(self):
        return self._index_initialized

    # 初始化索引
    def init_index(self):
        # 创建 HNSWFlat 索引对象
        index = faiss.IndexHNSWFlat(self.vector_size + 1, 512)
        # 设置 HNSW 索引的搜索参数
        index.hnsw.efSearch = 128
        index.hnsw.efConstruction = 200
        # 将创建的索引对象赋值给类属性
        self.index = index
        # 反序列化索引
        self._deserialize_index()
        # 标记索引已初始化
        self._index_initialized = True

    # 获取文档字典
    def get_doc_dicts(self, doc_ids: np.array):
        # 存储文档列表
        doc_list = []
        # 遍历文档 ID 列表
        for doc_ids_i in doc_ids:
            # 转换文档 ID 为字符串形式，并获取对应的文档内容
            ids = [str(int(doc_id)) for doc_id in doc_ids_i]
            docs = [self.passages[doc_id] for doc_id in ids]
            # 将文档内容添加到文档列表中
            doc_list.append(docs)
        # 存储文档字典列表
        doc_dicts = []
        # 遍历文档列表
        for docs in doc_list:
            # 创建文档字典，包含标题和文本内容
            doc_dict = {}
            doc_dict["title"] = [doc[1] for doc in docs]
            doc_dict["text"] = [doc[0] for doc in docs]
            # 将文档字典添加到列表中
            doc_dicts.append(doc_dict)
        # 返回文档字典列表
        return doc_dicts

    # 获取前 N 个最相关的文档
    def get_top_docs(self, question_hidden_states: np.ndarray, n_docs=5) -> Tuple[np.ndarray, np.ndarray]:
        # 创建辅助维度数组
        aux_dim = np.zeros(len(question_hidden_states), dtype="float32").reshape(-1, 1)
        # 合并问题隐藏状态和辅助维度，构成查询向量
        query_nhsw_vectors = np.hstack((question_hidden_states, aux_dim))
        # 使用索引对象搜索前 N 个最相关的文档
        _, docs_ids = self.index.search(query_nhsw_vectors, n_docs)
        # 获取文档向量和对应的数据库 ID
        vectors = [[self.index.reconstruct(int(doc_id))[:-1] for doc_id in doc_ids] for doc_ids in docs_ids]
        ids = [[int(self.index_id_to_db_id[doc_id]) for doc_id in doc_ids] for doc_ids in docs_ids]
        # 返回数据库 ID 和文档向量
        return np.array(ids), np.array(vectors)
# HFIndexBase类继承自Index类, 用于构建基于Hugging Face Datasets的索引
class HFIndexBase(Index):
    # 初始化函数, 传入向量维度(vector_size), 数据集(dataset)和是否已初始化索引(index_initialized)
    def __init__(self, vector_size, dataset, index_initialized=False):
        # 将传入参数保存为类属性
        self.vector_size = vector_size
        self.dataset = dataset
        self._index_initialized = index_initialized
        # 检查数据集格式是否正确
        self._check_dataset_format(with_index=index_initialized)
        # 将数据集设置为numpy格式, 并指定列为["embeddings"], 输出所有列, 数据类型为float32
        dataset.set_format("numpy", columns=["embeddings"], output_all_columns=True, dtype="float32")

    # 检查数据集格式是否正确
    def _check_dataset_format(self, with_index: bool):
        # 如果数据集不是datasets.Dataset对象, 则触发异常
        if not isinstance(self.dataset, Dataset):
            raise ValueError(f"Dataset should be a datasets.Dataset object, but got {type(self.dataset)}")
        # 如果数据集不包含title、text和embeddings这三列, 则触发异常
        if len({"title", "text", "embeddings"} - set(self.dataset.column_names)) > 0:
            raise ValueError(
                "Dataset should be a dataset with the following columns: "
                "title (str), text (str) and embeddings (arrays of dimension vector_size), "
                f"but got columns {self.dataset.column_names}"
            )
        # 如果with_index为True且数据集中不存在embeddings索引, 则触发异常
        if with_index and "embeddings" not in self.dataset.list_indexes():
            raise ValueError(
                "Missing faiss index in the dataset. Make sure you called `dataset.add_faiss_index` to compute it "
                "or `dataset.load_faiss_index` to load one from the disk."
            )

    # 抽象方法, 初始化索引
    def init_index(self):
        raise NotImplementedError()

    # 判断索引是否已经初始化
    def is_initialized(self):
        return self._index_initialized

    # 根据文档ID获取对应的字典列表
    def get_doc_dicts(self, doc_ids: np.ndarray) -> List[dict]:
        return [self.dataset[doc_ids[i].tolist()] for i in range(doc_ids.shape[0])]

    # 根据问题的隐藏状态获取最相关的文档
    def get_top_docs(self, question_hidden_states: np.ndarray, n_docs=5) -> Tuple[np.ndarray, np.ndarray]:
        # 使用embeddings列在数据集上批量搜索与问题隐藏状态最相关的文档
        _, ids = self.dataset.search_batch("embeddings", question_hidden_states, n_docs)
        # 根据返回的文档索引列表获取对应的文档
        docs = [self.dataset[[i for i in indices if i >= 0]] for indices in ids]
        # 获取文档的embeddings
        vectors = [doc["embeddings"] for doc in docs]
        # 如果文档的数量少于n_docs, 则使用零向量补齐
        for i in range(len(vectors)):
            if len(vectors[i]) < n_docs:
                vectors[i] = np.vstack([vectors[i], np.zeros((n_docs - len(vectors[i]), self.vector_size))])
        # 返回文档的索引和embeddings, shapes分别为(batch_size, n_docs)和(batch_size, n_docs, d)
        return np.array(ids), np.array(vectors)


# CanonicalHFIndex类继承自HFIndexBase类, 是一个围绕在datasets.Dataset实例上的封装器类
class CanonicalHFIndex(HFIndexBase):
    """
    A wrapper around an instance of [`~datasets.Datasets`]. If `index_path` is set to `None`, we load the pre-computed
    index available with the [`~datasets.arrow_dataset.Dataset`], otherwise, we load the index from the indicated path
    on disk.
    """
    # 定义了一个 PassageEmbedding 类,用于加载和初始化用于dense retrieval的文本嵌入索引
    class PassageEmbedding:
        """
        参数:
            vector_size (`int`): 用于索引的文本 passage 嵌入的维度
            dataset_name (`str`, optional, defaults to `wiki_dpr`):
                要索引的数据集在 HuggingFace AWS存储桶上的标识符 (使用 `datasets.list_datasets()` 列出所有可用的数据集和 id)。
            dataset_split (`str`, optional, defaults to `train`)
                要加载的 `dataset` 的哪个分割。
            index_name (`str`, optional, defaults to `train`)
                与 `dataset` 关联的索引的索引名称。从 `index_path` 加载的索引将保存在此名称下。
            index_path (`str`, optional, defaults to `None`)
                磁盘上序列化的 faiss 索引的路径。
            use_dummy_dataset (`bool`, optional, defaults to `False`):
                如果为 True, 则使用数据集的虚拟配置进行测试。
        """
    
        def __init__(
            self,
            vector_size: int,
            dataset_name: str = "wiki_dpr",
            dataset_split: str = "train",
            index_name: Optional[str] = None,
            index_path: Optional[str] = None,
            use_dummy_dataset=False,
        ):
            # 如果既没有提供 index_name 也没有提供 index_path, 则引发错误
            if int(index_path is None) + int(index_name is None) != 1:
                raise ValueError("Please provide `index_name` or `index_path`.")
            # 设置实例属性
            self.dataset_name = dataset_name
            self.dataset_split = dataset_split
            self.index_name = index_name
            self.index_path = index_path
            self.use_dummy_dataset = use_dummy_dataset
            # 打印日志信息, 加载 dataset
            logger.info(f"Loading passages from {self.dataset_name}")
            dataset = load_dataset(
                self.dataset_name, with_index=False, split=self.dataset_split, dummy=self.use_dummy_dataset
            )
            # 调用父类的构造函数, 设置向量大小和数据集 (未初始化索引)
            super().__init__(vector_size, dataset, index_initialized=False)
    
        def init_index(self):
            # 如果提供了 index_path, 则从中加载索引
            if self.index_path is not None:
                logger.info(f"Loading index from {self.index_path}")
                self.dataset.load_faiss_index("embeddings", file=self.index_path)
            # 如果没有提供 index_path, 则从 dataset_name 和 index_name 加载索引
            else:
                logger.info(f"Loading index from {self.dataset_name} with index name {self.index_name}")
                self.dataset = load_dataset(
                    self.dataset_name,
                    with_embeddings=True,
                    with_index=True,
                    split=self.dataset_split,
                    index_name=self.index_name,
                    dummy=self.use_dummy_dataset,
                )
                self.dataset.set_format("numpy", columns=["embeddings"], output_all_columns=True)
            # 标记索引已初始化
            self._index_initialized = True
class CustomHFIndex(HFIndexBase):
    """
    A wrapper around an instance of [`~datasets.Datasets`]. The dataset and the index are both loaded from the
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

    def init_index(self):
        if not self.is_initialized():
            logger.info(f"Loading index from {self.index_path}")
            self.dataset.load_faiss_index("embeddings", file=self.index_path)
            self._index_initialized = True


class RagRetriever:
    """
    Retriever used to get documents from vector queries. It retrieves the documents embeddings as well as the documents
    contents, and it formats them to be used with a RagModel.

    Args:
        config ([`RagConfig`]):
            The configuration of the RAG model this Retriever is used with. Contains parameters indicating which
            `Index` to build. You can load your own custom dataset with `config.index_name="custom"` or use a canonical
            one (default) from the datasets library with `config.index_name="wiki_dpr"` for example.
        question_encoder_tokenizer ([`PreTrainedTokenizer`]):
            The tokenizer that was used to tokenize the question. It is used to decode the question and then use the
            generator_tokenizer.
        generator_tokenizer ([`PreTrainedTokenizer`]):
            The tokenizer used for the generator part of the RagModel.
        index ([`~models.rag.retrieval_rag.Index`], optional, defaults to the one defined by the configuration):
            If specified, use this index instead of the one built using the configuration

    Examples:

    ```python
    >>> # To load the default "wiki_dpr" dataset with 21M passages from wikipedia (index name is 'compressed' or 'exact')
    >>> from transformers import RagRetriever

    >>> retriever = RagRetriever.from_pretrained(


"""

# 定义了一个自定义的HFIndex类，继承自HFIndexBase类
# 一个HFIndex对象封装了一个[`~datasets.Datasets`]的实例，封装了数据集和索引加载到磁盘上的路径。
# 初始化方法需要传入三个参数：vector_size表示用于索引的向量维度，dataset指定了数据集的路径，包含了三列：title、text、embeddings，index_path指定了faiss索引的路径。索引初始化的时候，需要判断是否已经初始化过。
# load_from_disk方法可以加载已经保存在磁盘上的数据集和索引。
# init_index方法用于初始化索引，实现了加载索引的逻辑，通过调用dataset对象的load_faiss_index方法来加载已经保存在磁盘上的索引。
# RagRetriever类是用于根据向量查询获取文档的检索器，还会检索到文档的嵌入和内容，并将其格式化以便与RagModel一起使用。
# 初始化方法有4个参数：config是RAG模型的配置，question_encoder_tokenizer是用于解码问题的分词器，generator_tokenizer用于RagModel的生成器部分的分词器，index是用于构建索引的Index对象。
# 该类还包含一个静态方法load_from_pretrained，通过指定不同的index_name参数可以选择加载不同训练好的RagRetriever。
    # 初始化一个RagRetriever类对象，参数包括模型名称和数据集名称等
    """初始化一个RagRetriever类对象，参数包括模型名称和数据集名称等"""
    
    # 从transformers库中导入RagRetriever类
    """从transformers库中导入RagRetriever类"""
    
    # 加载自己构建的使用datasets库构建的索引数据集。在examples/rag/use_own_knowledge_dataset.py中有构建索引数据集的更多信息
    """加载自己构建的使用datasets库构建的索引数据集。在examples/rag/use_own_knowledge_dataset.py中有构建索引数据集的更多信息"""
    
    # 从transformers库中导入RagRetriever类
    """从transformers库中导入RagRetriever类"""
    
    # dataset必须是具有"title", "text"和"embeddings"列的datasets.Datasets对象，并且必须具有faiss索引
    """dataset必须是具有"title", "text"和"embeddings"列的datasets.Datasets对象，并且必须具有faiss索引"""
    
    # 加载自己构建的使用datasets库构建的保存在磁盘上的索引数据集。在examples/rag/use_own_knowledge_dataset.py中有更多信息
    """加载自己构建的使用datasets库构建的保存在磁盘上的索引数据集。在examples/rag/use_own_knowledge_dataset.py中有更多信息"""
    
    # 从transformers库中导入RagRetriever类
    """从transformers库中导入RagRetriever类"""
    
    # 初始化RagRetriever类对象，参数包括模型名称、索引名称、数据集路径和索引路径等
    """初始化RagRetriever类对象，参数包括模型名称、索引名称、数据集路径和索引路径等"""
    
    # 加载最初为Rag论文构建的遗留索引
    """加载最初为Rag论文构建的遗留索引"""
    
    # 从transformers库中导入RagRetriever类
    """从transformers库中导入RagRetriever类"""
    
    # 初始化一个RagRetriever类对象，参数包括模型名称和索引名称等
    """初始化一个RagRetriever类对象，参数包括模型名称和索引名称等"""
    
    # 静态方法
    """静态方法"""
    # 根据配置参数构建索引对象
    def _build_index(config):
        # 如果索引名是"legacy"，则创建一个LegacyIndex对象
        if config.index_name == "legacy":
            return LegacyIndex(
                config.retrieval_vector_size,
                config.index_path or LEGACY_INDEX_PATH,
            )
        # 如果索引名是"custom"，则从磁盘加载CustomHFIndex对象
        elif config.index_name == "custom":
            return CustomHFIndex.load_from_disk(
                vector_size=config.retrieval_vector_size,
                dataset_path=config.passages_path,
                index_path=config.index_path,
            )
        # 否则，创建一个CanonicalHFIndex对象
        else:
            return CanonicalHFIndex(
                vector_size=config.retrieval_vector_size,
                dataset_name=config.dataset,
                dataset_split=config.dataset_split,
                index_name=config.index_name,
                index_path=config.index_path,
                use_dummy_dataset=config.use_dummy_dataset,
            )

    # 根据预训练的模型名称或路径加载Retriever对象
    @classmethod
    def from_pretrained(cls, retriever_name_or_path, indexed_dataset=None, **kwargs):
        # 检查依赖的后端库是否安装
        requires_backends(cls, ["datasets", "faiss"])
        # 从预训练的模型名称或路径加载配置参数，若未提供则使用默认配置
        config = kwargs.pop("config", None) or RagConfig.from_pretrained(retriever_name_or_path, **kwargs)
        # 从预训练的模型名称或路径加载RagTokenizer对象
        rag_tokenizer = RagTokenizer.from_pretrained(retriever_name_or_path, config=config)
        # 获取问题编码器和生成器的Tokenizer
        question_encoder_tokenizer = rag_tokenizer.question_encoder
        generator_tokenizer = rag_tokenizer.generator
        if indexed_dataset is not None:
            # 如果提供了预先建立好的索引数据集，使用"custom"作为索引名，构建CustomHFIndex对象
            config.index_name = "custom"
            index = CustomHFIndex(config.retrieval_vector_size, indexed_dataset)
        else:
            # 否则根据配置参数构建索引对象
            index = cls._build_index(config)
        # 返回Retriever对象
        return cls(
            config,
            question_encoder_tokenizer=question_encoder_tokenizer,
            generator_tokenizer=generator_tokenizer,
            index=index,
        )

    # 将Retriever对象保存到磁盘
    def save_pretrained(self, save_directory):
        if isinstance(self.index, CustomHFIndex):
            if self.config.index_path is None:
                # 如果索引路径为空，则将索引数据保存到hf_dataset_index.faiss文件中，并更新配置参数���索引路径
                index_path = os.path.join(save_directory, "hf_dataset_index.faiss")
                self.index.dataset.get_index("embeddings").save(index_path)
                self.config.index_path = index_path
            if self.config.passages_path is None:
                # 如果索引数据集路径为空，则将索引数据集保存到hf_dataset文件夹中，并更新配置参数的数据集路径
                passages_path = os.path.join(save_directory, "hf_dataset")
                # 目前数据集不支持带有索引的保存到磁盘
                faiss_index = self.index.dataset._indexes.pop("embeddings")
                self.index.dataset.save_to_disk(passages_path)
                self.index.dataset._indexes["embeddings"] = faiss_index
                self.config.passages_path = passages_path
        # 保存配置参数到磁盘
        self.config.save_pretrained(save_directory)
        # 保存RagTokenizer对象到磁盘
        rag_tokenizer = RagTokenizer(
            question_encoder=self.question_encoder_tokenizer,
            generator=self.generator_tokenizer,
        )
        rag_tokenizer.save_pretrained(save_directory)
    def init_retrieval(self):
        """
        Retriever初始化函数。将索引加载到内存中。
        """

        # 初始化检索器
        logger.info("initializing retrieval")
        self.index.init_index()

    def postprocess_docs(self, docs, input_strings, prefix, n_docs, return_tensors=None):
        r"""
        后处理检索到的`docs`并将它们与`input_strings`组合。

        Args:
            docs  (`dict`):
                检索到的文档。
            input_strings (`str`):
                由`preprocess_query`解码的输入字符串。
            prefix (`str`):
                添加在每个输入开头的前缀，通常与基于T5的模型一起使用。

        Return:
            `tuple(tensors)`: 由两个元素组成的元组：上下文化的`input_ids`和兼容的`attention_mask`。
        """

        def cat_input_and_doc(doc_title, doc_text, input_string, prefix):
            # TODO（Patrick）：如果我们训练更多RAG模型，我希望放置输入到首位以利用轻松截断
            # TODO（piktus）：更好地处理截断
            if doc_title.startswith('"'):
                doc_title = doc_title[1:]
            if doc_title.endswith('"'):
                doc_title = doc_title[:-1]
            if prefix is None:
                prefix = ""
            out = (prefix + doc_title + self.config.title_sep + doc_text + self.config.doc_sep + input_string).replace(
                "  ", " "
            )
            return out

        # 创建RAG输入字符串
        rag_input_strings = [
            cat_input_and_doc(
                docs[i]["title"][j],
                docs[i]["text"][j],
                input_strings[i],
                prefix,
            )
            for i in range(len(docs))
            for j in range(n_docs)
        ]

        # 使用生成器令牌化器对RAG输入字符串进行处理
        contextualized_inputs = self.generator_tokenizer.batch_encode_plus(
            rag_input_strings,
            max_length=self.config.max_combined_length,
            return_tensors=return_tensors,
            padding="max_length",
            truncation=True,
        )

        return contextualized_inputs["input_ids"], contextualized_inputs["attention_mask"]

    def _chunk_tensor(self, t: Iterable, chunk_size: int) -> List[Iterable]:
        # 将输入的可迭代对象`t`按照指定大小`chunk_size`进行分块处理，并返回分块后的列表
        return [t[i : i + chunk_size] for i in range(0, len(t), chunk_size)]
```py  
    def _main_retrieve(self, question_hidden_states: np.ndarray, n_docs: int) -> Tuple[np.ndarray, np.ndarray]:
        # 将问题隐藏状态分成批次以便处理
        question_hidden_states_batched = self._chunk_tensor(question_hidden_states, self.batch_size)
        ids_batched = []
        vectors_batched = []
        for question_hidden_states in question_hidden_states_batched:
            start_time = time.time()
            # 获取顶部文档
            ids, vectors = self.index.get_top_docs(question_hidden_states, n_docs)
            logger.debug(
                # 记录索引搜索所需的时间和批处理大小
                f"index search time: {time.time() - start_time} sec, batch size {question_hidden_states.shape}"
            )
            ids_batched.extend(ids)
            vectors_batched.extend(vectors)
        return (
            np.array(ids_batched),
            np.array(vectors_batched),
        )  # shapes (batch_size, n_docs) and (batch_size, n_docs, d)

    def retrieve(self, question_hidden_states: np.ndarray, n_docs: int) -> Tuple[np.ndarray, List[dict]]:
        """
        Retrieves documents for specified `question_hidden_states`.

        Args:
            question_hidden_states (`np.ndarray` of shape `(batch_size, vector_size)`):
                A batch of query vectors to retrieve with.
            n_docs (`int`):
                The number of docs retrieved per query.

        Return:
            `Tuple[np.ndarray, np.ndarray, List[dict]]`: A tuple with the following objects:

            - **retrieved_doc_embeds** (`np.ndarray` of shape `(batch_size, n_docs, dim)`) -- The retrieval embeddings
              of the retrieved docs per query.
            - **doc_ids** (`np.ndarray` of shape `(batch_size, n_docs)`) -- The ids of the documents in the index
            - **doc_dicts** (`List[dict]`): The `retrieved_doc_embeds` examples per query.
        """

        doc_ids, retrieved_doc_embeds = self._main_retrieve(question_hidden_states, n_docs)
        return retrieved_doc_embeds, doc_ids, self.index.get_doc_dicts(doc_ids)

    def set_ctx_encoder_tokenizer(self, ctx_encoder_tokenizer: PreTrainedTokenizer):
        # 用于端到端检索器训练
        self.ctx_encoder_tokenizer = ctx_encoder_tokenizer
        self.return_tokenized_docs = True

    def __call__(
        self,
        question_input_ids: List[List[int]],
        question_hidden_states: np.ndarray,
        prefix=None,
        n_docs=None,
        return_tensors=None,
```