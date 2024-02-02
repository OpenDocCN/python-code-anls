# AutoGPT源码解析 13

# `autogpts/autogpt/autogpt/memory/vector/utils.py`

这段代码定义了一个嵌入层（Embedding）和一个文本层（TText），并使用了来自`autogpt.config`的`Config`类。以下是代码的主要功能和组件：

1. 导入`logging`模块以创建一个名为`logger`的 logger。
2. 从`contextlib`模块的`suppress`函数中获取忽略输出（suppress=True）的上下文（context）。
3. 从`typing`模块的`Any`类型中创建一个序列（Sequence）类型的变量`AnySequence`。
4. 从`numpy`模块的`float`类型中创建一个列表（List）类型的变量`Embedding`。
5. 从`autogpt.config`模块的`Config`类中获取一个名为`logger`的配置对象（Object）。
6. 将步骤2中的上下文中的`suppress`函数与配置对象`logger`关联，这样在代码块中输出将被 suppressed。
7. 将步骤3中的`AnySequence`类型变量`ctx`与`Embedding`列表类型变量`Embedding`关联，这样当`ctx`为`Embedding`时，对应的`Embedding`值将被聚合。
8. 从`TText`类型创建一个序列（Sequence）类型的变量`TText`。

这段代码定义了一个具有`TText`类型的变量`ctx`，该变量将作为`TextField`的`field_name`参数。`ctx`将包含一个包含`TText`的最大长度（这里设置为256），以及一个涵盖模型训练和部署信息的配置对象（这里使用`Config`类从`autogpt.config`中获取）。


```py
import logging
from contextlib import suppress
from typing import Any, Sequence, overload

import numpy as np

from autogpt.config import Config

logger = logging.getLogger(__name__)

Embedding = list[float] | list[np.float32] | np.ndarray[Any, np.dtype[np.float32]]
"""Embedding vector"""

TText = Sequence[int]
"""Tokenized text"""


```

This is a function that gets an embedding from the Ada model. It takes one or more inputs, encoded as a string or array of tokens, and config for the embedding model.

The function uses the `_get_embedding_with_plugin` function to get the embedding. This function is not defined in the provided code, so it needs to be defined elsewhere.

The `get_embedding` function returns a list of embeddings, or a single embedding if the input is a list of strings or token arrays.

If the input is a string, the function replaces the newline characters with spaces to convert the input to be compatible with the Ada model.

If the input is a list of strings or a list of token arrays, the function applies the `_get_embedding_with_plugin` function to each token and returns the list of embeddings.

If the input is a list of strings or a list of token arrays, the function also applies the `_get_embedding_with_plugin` function to the first token in the list and returns the single embedding.


```py
@overload
def get_embedding(input: str | TText, config: Config) -> Embedding:
    ...


@overload
def get_embedding(input: list[str] | list[TText], config: Config) -> list[Embedding]:
    ...


def get_embedding(
    input: str | TText | list[str] | list[TText], config: Config
) -> Embedding | list[Embedding]:
    """Get an embedding from the ada model.

    Args:
        input: Input text to get embeddings for, encoded as a string or array of tokens.
            Multiple inputs may be given as a list of strings or token arrays.

    Returns:
        List[float]: The embedding.
    """
    multiple = isinstance(input, list) and all(not isinstance(i, int) for i in input)

    if isinstance(input, str):
        input = input.replace("\n", " ")

        with suppress(NotImplementedError):
            return _get_embedding_with_plugin(input, config)

    elif multiple and isinstance(input[0], str):
        input = [text.replace("\n", " ") for text in input]

        with suppress(NotImplementedError):
            return [_get_embedding_with_plugin(i, config) for i in input]

    model = config.embedding_model
    kwargs = {"model": model}
    kwargs.update(config.get_openai_credentials(model))

    logger.debug(
        f"Getting embedding{f's for {len(input)} inputs' if multiple else ''}"
        f" with model '{model}'"
        + (f" via Azure deployment '{kwargs['engine']}'" if config.use_azure else "")
    )

    embeddings = embedding_provider.create_embedding(
        input,
        **kwargs,
    ).data

    if not multiple:
        return embeddings[0]["embedding"]

    embeddings = sorted(embeddings, key=lambda x: x["index"])
    return [d["embedding"] for d in embeddings]


```

这段代码是一个函数，名为 `_get_embedding_with_plugin`，它接收两个参数 `text` 和 `config`，并返回一个 `Embedding` 对象。函数的作用是：

1. 遍历 `config` 对象中的所有插件，每个插件都需要支持文本嵌入。
2. 如果某个插件支持文本嵌入，就调用该插件的 `handle_text_embedding` 方法将文本嵌入到一个新的内存空间中。
3. 如果得到一个新的嵌入后，将之前的结果返回。
4. 如果所有插件都不支持文本嵌入，就 raise `NotImplementedError`。


```py
def _get_embedding_with_plugin(text: str, config: Config) -> Embedding:
    for plugin in config.plugins:
        if plugin.can_handle_text_embedding(text):
            embedding = plugin.handle_text_embedding(text)
            if embedding is not None:
                return embedding

    raise NotImplementedError

```

# `autogpts/autogpt/autogpt/memory/vector/__init__.py`

这段代码是一个Python脚本，它通过导入日志类`logging`，从`autogpt.config`包中继承了一个`Config`类。

接着从`memory_item`类开始定义了`MemoryItem`和`MemoryItemRelevance`类，这些类似乎和内存相关。

然后从`vector_memory`和`json_file`和`no_memory`这些名称来看，它们似乎是提供内存的方式，但不确定它们的确切实现。

接着定义了一个`supported_memory`列表，该列表列出了支持的所有内存 backend。

在最后，代码尝试导入了一个名为`RedisMemory`的内存提供者，但没有给出具体的实现。


```py
import logging

from autogpt.config import Config

from .memory_item import MemoryItem, MemoryItemRelevance
from .providers.base import VectorMemoryProvider as VectorMemory
from .providers.json_file import JSONFileMemory
from .providers.no_memory import NoMemory

# List of supported memory backends
# Add a backend to this list if the import attempt is successful
supported_memory = ["json_file", "no_memory"]

# try:
#     from .providers.redis import RedisMemory

```

这段代码使用了Python的异常处理机制，当遇到ImportError时，会执行except语句。如果except语句中的代码可以处理当前遇到的问题，则继续执行except语句，否则引发一个新的异常。

在这段代码中，首先定义了一个变量supported_memory，并将其初始化为空列表。然后，通过尝试从.providers目录中导入PineconeMemory、WeaviateMemory和Pinecone三个内存提供商，如果成功导入，则将对应的支持内存append到supported_memory列表中。如果尝试导入失败，则会引发新的ImportError，这样就可以捕获当前的异常，处理完当前的异常之后，继续执行except语句，从而避免引发新的异常。


```py
#     supported_memory.append("redis")
# except ImportError:
#     RedisMemory = None

# try:
#     from .providers.pinecone import PineconeMemory

#     supported_memory.append("pinecone")
# except ImportError:
#     PineconeMemory = None

# try:
#     from .providers.weaviate import WeaviateMemory

#     supported_memory.append("weaviate")
```

This is a function that creates a memory backend for a Redis application. It takes a `config` dictionary as input, which specifies the Redis connection settings, such as the Redis server URL, port, and password.

If the required dependencies, such as `redis-py`, `weaviate-client`, `AutoGPT`, `pymilvus`, `Significant-Gravitas/AutoGPT`, `WeaviateMemory`, `NoMemory`, or `JSONFileMemory`, are not installed, the function will raise a `NotImplementedError`.

If the specified memory backend is not available, the function will raise a `NotImplementedError`, and also provide instructions on how to install it.

If the `config` dictionary is missing the `memory_backend` key, the function will raise a `ValueError`.

If the `create_memory_backend` function successfully creates the memory backend, it will return the created memory backend.


```py
# except ImportError:
#     WeaviateMemory = None

# try:
#     from .providers.milvus import MilvusMemory

#     supported_memory.append("milvus")
# except ImportError:
#     MilvusMemory = None


def get_memory(config: Config) -> VectorMemory:
    """Returns a memory object corresponding to the memory backend specified in the config.

    The type of memory object returned depends on the value of the `memory_backend`
    attribute in the configuration. E.g. if `memory_backend` is set to "pinecone", a
    `PineconeMemory` object is returned. If it is set to "redis", a `RedisMemory`
    object is returned.
    By default, a `JSONFileMemory` object is returned.

    Params:
        config: A configuration object that contains information about the memory backend
            to be used and other relevant parameters.

    Returns:
        VectorMemory: an instance of a memory object based on the configuration provided.
    """
    memory = None

    match config.memory_backend:
        case "json_file":
            memory = JSONFileMemory(config)

        case "pinecone":
            raise NotImplementedError(
                "The Pinecone memory backend has been rendered incompatible by work on "
                "the memory system, and was removed. Whether support will be added back "
                "in the future is subject to discussion, feel free to pitch in: "
                "https://github.com/Significant-Gravitas/AutoGPT/discussions/4280"
            )
            # if not PineconeMemory:
            #     logger.warn(
            #         "Error: Pinecone is not installed. Please install pinecone"
            #         " to use Pinecone as a memory backend."
            #     )
            # else:
            #     memory = PineconeMemory(config)
            #     if clear:
            #         memory.clear()

        case "redis":
            raise NotImplementedError(
                "The Redis memory backend has been rendered incompatible by work on "
                "the memory system, and has been removed temporarily."
            )
            # if not RedisMemory:
            #     logger.warn(
            #         "Error: Redis is not installed. Please install redis-py to"
            #         " use Redis as a memory backend."
            #     )
            # else:
            #     memory = RedisMemory(config)

        case "weaviate":
            raise NotImplementedError(
                "The Weaviate memory backend has been rendered incompatible by work on "
                "the memory system, and was removed. Whether support will be added back "
                "in the future is subject to discussion, feel free to pitch in: "
                "https://github.com/Significant-Gravitas/AutoGPT/discussions/4280"
            )
            # if not WeaviateMemory:
            #     logger.warn(
            #         "Error: Weaviate is not installed. Please install weaviate-client to"
            #         " use Weaviate as a memory backend."
            #     )
            # else:
            #     memory = WeaviateMemory(config)

        case "milvus":
            raise NotImplementedError(
                "The Milvus memory backend has been rendered incompatible by work on "
                "the memory system, and was removed. Whether support will be added back "
                "in the future is subject to discussion, feel free to pitch in: "
                "https://github.com/Significant-Gravitas/AutoGPT/discussions/4280"
            )
            # if not MilvusMemory:
            #     logger.warn(
            #         "Error: pymilvus sdk is not installed."
            #         "Please install pymilvus to use Milvus or Zilliz Cloud as memory backend."
            #     )
            # else:
            #     memory = MilvusMemory(config)

        case "no_memory":
            memory = NoMemory()

        case _:
            raise ValueError(
                f"Unknown memory backend '{config.memory_backend}'. Please check your config."
            )

    if memory is None:
        memory = JSONFileMemory(config)

    return memory


```

这段代码定义了一个名为 `get_supported_memory_backends` 的函数，它返回一个列表，包含了多种内存后端（memory backend）的名称。

函数的作用是帮助用户在项目中选择合适的硬件内存后端。它返回的列表包含了以下内存后端：

* `supported_memory`：可能是服务器内存，也可能是客户端内存，具体取决于设置的环境变量。
* `NoMemory`：不提供内存，永远不会在项目中使用。
* `VectorMemory`：提供高性能的固态硬盘（SSD）内存，适用于需要快速启动和快速写的场景。
* `RedisMemory`：提供高性能的 Redis 内存，适用于需要高并发、高性能的道闸服务。
* `PineconeMemory`：提供高性能的 Pinecone 内存，适用于需要高并发、高性能的 AI 服务。
* `MilvusMemory`：提供高性能的 Milvus 内存，适用于需要高并发、高性能的图形数据库（例如 Docker、Kubernetes 等）。
* `WeaviateMemory`：提供高性能的 Weaviate 内存，适用于需要高并发、高性能的分布式数据存储系统（例如 Redis）。

此外，还定义了一个名为 `__all__` 的列表，其中包含了一些与函数 `get_supported_memory_backends` 相关的模块，如 `get_memory`、`MemoryItem`、`MemoryItemRelevance` 等。


```py
def get_supported_memory_backends():
    return supported_memory


__all__ = [
    "get_memory",
    "MemoryItem",
    "MemoryItemRelevance",
    "JSONFileMemory",
    "NoMemory",
    "VectorMemory",
    # "RedisMemory",
    # "PineconeMemory",
    # "MilvusMemory",
    # "WeaviateMemory",
]

```

# `autogpts/autogpt/autogpt/memory/vector/providers/base.py`

这段代码的作用是定义了一个名为 `MemoryItem` 的类，以及一个名为 `MemoryItemRelevance` 的类，和一个名为 `get_embedding` 的函数，同时引入了 `abc`、`functools` 和 `logging` 模块，以及 `numpy` 模块。

具体来说，这段代码定义了一个 `MemoryItem` 类，该类包含一个引用 `get_embedding` 函数，以及一个引用 `MemoryItemRelevance` 函数的元组。`MemoryItem` 类可能用于跟踪和记录与一个主题相关的信息，例如文本中的单词或短语。

`MemoryItemRelevance` 类可能用于计算与给定主题相关的信息对某个主题文本窗口的相对重要性。

`get_embedding` 函数用于从 `get_embedding` 函数中获取一个适当的嵌入值，该函数将在后面进行说明。

这段代码还定义了一个 `logging` 对象，用于在发生日志记录时记录日志。


```py
import abc
import functools
import logging
from typing import MutableSet, Sequence

import numpy as np

from autogpt.config.config import Config

from .. import MemoryItem, MemoryItemRelevance
from ..utils import Embedding, get_embedding

logger = logging.getLogger(__name__)


```

This is a class called `MemoryObject` which seems to be a database of memories. It has methods to search for relevant memories, get the top `k` most relevant memories, and get the statistics of the memories in the index.

The class has a `get_relevant` method which takes a query, number of relevant memories `k`, and a config object. This method returns a list of `MemoryItemRelevance` objects that correspond to the top `k` relevant memories.

The class also has a `get_stats` method which returns the statistics of the memories in the index, such as the number of memories and the number of chunks.

It is used to be used in a `MemorySearch` class which inherits from `Searchable`


```py
class VectorMemoryProvider(MutableSet[MemoryItem]):
    @abc.abstractmethod
    def __init__(self, config: Config):
        pass

    def get(self, query: str, config: Config) -> MemoryItemRelevance | None:
        """
        Gets the data from the memory that is most relevant to the given query.

        Args:
            query: The query used to retrieve information.
            config: The config Object.

        Returns: The most relevant Memory
        """
        result = self.get_relevant(query, 1, config)
        return result[0] if result else None

    def get_relevant(
        self, query: str, k: int, config: Config
    ) -> Sequence[MemoryItemRelevance]:
        """
        Returns the top-k most relevant memories for the given query

        Args:
            query: the query to compare stored memories to
            k: the number of relevant memories to fetch
            config: The config Object.

        Returns:
            list[MemoryItemRelevance] containing the top [k] relevant memories
        """
        if len(self) < 1:
            return []

        logger.debug(
            f"Searching for {k} relevant memories for query '{query}'; "
            f"{len(self)} memories in index"
        )

        relevances = self.score_memories_for_relevance(query, config)
        logger.debug(f"Memory relevance scores: {[str(r) for r in relevances]}")

        # take last k items and reverse
        top_k_indices = np.argsort([r.score for r in relevances])[-k:][::-1]

        return [relevances[i] for i in top_k_indices]

    def score_memories_for_relevance(
        self, for_query: str, config: Config
    ) -> Sequence[MemoryItemRelevance]:
        """
        Returns MemoryItemRelevance for every memory in the index.
        Implementations may override this function for performance purposes.
        """
        e_query: Embedding = get_embedding(for_query, config)
        return [m.relevance_for(for_query, e_query) for m in self]

    def get_stats(self) -> tuple[int, int]:
        """
        Returns:
            tuple (n_memories: int, n_chunks: int): the stats of the memory index
        """
        return len(self), functools.reduce(lambda t, m: t + len(m.e_chunks), self, 0)

```

# `autogpts/autogpt/autogpt/memory/vector/providers/json_file.py`

这段代码是一个名为`MemoryItem`的类，它继承自`VectorMemoryProvider`类。`MemoryItem`类用于实现一个内存块的内存管理，包括从磁盘读取数据到内存，以及从内存写入数据到磁盘。

具体来说，这段代码实现了一个`MemoryItem`的私有化，该私有化使用`__future__`注解从`typing`中引进了`Iterator`类型。`Iterator`类型允许我们将一个对象的所有元素一次遍历出来。因此，`MemoryItem`可以使用`for`循环来遍历内存块中的数据。

`MemoryItem`类包含一个`config`属性，用于访问`autogpt.config.Config`类中定义的选项。`Config`类用于配置`autogpt`模型的参数和选项。

`MemoryItem`类中还包含一个`vector_memory_provider`属性，该属性实例化了一个`VectorMemoryProvider`类，用于管理内存块。`VectorMemoryProvider`类实现了`typing.Queue`接口，用于管理一个队列中的数据。通过将`Queue`类型的数据转换为`typing.Iterable`类型，可以实现一次遍历。

`MemoryItem`类中还包含一个`path_to_file`属性，用于指定从磁盘中读取的数据源。

最后，通过组合`MemoryItem`类和`VectorMemoryProvider`类，可以实现一个完整的`autogpt.model.MemoryItem`类，用于管理磁盘上的数据，并将其加载到内存中。


```py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import orjson

from autogpt.config import Config

from ..memory_item import MemoryItem
from .base import VectorMemoryProvider

logger = logging.getLogger(__name__)


```

This is a Python class called "MemoryFile" that implements a file-based data structure for storing and retrieving memories ( objects ) from a file.

It has the following methods:

* `__init__(self, file_path, save_index=True)`: Initializes the file object and sets the `save_index` flag to `True`.
* `load_index(self)`: Loads all memories from the index file specified by `file_path`.
* `discard(self, item: MemoryItem)`: Discards the item from the current memory file.
* `clear(self)`: Clears the data in the current memory file.
* `add(self, item: MemoryItem)`: Adds the item to the current memory file.
* `remove(self, item: MemoryItem)`: Removes the item from the current memory file.
* `__len__(self)`: Returns the length of the current memory file.
* `__contains__(self, x: MemoryItem)`: Checks if the item is in the current memory file.
* `__getitem__(self, index)`: Retrieves the item at the specified index in the current memory file.
* `__setitem__(self, index, item)`: Sets the item at the specified index in the current memory file.
* `__delitem__(self, index)`: Removes the item at the specified index in the current memory file.
* `__bool__(self)`: Checks if the current memory file exists.

It also has a ` dump(self)` method which dumps the current memory file to a string.


```py
class JSONFileMemory(VectorMemoryProvider):
    """Memory backend that stores memories in a JSON file"""

    SAVE_OPTIONS = orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_SERIALIZE_DATACLASS

    file_path: Path
    memories: list[MemoryItem]

    def __init__(self, config: Config) -> None:
        """Initialize a class instance

        Args:
            config: Config object

        Returns:
            None
        """
        self.file_path = config.workspace_path / f"{config.memory_index}.json"
        self.file_path.touch()
        logger.debug(
            f"Initialized {__class__.__name__} with index path {self.file_path}"
        )

        self.memories = []
        try:
            self.load_index()
            logger.debug(f"Loaded {len(self.memories)} MemoryItems from file")
        except Exception as e:
            logger.warn(f"Could not load MemoryItems from file: {e}")
            self.save_index()

    def __iter__(self) -> Iterator[MemoryItem]:
        return iter(self.memories)

    def __contains__(self, x: MemoryItem) -> bool:
        return x in self.memories

    def __len__(self) -> int:
        return len(self.memories)

    def add(self, item: MemoryItem):
        self.memories.append(item)
        logger.debug(f"Adding item to memory: {item.dump()}")
        self.save_index()
        return len(self.memories)

    def discard(self, item: MemoryItem):
        try:
            self.remove(item)
        except:
            pass

    def clear(self):
        """Clears the data in memory."""
        self.memories.clear()
        self.save_index()

    def load_index(self):
        """Loads all memories from the index file"""
        if not self.file_path.is_file():
            logger.debug(f"Index file '{self.file_path}' does not exist")
            return
        with self.file_path.open("r") as f:
            logger.debug(f"Loading memories from index file '{self.file_path}'")
            json_index = orjson.loads(f.read())
            for memory_item_dict in json_index:
                self.memories.append(MemoryItem.parse_obj(memory_item_dict))

    def save_index(self):
        logger.debug(f"Saving memory index to file {self.file_path}")
        with self.file_path.open("wb") as f:
            return f.write(
                orjson.dumps(
                    [m.dict() for m in self.memories], option=self.SAVE_OPTIONS
                )
            )

```

# `autogpts/autogpt/autogpt/memory/vector/providers/no_memory.py`

这段代码定义了一个名为 `NoMemory` 的类，它实现了 `VectorMemoryProvider` 的接口。这个类的特点是不存储任何数据，因此它是一个默认的内存提供者。

从这段代码中，我们可以看到以下几点解释：

1. `NoMemory` 类继承自 `VectorMemoryProvider` 类，这个类可能实现了某些具体的内存提供功能。但是在这段注释中，我们并没有看到任何实现。
2. `NoMemory` 类包含了一个 `__init__` 方法，这个方法可以被用于初始化对象。在初始化时，我们可以传入一个 `config` 参数，它可能是 `Config` 类的实例。
3. `NoMemory` 类包含了一个 `__iter__` 方法，这个方法返回一个迭代器。在 `__iter__` 中，对象会被遍历并返回每个内存项。
4. `NoMemory` 类包含了一个 `__contains__` 方法，这个方法用于判断一个内存项是否属于这个对象。在 `__contains__` 中，如果对象包含这个内存项，那么返回 `True`，否则返回 `False`。
5. `NoMemory` 类包含了一个 `__len__` 方法，这个方法用于返回对象中内存项的数量。在 `__len__` 中，如果没有内存项，那么返回 0。
6. `NoMemory` 类包含了一个 `add` 方法，这个方法用于添加一个新的内存项。在 `add` 中，我们可以看到一些方法来设置或删除内存项，但是这些方法并没有提供具体的数据存储功能。
7. `NoMemory` 类包含了一个 `discard` 方法，这个方法用于丢弃一个内存项。在 `discard` 中，我们可以看到一些方法来设置或删除内存项，但是这些方法并没有提供具体的数据存储功能。
8. `NoMemory` 类还包含了一个 `clear` 方法，这个方法用于清除对象中的所有内存项。在 `clear` 中，我们可以看到一些方法来设置或删除内存项，但是这些方法并没有提供具体的数据存储功能。


```py
"""A class that does not store any data. This is the default memory provider."""
from __future__ import annotations

from typing import Iterator, Optional

from autogpt.config.config import Config

from .. import MemoryItem
from .base import VectorMemoryProvider


class NoMemory(VectorMemoryProvider):
    """
    A class that does not store any data. This is the default memory provider.
    """

    def __init__(self, config: Optional[Config] = None):
        pass

    def __iter__(self) -> Iterator[MemoryItem]:
        return iter([])

    def __contains__(self, x: MemoryItem) -> bool:
        return False

    def __len__(self) -> int:
        return 0

    def add(self, item: MemoryItem):
        pass

    def discard(self, item: MemoryItem):
        pass

    def clear(self):
        pass

```

# `autogpts/autogpt/autogpt/memory/vector/providers/__init__.py`

这段代码定义了两个函数分别从不同的命名空间 import 和导入了两个不同的模块。

首先，from .json_file import JSONFileMemory，它从 .json_file 命名空间中继承了一个名为 JSONFileMemory 的类，这个类可能是一个用于读取或写入 JSON文件的类。

接着，from .no_memory import NoMemory，它从 .no_memory 命名空间中继承了一个名为 NoMemory 的类，这个类可能是一个用于管理内存的类。


```py
from .json_file import JSONFileMemory
from .no_memory import NoMemory

__all__ = [
    "JSONFileMemory",
    "NoMemory",
]

```

# `autogpts/autogpt/autogpt/models/action_history.py`

这段代码是一个Python类，来自自动gradient（Autograd）库。它定义了一个名为“Action”的模型类，该模型类包含三个属性：name、args和reasoning。

在定义这些属性的同时，还使用了未来时态的类型注释，这意味着这些属性的值在代码运行时可能会发生变化。

此代码的作用是定义一个Action类，用于表示在自动gradient系统中产生的动作。这些动作可以用于在训练过程中调整模型参数，例如在训练过程中调整学习率、优化器设置等。

此类的实现有助于将定义的动作与自动gradient系统中的实际动作（如在训练过程中调整模型参数）分离，使得代码更加清晰、易于理解和维护。


```py
from __future__ import annotations

from typing import Any, Iterator, Literal, Optional

from pydantic import BaseModel, Field

from autogpt.prompts.utils import format_numbered_list, indent


class Action(BaseModel):
    name: str
    args: dict[str, Any]
    reasoning: str

    def format_call(self) -> str:
        return f"{self.name}({', '.join([f'{a}={repr(v)}' for a, v in self.args.items()])})"


```

这段代码定义了两个类，一个是 `ActionSuccessResult`，另一个是 `ErrorInfo`。它们都属于 `BaseModel` 类，意味着它们都继承自同一个模型类。

`ActionSuccessResult` 类有两个主要方法：`__str__` 和 `__repr__`。

`__str__` 方法返回一个字符串，它是类的字符串表示形式。在这个方法中，`self` 对象被遍历，并输出它的 `outputs` 属性。如果 `self` 是 `ActionSuccessResult` 对象，输出将是 `"success"`。

`__repr__` 方法返回一个字符串，它是类的元字符表示形式。在这个方法中，`self` 对象被遍历，并输出它的 `repr` 属性。如果 `self` 是 `ActionSuccessResult` 对象，输出将是 `"ActionSuccessResult"`。

`ErrorInfo` 类也有两个主要方法：`__str__` 和 `__repr__`。

`__str__` 方法返回一个字符串，它是类的字符串表示形式。在这个方法中，`self` 对象被遍历，并输出它的 `args` 和 `message` 属性。

`__repr__` 方法返回一个字符串，它是类的元字符表示形式。在这个方法中，`self` 对象被遍历，并输出它的 `repr` 属性。


```py
class ActionSuccessResult(BaseModel):
    outputs: Any
    status: Literal["success"] = "success"

    def __str__(self) -> str:
        outputs = str(self.outputs).replace("```", r"\```py")
        multiline = "\n" in outputs
        return f"```\n{self.outputs}\n```py" if multiline else str(self.outputs)


class ErrorInfo(BaseModel):
    args: tuple
    message: str
    exception_type: str
    repr: str

    @staticmethod
    def from_exception(exception: Exception) -> ErrorInfo:
        return ErrorInfo(
            args=exception.args,
            message=getattr(exception, "message", exception.args[0]),
            exception_type=exception.__class__.__name__,
            repr=repr(exception),
        )

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return self.repr


```py

这段代码定义了一个名为 ActionErrorResult 的类，其继承自 BaseModel 类(可能是某个具体库中的模型类)。

ActionErrorResult 类包含三个成员变量：reason(失败原因), error(错误信息), status(状态)，其中 status 成员变量设置为 "error"。

另外，还定义了一个 from_exception() 方法，用于将给定的异常对象转换为 ActionErrorResult 对象。这个方法从异常对象中获取失败原因，并将其与一个名为 ErrorInfo 的类对象关联起来。

最后，定义了一个 __str__() 方法，用于在字符串中格式化返回失败信息。

这段代码的主要作用是定义了一个 ActionErrorResult 类，用于表示在 REST API 中发生的错误。当在 API 方法中发生错误时，可以创建一个 ActionErrorResult 对象，其中包含失败原因、错误信息以及状态。通过调用 from_exception() 方法可以将一个异常对象转换为 ActionErrorResult 对象，通过调用 __str__() 方法可以将 ActionErrorResult 对象作为字符串返回。


```py
class ActionErrorResult(BaseModel):
    reason: str
    error: Optional[ErrorInfo] = None
    status: Literal["error"] = "error"

    @staticmethod
    def from_exception(exception: Exception) -> ActionErrorResult:
        return ActionErrorResult(
            reason=getattr(exception, "message", exception.args[0]),
            error=ErrorInfo.from_exception(exception),
        )

    def __str__(self) -> str:
        return f"Action failed: '{self.reason}'"


```py

这段代码定义了一个名为 "ActionInterruptedByHuman" 的类，继承自 "BaseModel" 类。该类包含一个 "feedback" 字段，一个名为 "status"，其值为 "interrupted_by_human"。

在类的 "__str__" 方法中，返回了对象的字符串表示，其中包括对象的属性和一个 "feedback" 字符串。

该类还有一个 "ActionResult" 类，该类包含了 "ActionSuccessResult"、"ActionErrorResult" 和 "ActionInterruptedByHuman" 三种类型。

该代码的最后一个类是 "Episode"，该类继承自 "BaseModel" 类。该类包含一个 "action" 字段和一个 "result" 字段。在类的 "__str__" 方法中，返回了执行的动作 "action" 和最终的 "ActionResult" 或 "None"。


```py
class ActionInterruptedByHuman(BaseModel):
    feedback: str
    status: Literal["interrupted_by_human"] = "interrupted_by_human"

    def __str__(self) -> str:
        return f'The user interrupted the action with the following feedback: "{self.feedback}"'


ActionResult = ActionSuccessResult | ActionErrorResult | ActionInterruptedByHuman


class Episode(BaseModel):
    action: Action
    result: ActionResult | None

    def __str__(self) -> str:
        executed_action = f"Executed `{self.action.format_call()}`"
        action_result = f": {self.result}" if self.result else "."
        return executed_action + action_result


```py

This is a class that appears to organize the steps of a process, such as a computation or a conversation. It has a current_episode member variable that is a list of episodes, which are themselves objects that represent the steps of the process. It also has a number_of_episodes member variable that is the number of episodes in the current cycle.

The class has several methods for manipulating the current cycle, including a fmt\_list method for rendering the current episodes as a formatted list, a fmt\_paragraph method for rendering the current episodes as a paragraph. These methods also allow the current episodes to be re-used by calling the respective methods.

The class also has a method called fmt\_numbered\_list, which is a helper method for formatting a list of episodes in a specific style. This method takes the current episodes as input and returns a formatted string, with each episode being indented according to its position in the list.


```py
class EpisodicActionHistory(BaseModel):
    """Utility container for an action history"""

    episodes: list[Episode] = Field(default_factory=list)
    cursor: int = 0

    @property
    def current_episode(self) -> Episode | None:
        if self.cursor == len(self):
            return None
        return self[self.cursor]

    def __getitem__(self, key: int) -> Episode:
        return self.episodes[key]

    def __iter__(self) -> Iterator[Episode]:
        return iter(self.episodes)

    def __len__(self) -> int:
        return len(self.episodes)

    def __bool__(self) -> bool:
        return len(self.episodes) > 0

    def register_action(self, action: Action) -> None:
        if not self.current_episode:
            self.episodes.append(Episode(action=action, result=None))
            assert self.current_episode
        elif self.current_episode.action:
            raise ValueError("Action for current cycle already set")

    def register_result(self, result: ActionResult) -> None:
        if not self.current_episode:
            raise RuntimeError("Cannot register result for cycle without action")
        elif self.current_episode.result:
            raise ValueError("Result for current cycle already set")

        self.current_episode.result = result
        self.cursor = len(self.episodes)

    def rewind(self, number_of_episodes: int = 0) -> None:
        """Resets the history to an earlier state.

        Params:
            number_of_cycles (int): The number of cycles to rewind. Default is 0.
                When set to 0, it will only reset the current cycle.
        """
        # Remove partial record of current cycle
        if self.current_episode:
            if self.current_episode.action and not self.current_episode.result:
                self.episodes.pop(self.cursor)

        # Rewind the specified number of cycles
        if number_of_episodes > 0:
            self.episodes = self.episodes[:-number_of_episodes]
            self.cursor = len(self.episodes)

    def fmt_list(self) -> str:
        return format_numbered_list(self.episodes)

    def fmt_paragraph(self) -> str:
        steps: list[str] = []

        for i, c in enumerate(self.episodes, 1):
            step = f"### Step {i}: Executed `{c.action.format_call()}`\n"
            step += f'- **Reasoning:** "{c.action.reasoning}"\n'
            step += (
                f"- **Status:** `{c.result.status if c.result else 'did_not_finish'}`\n"
            )
            if c.result:
                if c.result.status == "success":
                    result = str(c.result)
                    result = "\n" + indent(result) if "\n" in result else result
                    step += f"- **Output:** {result}"
                elif c.result.status == "error":
                    step += f"- **Reason:** {c.result.reason}\n"
                    if c.result.error:
                        step += f"- **Error:** {c.result.error}\n"
                elif c.result.status == "interrupted_by_human":
                    step += f"- **Feedback:** {c.result.feedback}\n"

            steps.append(step)

        return "\n\n".join(steps)

```py

# `autogpts/autogpt/autogpt/models/base_open_ai_plugin.py`

This is a chatbot skeleton implementation using NLTK, spaCy, and the chatbot framework.

First, we need to install the required packages by running the following command:
```pydiff
!pip install nltk spaCy chatbot-fr-api
```py
The `!pip install` command installs the required packages for this project. `nlter` is the chatbot library that provides the chatbot framework. `spaCy` is a popular natural language processing library. `chatbot-fr-api` is a chatbot API library for France.

After installing the required packages, we can create a new chatbot object and initialize its components as follows:
```pypython
from nltk import config
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from spaCy import Text
from spaCy.notify import和我聊
from chatbot_fr_api import Chatbot
```py
The `spaCy` and `chatbot_fr_api` objects are initialized with their respective APIs. `Text` is a class to represent text documents. `和我聊` is a class to handle the user's message.

Now, we can create a new chatbot object by passing the required parameters to the `Chatbot` constructor:
```py
```py


```py
"""Handles loading of plugins."""
from typing import Any, Dict, List, Optional, Tuple, TypedDict, TypeVar

from auto_gpt_plugin_template import AutoGPTPluginTemplate

PromptGenerator = TypeVar("PromptGenerator")


class Message(TypedDict):
    role: str
    content: str


class BaseOpenAIPlugin(AutoGPTPluginTemplate):
    """
    This is a BaseOpenAIPlugin class for generating AutoGPT plugins.
    """

    def __init__(self, manifests_specs_clients: dict):
        # super().__init__()
        self._name = manifests_specs_clients["manifest"]["name_for_model"]
        self._version = manifests_specs_clients["manifest"]["schema_version"]
        self._description = manifests_specs_clients["manifest"]["description_for_model"]
        self._client = manifests_specs_clients["client"]
        self._manifest = manifests_specs_clients["manifest"]
        self._openapi_spec = manifests_specs_clients["openapi_spec"]

    def can_handle_on_response(self) -> bool:
        """This method is called to check that the plugin can
        handle the on_response method.
        Returns:
            bool: True if the plugin can handle the on_response method."""
        return False

    def on_response(self, response: str, *args, **kwargs) -> str:
        """This method is called when a response is received from the model."""
        return response

    def can_handle_post_prompt(self) -> bool:
        """This method is called to check that the plugin can
        handle the post_prompt method.
        Returns:
            bool: True if the plugin can handle the post_prompt method."""
        return False

    def post_prompt(self, prompt: PromptGenerator) -> PromptGenerator:
        """This method is called just after the generate_prompt is called,
            but actually before the prompt is generated.
        Args:
            prompt (PromptGenerator): The prompt generator.
        Returns:
            PromptGenerator: The prompt generator.
        """
        return prompt

    def can_handle_on_planning(self) -> bool:
        """This method is called to check that the plugin can
        handle the on_planning method.
        Returns:
            bool: True if the plugin can handle the on_planning method."""
        return False

    def on_planning(
        self, prompt: PromptGenerator, messages: List[Message]
    ) -> Optional[str]:
        """This method is called before the planning chat completion is done.
        Args:
            prompt (PromptGenerator): The prompt generator.
            messages (List[str]): The list of messages.
        """

    def can_handle_post_planning(self) -> bool:
        """This method is called to check that the plugin can
        handle the post_planning method.
        Returns:
            bool: True if the plugin can handle the post_planning method."""
        return False

    def post_planning(self, response: str) -> str:
        """This method is called after the planning chat completion is done.
        Args:
            response (str): The response.
        Returns:
            str: The resulting response.
        """
        return response

    def can_handle_pre_instruction(self) -> bool:
        """This method is called to check that the plugin can
        handle the pre_instruction method.
        Returns:
            bool: True if the plugin can handle the pre_instruction method."""
        return False

    def pre_instruction(self, messages: List[Message]) -> List[Message]:
        """This method is called before the instruction chat is done.
        Args:
            messages (List[Message]): The list of context messages.
        Returns:
            List[Message]: The resulting list of messages.
        """
        return messages

    def can_handle_on_instruction(self) -> bool:
        """This method is called to check that the plugin can
        handle the on_instruction method.
        Returns:
            bool: True if the plugin can handle the on_instruction method."""
        return False

    def on_instruction(self, messages: List[Message]) -> Optional[str]:
        """This method is called when the instruction chat is done.
        Args:
            messages (List[Message]): The list of context messages.
        Returns:
            Optional[str]: The resulting message.
        """

    def can_handle_post_instruction(self) -> bool:
        """This method is called to check that the plugin can
        handle the post_instruction method.
        Returns:
            bool: True if the plugin can handle the post_instruction method."""
        return False

    def post_instruction(self, response: str) -> str:
        """This method is called after the instruction chat is done.
        Args:
            response (str): The response.
        Returns:
            str: The resulting response.
        """
        return response

    def can_handle_pre_command(self) -> bool:
        """This method is called to check that the plugin can
        handle the pre_command method.
        Returns:
            bool: True if the plugin can handle the pre_command method."""
        return False

    def pre_command(
        self, command_name: str, arguments: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """This method is called before the command is executed.
        Args:
            command_name (str): The command name.
            arguments (Dict[str, Any]): The arguments.
        Returns:
            Tuple[str, Dict[str, Any]]: The command name and the arguments.
        """
        return command_name, arguments

    def can_handle_post_command(self) -> bool:
        """This method is called to check that the plugin can
        handle the post_command method.
        Returns:
            bool: True if the plugin can handle the post_command method."""
        return False

    def post_command(self, command_name: str, response: str) -> str:
        """This method is called after the command is executed.
        Args:
            command_name (str): The command name.
            response (str): The response.
        Returns:
            str: The resulting response.
        """
        return response

    def can_handle_chat_completion(
        self, messages: Dict[Any, Any], model: str, temperature: float, max_tokens: int
    ) -> bool:
        """This method is called to check that the plugin can
          handle the chat_completion method.
        Args:
            messages (List[Message]): The messages.
            model (str): The model name.
            temperature (float): The temperature.
            max_tokens (int): The max tokens.
          Returns:
              bool: True if the plugin can handle the chat_completion method."""
        return False

    def handle_chat_completion(
        self, messages: List[Message], model: str, temperature: float, max_tokens: int
    ) -> str:
        """This method is called when the chat completion is done.
        Args:
            messages (List[Message]): The messages.
            model (str): The model name.
            temperature (float): The temperature.
            max_tokens (int): The max tokens.
        Returns:
            str: The resulting response.
        """

    def can_handle_text_embedding(self, text: str) -> bool:
        """This method is called to check that the plugin can
          handle the text_embedding method.

        Args:
            text (str): The text to be convert to embedding.
        Returns:
            bool: True if the plugin can handle the text_embedding method."""
        return False

    def handle_text_embedding(self, text: str) -> list[float]:
        """This method is called to create a text embedding.

        Args:
            text (str): The text to be convert to embedding.
        Returns:
            list[float]: The created embedding vector.
        """

    def can_handle_user_input(self, user_input: str) -> bool:
        """This method is called to check that the plugin can
        handle the user_input method.

        Args:
            user_input (str): The user input.

        Returns:
            bool: True if the plugin can handle the user_input method."""
        return False

    def user_input(self, user_input: str) -> str:
        """This method is called to request user input to the user.

        Args:
            user_input (str): The question or prompt to ask the user.

        Returns:
            str: The user input.
        """

    def can_handle_report(self) -> bool:
        """This method is called to check that the plugin can
        handle the report method.

        Returns:
            bool: True if the plugin can handle the report method."""
        return False

    def report(self, message: str) -> None:
        """This method is called to report a message to the user.

        Args:
            message (str): The message to report.
        """

```py

# `autogpts/autogpt/autogpt/models/command.py`

这段代码是一个命令参数的定义，其中包括了两个参数：一个是命令参数类型(Any)，另一个是上下文项类型(ContextItem)。通过从`__future__`模块中导入`annotations`，可以定义出这些参数的类型。

同时，通过使用`from typing import TYPE_CHECKING`来定义了命令参数的两个参数类型，即`from autogpt.agents.base import BaseAgent`和`from autogpt.config import Config`。

接着，定义了`CommandReturnValue`类型为`Any`类型，表示命令参数返回的结果可以是任何类型。

定义了`CommandOutput`类型为`CommandReturnValue | tuple[CommandReturnValue, ContextItem]`类型。其中，`CommandReturnValue`类型为`CommandReturnValue`，表示命令参数返回的结果为命令返回值，`tuple`函数表示将两个参数打包成一个元组，第一个参数为`CommandReturnValue`，第二个参数为`ContextItem`。

最后，通过`from .command_parameter import CommandParameter`和`from .context_item import ContextItem`模块，可以导入命令参数的具体实现。


```py
from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional

if TYPE_CHECKING:
    from autogpt.agents.base import BaseAgent
    from autogpt.config import Config

from .command_parameter import CommandParameter
from .context_item import ContextItem

CommandReturnValue = Any
CommandOutput = CommandReturnValue | tuple[CommandReturnValue, ContextItem]


```py

This is a class definition for a `Command` object that has a method for executing a specified function and a list of parameters. The method has an optional `disabled_reason` parameter to specify why the command is disabled, and an `available` parameter to specify whether the command is available to the agent. The method also has a `__call__` method that is a special method in the class that allows the command to be executed. The class has a `__init__` method that is used to initialize the command object with the given parameters and options. The class also defines a property `is_async` that indicates whether the method is an asynchronous function.


```py
class Command:
    """A class representing a command.

    Attributes:
        name (str): The name of the command.
        description (str): A brief description of what the command does.
        parameters (list): The parameters of the function that the command executes.
    """

    def __init__(
        self,
        name: str,
        description: str,
        method: Callable[..., CommandOutput],
        parameters: list[CommandParameter],
        enabled: Literal[True] | Callable[[Config], bool] = True,
        disabled_reason: Optional[str] = None,
        aliases: list[str] = [],
        available: Literal[True] | Callable[[BaseAgent], bool] = True,
    ):
        self.name = name
        self.description = description
        self.method = method
        self.parameters = parameters
        self.enabled = enabled
        self.disabled_reason = disabled_reason
        self.aliases = aliases
        self.available = available

    @property
    def is_async(self) -> bool:
        return inspect.iscoroutinefunction(self.method)

    def __call__(self, *args, agent: BaseAgent, **kwargs) -> Any:
        if callable(self.enabled) and not self.enabled(agent.legacy_config):
            if self.disabled_reason:
                raise RuntimeError(
                    f"Command '{self.name}' is disabled: {self.disabled_reason}"
                )
            raise RuntimeError(f"Command '{self.name}' is disabled")

        if callable(self.available) and not self.available(agent):
            raise RuntimeError(f"Command '{self.name}' is not available")

        return self.method(*args, **kwargs, agent=agent)

    def __str__(self) -> str:
        params = [
            f"{param.name}: {param.spec.type.value if param.spec.required else f'Optional[{param.spec.type.value}]'}"
            for param in self.parameters
        ]
        return f"{self.name}: {self.description.rstrip('.')}. Params: ({', '.join(params)})"

```py

# `autogpts/autogpt/autogpt/models/command_parameter.py`

这段代码使用了Python的dataclasses库来自动生成类的定义，以便与其他代码片段进行交互和复用。

在代码中，首先从autogpt.core.utils.json_schema模块中引入了JSONSchema类，该类用于描述数据模型的JSON规范。

接着定义了一个CommandParameter类，该类包含一个name属性、一个spec属性(该属性是一个JSONSchema对象)、以及__repr__方法。

__repr__方法是Python 3.6中新增的特性，用于打印对象的唯一字符串表示形式，该方法使用了format_召集了self和spec进入参数列表，并打印了模型的名称、spec对象的类型、description属性的值以及required属性的值。

最后，在CommandParameter类的定义中导入了dataclasses库，以便能够使用@dataclasses.dataclass注解来自动生成该类的定义。


```py
import dataclasses

from autogpt.core.utils.json_schema import JSONSchema


@dataclasses.dataclass
class CommandParameter:
    name: str
    spec: JSONSchema

    def __repr__(self):
        return f"CommandParameter('{self.name}', '{self.spec.type}', '{self.spec.description}', {self.spec.required})"

```py

# `autogpts/autogpt/autogpt/models/command_registry.py`

这段代码是一个自定义的 Python 类，其目的是在定义自定义类时自动注册一个自动机器学习(AutoML)的类。这个类包含了两个静态方法，一个方法是使用 `from typing import TYPE_CHECKING` 带来的特性，用于检查特定类的类型是否符合某个特定的机器学习模式，另一个方法则是注册该自定义类到 `__init__` 函数中，以便在创建自定义类实例时自动创建一个 instance。

具体来说，这段代码的作用是：

1. 从 `__future__` 导入 `annotations` 模块，以便使用 `@dataclass` 和 `@field` 注解来定义数据类和数据字段。

2. 导入 `importlib`、`inspect`、`logging` 和 `types` 模块，以便能够方便地使用这些库。

3. 从 `types` 模块中导入 `ModuleType` 类型，以便在需要自动注册某个机器学习模型的类中使用。

4. 从 `dataclasses` 模块中导入 `dataclass` 和 `field` 注解，以便能够定义数据类。

5. 从 `typing` 模块中导入 `TYPE_CHECKING` 和 `Any` 类型，以便能够创建一个可以作为所有类型对象的通用类型。

6. 从 `ai.turntill.api` 库中导入 `BaseAgent` 和 `Config` 类，以便能够在自定义机器学习模型的训练和部署中使用。

7. 在 `__init__` 函数中，注册了自定义类的两个静态方法：`register_my_model_if_type` 和 `create_agent_from_config`。

8. 在 `register_my_model_if_type` 方法中，使用了 `inspect.signature.f高科技` 函数，这个函数可以解析特定类型的机器学习模型的 `__init__` 和 `__new__` 函数的签名，并创建一个类的实例。

9. 在 `create_agent_from_config` 方法中，使用了 `BaseAgent` 和 `Config` 类，创建了一个新的机器学习模型类，并注册了这个类到 `__init__` 函数中。

10. 最后，在 `if TYPE_CHECKING` 注释下，可以保证 `AUTO_GPT_COMMAND_IDENTIFIER` 类是一个有效的机器学习模型类，从而可以自动注册。


```py
from __future__ import annotations

import importlib
import inspect
import logging
from dataclasses import dataclass, field
from types import ModuleType
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from autogpt.agents.base import BaseAgent
    from autogpt.config import Config


from autogpt.command_decorator import AUTO_GPT_COMMAND_IDENTIFIER
```py

This is a Python class that provides a method for registering command plugins in an application.

The class has a `register_command_module` method, which takes the name of the Python module
to be imported as a parameter. This method imports the module, registers any functions or classes
that are decorated with the `AUTO_GPT_COMMAND_IDENTIFIER` attribute as `Command` objects,
and adds them to the `commands` dictionary of a `CommandRegistry` object.

The class also has a `register_module_category` method, which is used for registering command categories. This method
is used when a command plugin is imported from a module that has not an existing `COMMAND_CATEGORY`
attribute. It takes the module object as a parameter and returns a `CommandCategory` object.

The `register_module_category` method, if called successfully, will register the commands of the
module in the category, where each command is a function or class that is decorated
with the `AUTO_GPT_COMMAND_IDENTIFIER` attribute.


```py
from autogpt.models.command import Command

logger = logging.getLogger(__name__)


class CommandRegistry:
    """
    The CommandRegistry class is a manager for a collection of Command objects.
    It allows the registration, modification, and retrieval of Command objects,
    as well as the scanning and loading of command plugins from a specified
    directory.
    """

    commands: dict[str, Command]
    commands_aliases: dict[str, Command]

    # Alternative way to structure the registry; currently redundant with self.commands
    categories: dict[str, CommandCategory]

    @dataclass
    class CommandCategory:
        name: str
        title: str
        description: str
        commands: list[Command] = field(default_factory=list[Command])
        modules: list[ModuleType] = field(default_factory=list[ModuleType])

    def __init__(self):
        self.commands = {}
        self.commands_aliases = {}
        self.categories = {}

    def __contains__(self, command_name: str):
        return command_name in self.commands or command_name in self.commands_aliases

    def _import_module(self, module_name: str) -> Any:
        return importlib.import_module(module_name)

    def _reload_module(self, module: Any) -> Any:
        return importlib.reload(module)

    def register(self, cmd: Command) -> None:
        if cmd.name in self.commands:
            logger.warn(
                f"Command '{cmd.name}' already registered and will be overwritten!"
            )
        self.commands[cmd.name] = cmd

        if cmd.name in self.commands_aliases:
            logger.warn(
                f"Command '{cmd.name}' will overwrite alias with the same name of "
                f"'{self.commands_aliases[cmd.name]}'!"
            )
        for alias in cmd.aliases:
            self.commands_aliases[alias] = cmd

    def unregister(self, command: Command) -> None:
        if command.name in self.commands:
            del self.commands[command.name]
            for alias in command.aliases:
                del self.commands_aliases[alias]
        else:
            raise KeyError(f"Command '{command.name}' not found in registry.")

    def reload_commands(self) -> None:
        """Reloads all loaded command plugins."""
        for cmd_name in self.commands:
            cmd = self.commands[cmd_name]
            module = self._import_module(cmd.__module__)
            reloaded_module = self._reload_module(module)
            if hasattr(reloaded_module, "register"):
                reloaded_module.register(self)

    def get_command(self, name: str) -> Command | None:
        if name in self.commands:
            return self.commands[name]

        if name in self.commands_aliases:
            return self.commands_aliases[name]

    def call(self, command_name: str, agent: BaseAgent, **kwargs) -> Any:
        if command := self.get_command(command_name):
            return command(**kwargs, agent=agent)
        raise KeyError(f"Command '{command_name}' not found in registry")

    def list_available_commands(self, agent: BaseAgent) -> Iterator[Command]:
        """Iterates over all registered commands and yields those that are available.

        Params:
            agent (BaseAgent): The agent that the commands will be checked against.

        Yields:
            Command: The next available command.
        """

        for cmd in self.commands.values():
            available = cmd.available
            if callable(cmd.available):
                available = cmd.available(agent)
            if available:
                yield cmd

    # def command_specs(self) -> str:
    #     """Returns a technical declaration of all commands in the registry for use in a prompt"""
    #
    #     Declaring functions or commands should be done in a model-specific way to achieve
    #     optimal results. For this reason, it should NOT be implemented here, but in an
    #     LLM provider module.
    #     MUST take command AVAILABILITY into account.

    @staticmethod
    def with_command_modules(modules: list[str], config: Config) -> CommandRegistry:
        new_registry = CommandRegistry()

        logger.debug(
            f"The following command categories are disabled: {config.disabled_command_categories}"
        )
        enabled_command_modules = [
            x for x in modules if x not in config.disabled_command_categories
        ]

        logger.debug(
            f"The following command categories are enabled: {enabled_command_modules}"
        )

        for command_module in enabled_command_modules:
            new_registry.import_command_module(command_module)

        # Unregister commands that are incompatible with the current config
        for command in [c for c in new_registry.commands.values()]:
            if callable(command.enabled) and not command.enabled(config):
                new_registry.unregister(command)
                logger.debug(
                    f"Unregistering incompatible command '{command.name}':"
                    f" \"{command.disabled_reason or 'Disabled by current config.'}\""
                )

        return new_registry

    def import_command_module(self, module_name: str) -> None:
        """
        Imports the specified Python module containing command plugins.

        This method imports the associated module and registers any functions or
        classes that are decorated with the `AUTO_GPT_COMMAND_IDENTIFIER` attribute
        as `Command` objects. The registered `Command` objects are then added to the
        `commands` dictionary of the `CommandRegistry` object.

        Args:
            module_name (str): The name of the module to import for command plugins.
        """

        module = importlib.import_module(module_name)

        category = self.register_module_category(module)

        for attr_name in dir(module):
            attr = getattr(module, attr_name)

            command = None

            # Register decorated functions
            if getattr(attr, AUTO_GPT_COMMAND_IDENTIFIER, False):
                command = attr.command

            # Register command classes
            elif (
                inspect.isclass(attr) and issubclass(attr, Command) and attr != Command
            ):
                command = attr()

            if command:
                self.register(command)
                category.commands.append(command)

    def register_module_category(self, module: ModuleType) -> CommandCategory:
        if not (category_name := getattr(module, "COMMAND_CATEGORY", None)):
            raise ValueError(f"Cannot import invalid command module {module.__name__}")

        if category_name not in self.categories:
            self.categories[category_name] = CommandRegistry.CommandCategory(
                name=category_name,
                title=getattr(
                    module, "COMMAND_CATEGORY_TITLE", category_name.capitalize()
                ),
                description=getattr(module, "__doc__", ""),
            )

        category = self.categories[category_name]
        if module not in category.modules:
            category.modules.append(module)

        return category

```