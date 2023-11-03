# AutoGPT源码解析 22

# `autogpts/forge/forge/sdk/memory/chroma_memstore.py`

This is a Python class that provides methods for interacting with a MemStore database.

It has a `query` method that allows for searching for documents based on filters and the value of a `document_search` parameter.

The `get` method retrieves a single document from the MemStore based on its ID and optionally retrieves a list of documents matching a given filter.

The `update` method updates the specified document or documents in the MemStore.

The `delete` method deletes a document from the MemStore.

The MemStore database is automatically created when the class is instantiated, and the class inherits from the `docs.Document` class.


```py
from .memstore import MemStore

import chromadb
from chromadb.config import Settings
import hashlib


class ChromaMemStore:
    """
    A class used to represent a Memory Store
    """

    def __init__(self, store_path: str):
        """
        Initialize the MemStore with a given store path.

        Args:
            store_path (str): The path to the store.
        """
        self.client = chromadb.PersistentClient(
            path=store_path, settings=Settings(anonymized_telemetry=False)
        )

    def add(self, task_id: str, document: str, metadatas: dict) -> None:
        """
        Add a document to the MemStore.

        Args:
            task_id (str): The ID of the task.
            document (str): The document to be added.
            metadatas (dict): The metadata of the document.
        """
        doc_id = hashlib.sha256(document.encode()).hexdigest()[:20]
        collection = self.client.get_or_create_collection(task_id)
        collection.add(documents=[document], metadatas=[metadatas], ids=[doc_id])

    def query(
        self,
        task_id: str,
        query: str,
        filters: dict = None,
        document_search: dict = None,
    ) -> dict:
        """
        Query the MemStore.

        Args:
            task_id (str): The ID of the task.
            query (str): The query string.
            filters (dict, optional): The filters to be applied. Defaults to None.
            search_string (str, optional): The search string. Defaults to None.

        Returns:
            dict: The query results.
        """
        collection = self.client.get_or_create_collection(task_id)

        kwargs = {
            "query_texts": [query],
            "n_results": 10,
        }

        if filters:
            kwargs["where"] = filters

        if document_search:
            kwargs["where_document"] = document_search

        return collection.query(**kwargs)

    def get(self, task_id: str, doc_ids: list = None, filters: dict = None) -> dict:
        """
        Get documents from the MemStore.

        Args:
            task_id (str): The ID of the task.
            doc_ids (list, optional): The IDs of the documents to be retrieved. Defaults to None.
            filters (dict, optional): The filters to be applied. Defaults to None.

        Returns:
            dict: The retrieved documents.
        """
        collection = self.client.get_or_create_collection(task_id)
        kwargs = {}
        if doc_ids:
            kwargs["ids"] = doc_ids
        if filters:
            kwargs["where"] = filters
        return collection.get(**kwargs)

    def update(self, task_id: str, doc_ids: list, documents: list, metadatas: list):
        """
        Update documents in the MemStore.

        Args:
            task_id (str): The ID of the task.
            doc_ids (list): The IDs of the documents to be updated.
            documents (list): The updated documents.
            metadatas (list): The updated metadata.
        """
        collection = self.client.get_or_create_collection(task_id)
        collection.update(ids=doc_ids, documents=documents, metadatas=metadatas)

    def delete(self, task_id: str, doc_id: str):
        """
        Delete a document from the MemStore.

        Args:
            task_id (str): The ID of the task.
            doc_id (str): The ID of the document to be deleted.
        """
        collection = self.client.get_or_create_collection(task_id)
        collection.delete(ids=[doc_id])


```

这段代码是一个Python脚本，它解释了一个名为“MemStore”的数据存储组件的交互式使用方法。MemStore是一个二进制内存数据结构，用于存储存储器中的数据，该数据结构使用机器码(x86二进制文件)。

该脚本的主要目的是以下任务：

1. 初始化MemStore：创建一个名为“.agent_mem_store”的MemStore，并将其初始化为空。
2. 添加数据：向MemStore中添加一些数据，包括一些测试文档和相应的元数据。
3. 查询MemStore：通过MemStore中存储的查询函数，查询MemStore中的数据。
4. 获取MemStore：通过MemStore中存储的获取函数，获取MemStore中的数据。
5. 更新MemStore：通过MemStore中存储的更新函数，更新MemStore中的数据。
6. 删除MemStore：通过MemStore中存储的删除函数，从MemStore中删除指定的数据。
7. 测试代码：运行脚本并输出结果，包括查询、获取、更新和删除等操作的结果。


```py
if __name__ == "__main__":
    print("#############################################")
    # Initialize MemStore
    mem = ChromaMemStore(".agent_mem_store")

    # Test add function
    task_id = "test_task"
    document = "This is a another new test document."
    metadatas = {"metadata": "test_metadata"}
    mem.add(task_id, document, metadatas)

    task_id = "test_task"
    document = "The quick brown fox jumps over the lazy dog."
    metadatas = {"metadata": "test_metadata"}
    mem.add(task_id, document, metadatas)

    task_id = "test_task"
    document = "AI is a new technology that will change the world."
    metadatas = {"timestamp": 1623936000}
    mem.add(task_id, document, metadatas)

    doc_id = hashlib.sha256(document.encode()).hexdigest()[:20]
    # Test query function
    query = "test"
    filters = {"metadata": {"$eq": "test"}}
    search_string = {"$contains": "test"}
    doc_ids = [doc_id]
    documents = ["This is an updated test document."]
    updated_metadatas = {"metadata": "updated_test_metadata"}

    print("Query:")
    print(mem.query(task_id, query))

    # Test get function
    print("Get:")

    print(mem.get(task_id))

    # Test update function
    print("Update:")
    print(mem.update(task_id, doc_ids, documents, updated_metadatas))

    print("Delete:")
    # Test delete function
    print(mem.delete(task_id, doc_ids[0]))

```

# `autogpts/forge/forge/sdk/memory/memstore.py`

This is a class that manages the MemStore of a specific collection MemStore.
It provides several methods for adding, updating, querying, and deleting documents and metadata within that collection.
The class is abstract and the methods are marked as abstractmethods which means that they must be implemented by the class if usage is to add those features.


```py
import abc
import hashlib

import chromadb
from chromadb.config import Settings


class MemStore(abc.ABC):
    """
    An abstract class that represents a Memory Store
    """

    @abc.abstractmethod
    def __init__(self, store_path: str):
        """
        Initialize the MemStore with a given store path.

        Args:
            store_path (str): The path to the store.
        """
        pass

    @abc.abstractmethod
    def add_task_memory(self, task_id: str, document: str, metadatas: dict) -> None:
        """
        Add a document to the current tasks MemStore.
        This function calls the base version with the task_id as the collection_name.

        Args:
            task_id (str): The ID of the task.
            document (str): The document to be added.
            metadatas (dict): The metadata of the document.
        """
        self.add(collection_name=task_id, document=document, metadatas=metadatas)

    @abc.abstractmethod
    def query_task_memory(
        self,
        task_id: str,
        query: str,
        filters: dict = None,
        document_search: dict = None,
    ) -> dict:
        """
        Query the current tasks MemStore.
        This function calls the base version with the task_id as the collection_name.

        Args:
            task_id (str): The ID of the task.
            query (str): The query string.
            filters (dict, optional): The filters to be applied. Defaults to None.
            document_search (dict, optional): The search string. Defaults to None.

        Returns:
            dict: The query results.
        """
        return self.query(
            collection_name=task_id,
            query=query,
            filters=filters,
            document_search=document_search,
        )

    @abc.abstractmethod
    def get_task_memory(
        self, task_id: str, doc_ids: list = None, filters: dict = None
    ) -> dict:
        """
        Get documents from the current tasks MemStore.
        This function calls the base version with the task_id as the collection_name.

        Args:
            task_id (str): The ID of the task.
            doc_ids (list, optional): The IDs of the documents to be retrieved. Defaults to None.
            filters (dict, optional): The filters to be applied. Defaults to None.

        Returns:
            dict: The retrieved documents.
        """
        return self.get(collection_name=task_id, doc_ids=doc_ids, filters=filters)

    @abc.abstractmethod
    def update_task_memory(
        self, task_id: str, doc_ids: list, documents: list, metadatas: list
    ):
        """
        Update documents in the current tasks MemStore.
        This function calls the base version with the task_id as the collection_name.

        Args:
            task_id (str): The ID of the task.
            doc_ids (list): The IDs of the documents to be updated.
            documents (list): The updated documents.
            metadatas (list): The updated metadata.
        """
        self.update(
            collection_name=task_id,
            doc_ids=doc_ids,
            documents=documents,
            metadatas=metadatas,
        )

    @abc.abstractmethod
    def delete_task_memory(self, task_id: str, doc_id: str):
        """
        Delete a document from the current tasks MemStore.
        This function calls the base version with the task_id as the collection_name.

        Args:
            task_id (str): The ID of the task.
            doc_id (str): The ID of the document to be deleted.
        """
        self.delete(collection_name=task_id, doc_id=doc_id)

    @abc.abstractmethod
    def add(self, collection_name: str, document: str, metadatas: dict) -> None:
        """
        Add a document to the current collection's MemStore.

        Args:
            collection_name (str): The name of the collection.
            document (str): The document to be added.
            metadatas (dict): The metadata of the document.
        """
        pass

    @abc.abstractmethod
    def query(
        self,
        collection_name: str,
        query: str,
        filters: dict = None,
        document_search: dict = None,
    ) -> dict:
        pass

    @abc.abstractmethod
    def get(
        self, collection_name: str, doc_ids: list = None, filters: dict = None
    ) -> dict:
        pass

    @abc.abstractmethod
    def update(
        self, collection_name: str, doc_ids: list, documents: list, metadatas: list
    ):
        pass

    @abc.abstractmethod
    def delete(self, collection_name: str, doc_id: str):
        pass

```

# `autogpts/forge/forge/sdk/memory/memstore_test.py`

这段代码的作用是测试一个名为“test_mem_store”的内存存储器对象。该内存存储器被放置在当前工作目录的根目录下，名为“.test_mem_store”。

具体来说，这段代码使用Python标准库中的两个库：hashlib和shutil。hashlib库用于创建一个哈希表，而shutil库用于文件和目录的遍历、删除和压缩。

在代码中，首先通过import hashlib和import shutil的函数来导入哈希表和文件操作库。然后，定义了一个名为“memstore”的变量，该变量将作为fixture用于pytest的测试用例中。

接着，从forge.sdk.memory.memstore库中创建一个名为“test_mem_store”的内存存储器对象，并将其赋值给memstore变量。在内存存储器对象上使用yield语句，将其内容作为测试用例返回，以便在测试用例中对其进行使用。最后，使用shutil.rmtree函数删除内存存储器中的所有内容，并确保在函数内部Python标准库中的所有函数和库都处于导入状态。


```py
import hashlib
import shutil

import pytest

from forge.sdk.memory.memstore import ChromaMemStore


@pytest.fixture
def memstore():
    mem = ChromaMemStore(".test_mem_store")
    yield mem
    shutil.rmtree(".test_mem_store")


```

这两函数是在测试中用测试数据存儲器(memstore)进行操作。

```pypython
def test_add(memstore):
   # 在memstore中添加一个名为"test_task"的任务，一个名为"This is a test document."的文档，以及一个名为"test_metadata"的元数据。
   task_id = "test_task"
   document = "This is a test document."
   metadatas = {"metadata": "test_metadata"}
   memstore.add(task_id, document, metadatas)
   doc_id = hashlib.sha256(document.encode()).hexdigest()[:20]
   assert memstore.client.get_or_create_collection(task_id).count() == 1
```

这段代码模拟了一个向测试数据存储器中添加文档、元数据和创建集合操作。它创建了一个名为"test_task"的任务，将一个名为"This is a test document."的文档和名为"test_metadata"的元数据添加到了任务中。然后，它计算了文档的哈希值并将其存储到了名为"doc_id"的变量中。最后，它使用memstore的client方法获取名为"test_task"的集合，并检查返回的集合中包含的文档数量是否为1。

```pypython
def test_query(memstore):
   # 在memstore中添加一个名为"test_task"的任务，一个名为"This is a test document."的文档，以及一个名为"test_metadata"的元数据。
   task_id = "test_task"
   document = "This is a test document."
   metadatas = {"metadata": "test_metadata"}
   memstore.add(task_id, document, metadatas)
   query = "test"
   assert len(memstore.query(task_id, query)["documents"]) == 1
```

这段代码模拟了一个向测试数据存储器中查询元数据和返回文档的操作。它创建了一个名为"test_task"的任务，将一个名为"This is a test document."的文档和名为"test_metadata"的元数据添加到了任务中。然后，它使用query参数查询名为"test_task"的集合，并将其存储到的变量中。最后，它使用len()函数获取返回的集合中包含的文档数量，并将其存储到print中。


```py
def test_add(memstore):
    task_id = "test_task"
    document = "This is a test document."
    metadatas = {"metadata": "test_metadata"}
    memstore.add(task_id, document, metadatas)
    doc_id = hashlib.sha256(document.encode()).hexdigest()[:20]
    assert memstore.client.get_or_create_collection(task_id).count() == 1


def test_query(memstore):
    task_id = "test_task"
    document = "This is a test document."
    metadatas = {"metadata": "test_metadata"}
    memstore.add(task_id, document, metadatas)
    query = "test"
    assert len(memstore.query(task_id, query)["documents"]) == 1


```

这段代码的作用是测试一个名为 "test_update" 的函数，该函数对一个名为 "memstore" 的存储器进行操作。其主要目的是在存储器中添加一个测试文档，然后测试存储器中是否包含关于该文档的信息，包括其元数据、哈希值和嵌入。

具体来说，函数首先定义了一个名为 "task_id" 的变量，用于标识要更新的测试任务。接下来，定义了一个名为 "document" 的变量，用于存储要更新的测试文档。接着，定义了一个名为 "metadatas" 的变量，其中包含一个名为 "metadata" 的元数据，其值为 "test_metadata"。

接下来，使用 Python 的 "hashlib" 库中的 "sha256" 函数对 "document" 进行哈希，并将其作为元数据的一部分添加到 "memstore" 中。然后，"memstore" 函数使用 "update" 方法对给定的 "task_id" 和元数据（包括 "document" 和 "metadatas"）进行更新，其中更新后的元数据包含一个名为 "updated_document" 的文档和一个名为 "updated_metadatas" 的元数据。

最后，函数使用 "assert" 语句测试 "memstore.get" 函数在给定的 "task_id" 和元数据的情况下是否正确地返回了包含更新后的文档、元数据和嵌入的元数据，以及哈希值。如果返回的元数据中包含 "updated_document"，则说明函数成功地更新了存储器，否则可能会引发一些错误。


```py
def test_update(memstore):
    task_id = "test_task"
    document = "This is a test document."
    metadatas = {"metadata": "test_metadata"}
    memstore.add(task_id, document, metadatas)
    doc_id = hashlib.sha256(document.encode()).hexdigest()[:20]
    updated_document = "This is an updated test document."
    updated_metadatas = {"metadata": "updated_test_metadata"}
    memstore.update(task_id, [doc_id], [updated_document], [updated_metadatas])
    assert memstore.get(task_id, [doc_id]) == {
        "documents": [updated_document],
        "metadatas": [updated_metadatas],
        "embeddings": None,
        "ids": [doc_id],
    }


```

这段代码是一个测试用例，名为 `test_delete`，属于一个名为 `memstore` 的内存存储库。该函数的作用是验证 `memstore` 库中一个名为 `test_delete` 的删除操作是否可以成功删除一个文档，并确保该文档在删除后，由于删除时使用到了哈希算法，文档的哈希值会被被删除。

具体来说，代码首先定义了一个名为 `task_id` 的变量，用于存储要测试的文档的唯一标识；接着定义了一个名为 `document` 的变量，用于存储要删除的文档的哈希值；接着定义了一个名为 `metadatas` 的变量，用于存储文档元数据，该变量在 `memstore` 被添加到文档和元数据之后会被保存。

在 `test_delete` 函数中，我们先将 `task_id` 和 `document` 以及 `metadatas` 存储到 `memstore` 中，然后使用哈希算法生成一个文档的哈希值，并将其存储到 `metadatas` 中，最后使用 `memstore.delete` 方法删除了该文档，同时使用哈希表存储技术 `hashlib` 中的 `sha256` 方法对文档进行哈希，将哈希值存储到 `metadatas` 中。

接着，代码使用 `assert` 语句来验证 `memstore.client.get_or_create_collection(task_id)` 函数在文档被成功删除后，返回的集合是否为空集，如果为空集，说明文档已经被成功删除，否则说明失败。


```py
def test_delete(memstore):
    task_id = "test_task"
    document = "This is a test document."
    metadatas = {"metadata": "test_metadata"}
    memstore.add(task_id, document, metadatas)
    doc_id = hashlib.sha256(document.encode()).hexdigest()[:20]
    memstore.delete(task_id, doc_id)
    assert memstore.client.get_or_create_collection(task_id).count() == 0

```

# `autogpts/forge/forge/sdk/memory/__init__.py`

这段代码的作用是创建两个名为MemStore的类对象，一个是从Memstore类中创建的，另一个是从ChromaMemStore类中创建的。MemStore是一个用于在多个进程之间共享内存数据的类，而ChromaMemStore是一个具体的MemStore子类，专为Chroma生态系统设计。通过创建这两个MemStore对象，可以实现不同进程之间的数据共享，有助于代码的正确性和高效性。


```py
from .memstore import MemStore
from .chroma_memstore import ChromaMemStore

```

# `autogpts/forge/forge/sdk/routes/agent_protocol.py`

这段代码定义了针对Agent服务的API路由。这个模块包含了多个API端点，但有一些端点因为其复杂性需要特别注意。

1. `execute_agent_task_step`:
这个路由是Agent服务中实际执行工作的函数。函数根据当前任务的狀態来执行下一步，而且需要特别小心以确保所有情景(如步骤的存在或缺失，或者指定为`last_step`的步骤)都得到正确处理。

2. `upload_agent_task_artifacts`:
这个路由允许上传各种URI类型的任务 artifacts。支持常见的URI类型(例如s3, gcs, ftp和http)。这个端点比`execute_agent_task_step`更加复杂，因此需要特别注意确保支持的所有URI类型都得到正确处理。请注意，AutoGPT团队将最终处理最常见的URI类型。


```py
"""
Routes for the Agent Service.

This module defines the API routes for the Agent service. While there are multiple endpoints provided by the service,
the ones that require special attention due to their complexity are:

1. `execute_agent_task_step`:
   This route is significant because this is where the agent actually performs the work. The function handles
   executing the next step for a task based on its current state, and it requires careful implementation to ensure
   all scenarios (like the presence or absence of steps or a step marked as `last_step`) are handled correctly.

2. `upload_agent_task_artifacts`:
   This route allows for the upload of artifacts, supporting various URI types (e.g., s3, gcs, ftp, http).
   The support for different URI types makes it a bit more complex, and it's important to ensure that all
   supported URI types are correctly managed. NOTE: The AutoGPT team will eventually handle the most common
   uri types for you.

```

这段代码是一个名为`create_agent_task`的函数，它在系统工作流程中起着关键作用，因为它负责创建一个新的任务。

while这段代码是一个较为简单的路线，但它们在系统中的行为中扮演着至关重要的角色。因此，开发人员和贡献者应格外小心，在对这些路线进行修改时确保一致性和正确性。


```py
3. `create_agent_task`:
   While this is a simpler route, it plays a crucial role in the workflow, as it's responsible for the creation
   of a new task.

Developers and contributors should be especially careful when making modifications to these routes to ensure
consistency and correctness in the system's behavior.
"""
import json
from typing import Optional

from fastapi import APIRouter, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse

from forge.sdk.errors import *
from forge.sdk.forge_log import ForgeLogger
```

这段代码使用了Python的一个名为Forge的高层次API开发框架。Forge提供了一个API路由器和ForgeLogger两个核心组件。

这段代码具体的作用如下：

1. 从Forge.sdk.schema模块中导入了所有定义了API路由的类。
2. 创建了一个名为base_router的API路由器。
3. 创建了一个名为LOG的ForgeLogger实例，用于输出日志信息。
4. 定义了一个名为root的API路由，该路由器返回一个欢迎消息。
5. 在该路由器的实现中，使用了ForgeLogger的调试输出功能，将类的名称和路由器的名称作为参数传递给Logger实例，以便输出调试信息。
6. 在该路由器的具体实现中，直接返回一个字符串"Welcome to the AutoGPT Forge"，作为路由器的响应。


```py
from forge.sdk.schema import *

base_router = APIRouter()

LOG = ForgeLogger(__name__)


@base_router.get("/", tags=["root"])
async def root():
    """
    Root endpoint that returns a welcome message.
    """
    return Response(content="Welcome to the AutoGPT Forge")


```

This is a FastAPI router that defines two endpoints.

The first endpoint is a GET endpoint at the path '/heartbeat' that is annotated with the tag 'server'. This endpoint simply returns a response indicating that the server is running.

The second endpoint is a POST endpoint at the path '/agent/tasks' that is annotated with the tag 'agent'. This endpoint takes a TaskRequestBody object as input and returns a Task object.

The TaskRequestBody object contains information about the task that was submitted by the agent, such as the task ID, the task data, and any additional input data.

The create_agent_task function is responsible for creating a new task using the information provided in the TaskRequestBody and returning the newly created task.

This router can be used by a client to check the status of the server and or to submit a task for the agent.


```py
@base_router.get("/heartbeat", tags=["server"])
async def check_server_status():
    """
    Check if the server is running.
    """
    return Response(content="Server is running.", status_code=200)


@base_router.post("/agent/tasks", tags=["agent"], response_model=Task)
async def create_agent_task(request: Request, task_request: TaskRequestBody) -> Task:
    """
    Creates a new task using the provided TaskRequestBody and returns a Task.

    Args:
        request (Request): FastAPI request object.
        task (TaskRequestBody): The task request containing input and additional input data.

    Returns:
        Task: A new task with task_id, input, additional_input, and empty lists for artifacts and steps.

    Example:
        Request (TaskRequestBody defined in schema.py):
            {
                "input": "Write the words you receive to the file 'output.txt'.",
                "additional_input": "python/code"
            }

        Response (Task defined in schema.py):
            {
                "task_id": "50da533e-3904-4401-8a07-c49adf88b5eb",
                "input": "Write the word 'Washington' to a .txt file",
                "additional_input": "python/code",
                "artifacts": [],
            }
    """
    agent = request["agent"]

    try:
        task_request = await agent.create_task(task_request)
        return Response(
            content=task_request.json(),
            status_code=200,
            media_type="application/json",
        )
    except Exception:
        LOG.exception(f"Error whilst trying to create a task: {task_request}")
        return Response(
            content=json.dumps({"error": "Internal server error"}),
            status_code=500,
            media_type="application/json",
        )


```

Here is the implementation of the `list_tasks` function using FastAPI:
```py
import json
from datetime import datetime, timedelta

from fastapi import FastAPI, Response
from pydantic import BaseModel
from typing import List, Any

app = FastAPI()

class Task(BaseModel):
   input: str
   additional_input: str = None
   task_id: str = None
   steps: list = []
   artifacts: list = []
   截止日期： datetime = datetime.utcnow()

class TaskListResponse(BaseModel):
   items: List[Task] = [];
   pagination: dict = {
       "total": 0,
       "pages": 0,
       "current": 0,
       "page_size": 0
   }

@app.get("/agent/tasks")
async def list_tasks(agent: str, page: int = 1, page_size: int = 10, **kwargs) -> Response:
   """
   FastAPI endpoint to fetch a list of tasks for an agent.

   Args:
       agent (str): The agent whose tasks should be fetched.
       page (int, optional): The page number for pagination. Defaults to 1.
       page_size (int, optional): The number of tasks per page for pagination. Defaults to 10.

   Returns:
       A FastAPI response object containing a list of tasks and pagination details.
   """
   # Connect to the FastAPI agent (http://localhost:8000/).
   agent.connect()

   # Fetch the list of tasks from the agent.
   tasks = await agent.list_tasks(page, page_size)

   # Compute the pagination details.
   if page > 0:
       pagination = {
           "total": tasks.json()["tasks"],
           "pages": math.ceil(len(tasks.json()) / page_size),
           "current": page,
           "page_size": page_size
       }
   else:
       pagination = {
           "total": tasks.json()["paging"],
           "pages": 1,
           "current": 1,
           "page_size": 10
       }

   # Return the response object with the list of tasks and pagination details.
   return Response(
       content=tasks.json(),
       status_code=200,
       media_type="application/json",
       pagination=pagination
   )
```
This implementation uses the `list_tasks` function as the main entry point for the FastAPI agent. When a client sends a GET request to the agent for a list of tasks, the agent connects to a FastAPI endpoint (http://localhost:8000/) and calls the `list_tasks` function with the required parameters (page, page\_size). The `list_tasks` function fetches the list of tasks from the agent and returns the response as the HTTP 200 status code with a JSON content type.

The response object contains a list of tasks, along with the pagination details. The `pagination` field is a dictionary that maps the different pages of the tasks list to the corresponding page numbers, the number of tasks per page, and the current page number.

If the client is not interested in the pagination, the `pagination` field will be an empty dictionary.


```py
@base_router.get("/agent/tasks", tags=["agent"], response_model=TaskListResponse)
async def list_agent_tasks(
    request: Request,
    page: Optional[int] = Query(1, ge=1),
    page_size: Optional[int] = Query(10, ge=1),
) -> TaskListResponse:
    """
    Retrieves a paginated list of all tasks.

    Args:
        request (Request): FastAPI request object.
        page (int, optional): The page number for pagination. Defaults to 1.
        page_size (int, optional): The number of tasks per page for pagination. Defaults to 10.

    Returns:
        TaskListResponse: A response object containing a list of tasks and pagination details.

    Example:
        Request:
            GET /agent/tasks?page=1&pageSize=10

        Response (TaskListResponse defined in schema.py):
            {
                "items": [
                    {
                        "input": "Write the word 'Washington' to a .txt file",
                        "additional_input": null,
                        "task_id": "50da533e-3904-4401-8a07-c49adf88b5eb",
                        "artifacts": [],
                        "steps": []
                    },
                    ...
                ],
                "pagination": {
                    "total": 100,
                    "pages": 10,
                    "current": 1,
                    "pageSize": 10
                }
            }
    """
    agent = request["agent"]
    try:
        tasks = await agent.list_tasks(page, page_size)
        return Response(
            content=tasks.json(),
            status_code=200,
            media_type="application/json",
        )
    except NotFoundError:
        LOG.exception("Error whilst trying to list tasks")
        return Response(
            content=json.dumps({"error": "Tasks not found"}),
            status_code=404,
            media_type="application/json",
        )
    except Exception:
        LOG.exception("Error whilst trying to list tasks")
        return Response(
            content=json.dumps({"error": "Internal server error"}),
            status_code=500,
            media_type="application/json",
        )


```

It looks like you are trying to provide an AWS Lambda function that can be deployed to AWS Lambda. The function is configured to write the value "Washington" to a file called "output.txt".

The JSON object in the `task.json()` method is likely passed by the client and contains the configuration for the task. The `additional_input` field is used to specify the client to use for the task.

It is also notable that the function returns a response, which suggests that it is returning some kind of output after writing the value "Washington" to the file. The response is specified in the `additional_output` field and can be used to continue the execution of the function.


```py
@base_router.get("/agent/tasks/{task_id}", tags=["agent"], response_model=Task)
async def get_agent_task(request: Request, task_id: str) -> Task:
    """
    Gets the details of a task by ID.

    Args:
        request (Request): FastAPI request object.
        task_id (str): The ID of the task.

    Returns:
        Task: The task with the given ID.

    Example:
        Request:
            GET /agent/tasks/50da533e-3904-4401-8a07-c49adf88b5eb

        Response (Task defined in schema.py):
            {
                "input": "Write the word 'Washington' to a .txt file",
                "additional_input": null,
                "task_id": "50da533e-3904-4401-8a07-c49adf88b5eb",
                "artifacts": [
                    {
                        "artifact_id": "7a49f31c-f9c6-4346-a22c-e32bc5af4d8e",
                        "file_name": "output.txt",
                        "agent_created": true,
                        "relative_path": "file://50da533e-3904-4401-8a07-c49adf88b5eb/output.txt"
                    }
                ],
                "steps": [
                    {
                        "task_id": "50da533e-3904-4401-8a07-c49adf88b5eb",
                        "step_id": "6bb1801a-fd80-45e8-899a-4dd723cc602e",
                        "input": "Write the word 'Washington' to a .txt file",
                        "additional_input": "challenge:write_to_file",
                        "name": "Write to file",
                        "status": "completed",
                        "output": "I am going to use the write_to_file command and write Washington to a file called output.txt <write_to_file('output.txt', 'Washington')>",
                        "additional_output": "Do you want me to continue?",
                        "artifacts": [
                            {
                                "artifact_id": "7a49f31c-f9c6-4346-a22c-e32bc5af4d8e",
                                "file_name": "output.txt",
                                "agent_created": true,
                                "relative_path": "file://50da533e-3904-4401-8a07-c49adf88b5eb/output.txt"
                            }
                        ],
                        "is_last": true
                    }
                ]
            }
    """
    agent = request["agent"]
    try:
        task = await agent.get_task(task_id)
        return Response(
            content=task.json(),
            status_code=200,
            media_type="application/json",
        )
    except NotFoundError:
        LOG.exception(f"Error whilst trying to get task: {task_id}")
        return Response(
            content=json.dumps({"error": "Task not found"}),
            status_code=404,
            media_type="application/json",
        )
    except Exception:
        LOG.exception(f"Error whilst trying to get task: {task_id}")
        return Response(
            content=json.dumps({"error": "Internal server error"}),
            status_code=500,
            media_type="application/json",
        )


```

This is a Python class that represents the TaskStepsListResponse object.

It has the following methods:

* `__init__`: Initializes the object with the task ID, page number, and page size.
* `get_task_steps`: Returns the list of steps for the task based on the given page and page size.
* `get_pagination`: Returns the pagination details, such as the total number of steps, the number of pages, the current page, and the page size.
* `json`: Returns the JSON string representation of the object.
* `raise_not_found_error`: Raises a `NotFoundError` if the task steps are not found.
* `raise_internal_server_error`: Raises an `InternalServerError` if the server raises an error.

Example usage:
```pyscss
async def fetch_task_steps(task_id, page, page_size):
   response = await client.get(
       f"https://example.com/agent/tasks/{task_id}",
       params={
           "page": page,
           "page_size": page_size,
       }
   )
   response.raise_not_found_error()
   response_task_steps = await response.json()
   return response_task_steps

async def main():
   client = ApiClient("https://example.com")
   task_id = "50da533e-3904-4401-8a07-c49adf88b5eb"
   page = 1
   page_size = 10
   response = await fetch_task_steps(task_id, page, page_size)
   LOG.debug(response)

if __name__ == "__main__":
   main()
```


```py
@base_router.get(
    "/agent/tasks/{task_id}/steps", tags=["agent"], response_model=TaskStepsListResponse
)
async def list_agent_task_steps(
    request: Request,
    task_id: str,
    page: Optional[int] = Query(1, ge=1),
    page_size: Optional[int] = Query(10, ge=1, alias="pageSize"),
) -> TaskStepsListResponse:
    """
    Retrieves a paginated list of steps associated with a specific task.

    Args:
        request (Request): FastAPI request object.
        task_id (str): The ID of the task.
        page (int, optional): The page number for pagination. Defaults to 1.
        page_size (int, optional): The number of steps per page for pagination. Defaults to 10.

    Returns:
        TaskStepsListResponse: A response object containing a list of steps and pagination details.

    Example:
        Request:
            GET /agent/tasks/50da533e-3904-4401-8a07-c49adf88b5eb/steps?page=1&pageSize=10

        Response (TaskStepsListResponse defined in schema.py):
            {
                "items": [
                    {
                        "task_id": "50da533e-3904-4401-8a07-c49adf88b5eb",
                        "step_id": "step1_id",
                        ...
                    },
                    ...
                ],
                "pagination": {
                    "total": 100,
                    "pages": 10,
                    "current": 1,
                    "pageSize": 10
                }
            }
    """
    agent = request["agent"]
    try:
        steps = await agent.list_steps(task_id, page, page_size)
        return Response(
            content=steps.json(),
            status_code=200,
            media_type="application/json",
        )
    except NotFoundError:
        LOG.exception("Error whilst trying to list steps")
        return Response(
            content=json.dumps({"error": "Steps not found"}),
            status_code=404,
            media_type="application/json",
        )
    except Exception:
        LOG.exception("Error whilst trying to list steps")
        return Response(
            content=json.dumps({"error": "Internal server error"}),
            status_code=500,
            media_type="application/json",
        )


```

The `Step` class is a data structure that represents the details of a step in an Azure API Management (APIM) agent. It contains information about the step, such as the step ID, the agent's反馈 on the step, and any additional output details.

The `StepRequestBody` class is used to define the details of the step. It is used to pass the step data as a request body to the Azure API Management agent. In the example, the `input` parameter is used to provide the details of the step.

The `execute_step` method is used to execute the step. It takes two arguments, the task ID and the step details defined in the `StepRequestBody`. This method returns the result of the step execution in a JSON object.

In the example response, the `task_id` is set to the step ID and the `output` is the feedback returned by the agent. Additionally, there are additional output details specified in the `additional_output` parameter of the `StepRequestBody`.


```py
@base_router.post("/agent/tasks/{task_id}/steps", tags=["agent"], response_model=Step)
async def execute_agent_task_step(
    request: Request, task_id: str, step: Optional[StepRequestBody] = None
) -> Step:
    """
    Executes the next step for a specified task based on the current task status and returns the
    executed step with additional feedback fields.

    Depending on the current state of the task, the following scenarios are supported:

    1. No steps exist for the task.
    2. There is at least one step already for the task, and the task does not have a completed step marked as `last_step`.
    3. There is a completed step marked as `last_step` already on the task.

    In each of these scenarios, a step object will be returned with two additional fields: `output` and `additional_output`.
    - `output`: Provides the primary response or feedback to the user.
    - `additional_output`: Supplementary information or data. Its specific content is not strictly defined and can vary based on the step or agent's implementation.

    Args:
        request (Request): FastAPI request object.
        task_id (str): The ID of the task.
        step (StepRequestBody): The details for executing the step.

    Returns:
        Step: Details of the executed step with additional feedback.

    Example:
        Request:
            POST /agent/tasks/50da533e-3904-4401-8a07-c49adf88b5eb/steps
            {
                "input": "Step input details...",
                ...
            }

        Response:
            {
                "task_id": "50da533e-3904-4401-8a07-c49adf88b5eb",
                "step_id": "step1_id",
                "output": "Primary feedback...",
                "additional_output": "Supplementary details...",
                ...
            }
    """
    agent = request["agent"]
    try:
        # An empty step request represents a yes to continue command
        if not step:
            step = StepRequestBody(input="y")

        step = await agent.execute_step(task_id, step)
        return Response(
            content=step.json(),
            status_code=200,
            media_type="application/json",
        )
    except NotFoundError:
        LOG.exception(f"Error whilst trying to execute a task step: {task_id}")
        return Response(
            content=json.dumps({"error": f"Task not found {task_id}"}),
            status_code=404,
            media_type="application/json",
        )
    except Exception as e:
        LOG.exception(f"Error whilst trying to execute a task step: {task_id}")
        return Response(
            content=json.dumps({"error": "Internal server error"}),
            status_code=500,
            media_type="application/json",
        )


```

这段代码是一个 FastAPI 路由装饰器，它定义了一个名为 `get_agent_task_step` 的函数，用于获取指定任务和步骤的详细信息。

具体来说，这个函数接受两个参数：一个`Request`对象和一个任务ID和一个步骤ID，它返回一个名为`Step`的对象。

函数内部使用了两个务必方法：`asyncio.get_event_logger()`和`jasyncio.get_step()`。`asyncio.get_event_logger()`用于记录请求的代理对象的日志，`jasyncio.get_step()`用于获取指定任务和步骤的详细信息。

如果请求代理对象中没有`agent`标签，或者指定的任务和步骤不存在，函数将会抛出`NotFoundError`异常并返回一个`404`状态码的响应。否则，函数将会尝试从请求代理对象中检索指定的任务和步骤，并返回一个包含任务和步骤详细信息的响应。


```py
@base_router.get(
    "/agent/tasks/{task_id}/steps/{step_id}", tags=["agent"], response_model=Step
)
async def get_agent_task_step(request: Request, task_id: str, step_id: str) -> Step:
    """
    Retrieves the details of a specific step for a given task.

    Args:
        request (Request): FastAPI request object.
        task_id (str): The ID of the task.
        step_id (str): The ID of the step.

    Returns:
        Step: Details of the specific step.

    Example:
        Request:
            GET /agent/tasks/50da533e-3904-4401-8a07-c49adf88b5eb/steps/step1_id

        Response:
            {
                "task_id": "50da533e-3904-4401-8a07-c49adf88b5eb",
                "step_id": "step1_id",
                ...
            }
    """
    agent = request["agent"]
    try:
        step = await agent.get_step(task_id, step_id)
        return Response(content=step.json(), status_code=200)
    except NotFoundError:
        LOG.exception(f"Error whilst trying to get step: {step_id}")
        return Response(
            content=json.dumps({"error": "Step not found"}),
            status_code=404,
            media_type="application/json",
        )
    except Exception:
        LOG.exception(f"Error whilst trying to get step: {step_id}")
        return Response(
            content=json.dumps({"error": "Internal server error"}),
            status_code=500,
            media_type="application/json",
        )


```

This is an example of a FastAPI function that retrieves a paginated list of artifacts associated with a specific task. The function takes a request object and a task ID as input, and options for page and page size. It returns a TaskArtifactsListResponse object containing a list of artifacts and pagination details.

If the task is not found or the paginated list is empty, the function returns a Response object with an error message.


```py
@base_router.get(
    "/agent/tasks/{task_id}/artifacts",
    tags=["agent"],
    response_model=TaskArtifactsListResponse,
)
async def list_agent_task_artifacts(
    request: Request,
    task_id: str,
    page: Optional[int] = Query(1, ge=1),
    page_size: Optional[int] = Query(10, ge=1, alias="pageSize"),
) -> TaskArtifactsListResponse:
    """
    Retrieves a paginated list of artifacts associated with a specific task.

    Args:
        request (Request): FastAPI request object.
        task_id (str): The ID of the task.
        page (int, optional): The page number for pagination. Defaults to 1.
        page_size (int, optional): The number of items per page for pagination. Defaults to 10.

    Returns:
        TaskArtifactsListResponse: A response object containing a list of artifacts and pagination details.

    Example:
        Request:
            GET /agent/tasks/50da533e-3904-4401-8a07-c49adf88b5eb/artifacts?page=1&pageSize=10

        Response (TaskArtifactsListResponse defined in schema.py):
            {
                "items": [
                    {"artifact_id": "artifact1_id", ...},
                    {"artifact_id": "artifact2_id", ...},
                    ...
                ],
                "pagination": {
                    "total": 100,
                    "pages": 10,
                    "current": 1,
                    "pageSize": 10
                }
            }
    """
    agent = request["agent"]
    try:
        artifacts: TaskArtifactsListResponse = await agent.list_artifacts(
            task_id, page, page_size
        )
        return artifacts
    except NotFoundError:
        LOG.exception("Error whilst trying to list artifacts")
        return Response(
            content=json.dumps({"error": "Artifacts not found for task_id"}),
            status_code=404,
            media_type="application/json",
        )
    except Exception:
        LOG.exception("Error whilst trying to list artifacts")
        return Response(
            content=json.dumps({"error": "Internal server error"}),
            status_code=500,
            media_type="application/json",
        )


```

This is a Flask endpoint that handles the file upload for an artifact. The file is specified by the `relative_path` parameter in the request URL, and the file is stored in the `file` parameter. The endpoint uses the `agent` object in the request to upload the file and returns the artifact object. If there is an error or the file is not found, the endpoint returns an error message or a server error.


```py
@base_router.post(
    "/agent/tasks/{task_id}/artifacts", tags=["agent"], response_model=Artifact
)
async def upload_agent_task_artifacts(
    request: Request, task_id: str, file: UploadFile, relative_path: Optional[str] = ""
) -> Artifact:
    """
    This endpoint is used to upload an artifact associated with a specific task. The artifact is provided as a file.

    Args:
        request (Request): The FastAPI request object.
        task_id (str): The unique identifier of the task for which the artifact is being uploaded.
        file (UploadFile): The file being uploaded as an artifact.
        relative_path (str): The relative path for the file. This is a query parameter.

    Returns:
        Artifact: An object containing metadata of the uploaded artifact, including its unique identifier.

    Example:
        Request:
            POST /agent/tasks/50da533e-3904-4401-8a07-c49adf88b5eb/artifacts?relative_path=my_folder/my_other_folder
            File: <uploaded_file>

        Response:
            {
                "artifact_id": "b225e278-8b4c-4f99-a696-8facf19f0e56",
                "created_at": "2023-01-01T00:00:00Z",
                "modified_at": "2023-01-01T00:00:00Z",
                "agent_created": false,
                "relative_path": "/my_folder/my_other_folder/",
                "file_name": "main.py"
            }
    """
    agent = request["agent"]

    if file is None:
        return Response(
            content=json.dumps({"error": "File must be specified"}),
            status_code=404,
            media_type="application/json",
        )
    try:
        artifact = await agent.create_artifact(task_id, file, relative_path)
        return Response(
            content=artifact.json(),
            status_code=200,
            media_type="application/json",
        )
    except Exception:
        LOG.exception(f"Error whilst trying to upload artifact: {task_id}")
        return Response(
            content=json.dumps({"error": "Internal server error"}),
            status_code=500,
            media_type="application/json",
        )


```

这段代码是一个 FastAPI 路由装饰器，它定义了一个 HTTP 方法，用于下载特定任务下的指定工件。

具体来说，这个方法接收两个参数：任务 ID 和工件 ID。在 method 内部，使用 `@base_router.get()` 装饰器来发送 HTTP GET 请求，并在路由装饰器中定义了请求的路由标签（tags）为 "agent"。

请求收到后会执行 `agent.get_artifact()` 方法来下载对应任务下的工件，如果工件不存在，则会产生一个错误并返回相应的错误信息。如果下载成功，则会将下载的工件内容作为文件内容返回。

可以认为这个方法的作用是下载特定任务下的指定工件，用于快速评估（agent）是否可用。


```py
@base_router.get(
    "/agent/tasks/{task_id}/artifacts/{artifact_id}", tags=["agent"], response_model=str
)
async def download_agent_task_artifact(
    request: Request, task_id: str, artifact_id: str
) -> FileResponse:
    """
    Downloads an artifact associated with a specific task.

    Args:
        request (Request): FastAPI request object.
        task_id (str): The ID of the task.
        artifact_id (str): The ID of the artifact.

    Returns:
        FileResponse: The downloaded artifact file.

    Example:
        Request:
            GET /agent/tasks/50da533e-3904-4401-8a07-c49adf88b5eb/artifacts/artifact1_id

        Response:
            <file_content_of_artifact>
    """
    agent = request["agent"]
    try:
        return await agent.get_artifact(task_id, artifact_id)
    except NotFoundError:
        LOG.exception(f"Error whilst trying to download artifact: {task_id}")
        return Response(
            content=json.dumps(
                {
                    "error": f"Artifact not found - task_id: {task_id}, artifact_id: {artifact_id}"
                }
            ),
            status_code=404,
            media_type="application/json",
        )
    except Exception:
        LOG.exception(f"Error whilst trying to download artifact: {task_id}")
        return Response(
            content=json.dumps(
                {
                    "error": f"Internal server error - task_id: {task_id}, artifact_id: {artifact_id}"
                }
            ),
            status_code=500,
            media_type="application/json",
        )

```

# `autogpts/forge/forge/sdk/routes/__init__.py`

很抱歉，我无法不输出源代码。请提供需要解释的代码，以便我为您提供详细的解释。


```py

```

You developed a tool that could help people build agents?

Fork this repository, integrate your tool to the forge and send us the link of your fork in the autogpt discord: https://discord.gg/autogpt (ping maintainers)

PS: make sure the way you integrate your tool allows for easy rebases from upstream.


## [AutoGPT Forge Part 1: A Comprehensive Guide to Your First Steps](https://aiedge.medium.com/autogpt-forge-a-comprehensive-guide-to-your-first-steps-a1dfdf46e3b4)

![Header](../../../docs/content/imgs/quickstart/000_header_img.png)

**Written by Craig Swift & [Ryan Brandt](https://github.com/paperMoose)**


Welcome to the getting started Tutorial! This tutorial is designed to walk you through the process of setting up and running your own AutoGPT agent in the Forge environment. Whether you are a seasoned AI developer or just starting out, this guide will equip you with the necessary steps to jumpstart your journey in the world of AI development with AutoGPT.

## Section 1: Understanding the Forge

The Forge serves as a comprehensive template for building your own AutoGPT agent. It not only provides the setting for setting up, creating, and running your agent, but also includes the benchmarking system and the frontend for testing it. We'll touch more on those later! For now just think of the forge as a way to easily generate your boilerplate in a standardized way.

## Section 2: Setting up the Forge Environment

To begin, you need to fork the [repository](https://github.com/Significant-Gravitas/AutoGPT) by navigating to the main page of the repository and clicking **Fork** in the top-right corner. 

![The Github repository](../../../docs/content/imgs/quickstart/001_repo.png)

Follow the on-screen instructions to complete the process. 

![Create Fork Page](../../../docs/content/imgs/quickstart/002_fork.png)

### Cloning the Repository
Next, clone your newly forked repository to your local system. Ensure you have Git installed to proceed with this step. You can download Git from [here](https://git-scm.com/downloads). Then clone the repo using the following command and the url for your repo. You can find the correct url by clicking on the green Code button on your repos main page.
![img_1.png](../../../docs/content/imgs/quickstart/003A_clone.png)

```pybash
# replace the url with the one for your forked repo
git clone https://github.com/<YOUR REPO PATH HERE>
```

![Clone the Repository](../../../docs/content/imgs/quickstart/003_clone.png)

### Setting up the Project

Once you have clone the project change your directory to the newly cloned project:
```pybash
# The name of the directory will match the name you gave your fork. The default is AutoGPT
cd AutoGPT
```
To set up the project, utilize the `./run setup` command in the terminal. Follow the instructions to install necessary dependencies and set up your GitHub access token.

![Setup the Project](../../../docs/content/imgs/quickstart/005_setup.png)
![Setup Complete](../../../docs/content/imgs/quickstart/006_setup_complete.png)

## Section 3: Creating Your Agent

Choose a suitable name for your agent. It should be unique and descriptive. Examples of valid names include swiftyosgpt, SwiftyosAgent, or swiftyos_agent.

Create your agent template using the command:

```pybash
 ./run agent create YOUR_AGENT_NAME
 ```
 Replacing YOUR_AGENT_NAME with the name you chose in the previous step.

![Create an Agent](../../../docs/content/imgs/quickstart/007_create_agent.png)

### Entering the Arena 
The Arena is a collection of all AutoGPT agents ranked by performance on our benchmark. Entering the Arena is a required step for participating in AutoGPT hackathons. It's early days, so show us what you've got!

Officially enter the Arena by executing the command:

```pybash
./run arena enter YOUR_AGENT_NAME
```

![Enter the Arena](../../../docs/content/imgs/quickstart/008_enter_arena.png)

## Section 4: Running Your Agent

Begin by starting your agent using the command:

```pybash
./run agent start YOUR_AGENT_NAME
```
This will initiate the agent on `http://localhost:8000/`.

![Start the Agent](../../../docs/content/imgs/quickstart/009_start_agent.png)

### Logging in and Sending Tasks to Your Agent
Access the frontend at `http://localhost:8000/` and log in using a Google or GitHub account. Once you're logged you'll see the agent tasking interface! However... the agent won't do anything yet. We'll implement the logic for our agent to run tasks in the upcoming tutorial chapters. 

![Login](../../../docs/content/imgs/quickstart/010_login.png)
![Home](../../../docs/content/imgs/quickstart/011_home.png)

### Stopping and Restarting Your Agent
When needed, use Ctrl+C to end the session or use the stop command:
```pybash
./run agent stop
``` 
This command forcefully stops the agent. You can also restart it using the start command.

## To Recap
- We've forked the AutoGPT repo and cloned it locally on your machine.
- we connected the library with our personal github access token as part of the setup.
- We've created and named our first agent, and entered it into the arena!
- We've run the agent and it's tasking server successfully without an error.
- We've logged into the server site at localhost:8000 using our github account.

Make sure you've completed every step successfully before moving on :). 
### Next Steps: Building and Enhancing Your Agent
With our foundation set, you are now ready to build and enhance your agent! The next tutorial will look into the anatomy of an agent and how to add basic functionality.

## Additional Resources

### Links to Documentation and Community Forums
- [Windows Subsystem for Linux (WSL) Installation](https://learn.microsoft.com/en-us/windows/wsl/)
- [Git Download](https://git-scm.com/downloads)

## Appendix

### Troubleshooting Common Issues
- Ensure Git is correctly installed before cloning the repository.
- Follow the setup instructions carefully to avoid issues during project setup.
- If encountering issues during agent creation, refer to the guide for naming conventions.
- make sure your github token has the `repo` scopes toggled. 

### Glossary of Terms
- **Repository**: A storage space where your project resides.
- **Forking**: Creating a copy of a repository under your GitHub account.
- **Cloning**: Making a local copy of a repository on your system.
- **Agent**: The AutoGPT you will be creating and developing in this project.
- **Benchmarking**: The process of testing your agent's skills in various categories using the Forge's integrated benchmarking system.
- **Forge**: The comprehensive template for building your AutoGPT agent, including the setting for setup, creation, running, and benchmarking your agent.
- **Frontend**: The user interface where you can log in, send tasks to your agent, and view the task history.


### System Requirements

This project supports Linux (Debian based), Mac, and Windows Subsystem for Linux (WSL). If you are using a Windows system, you will need to install WSL. You can find the installation instructions for WSL [here](https://learn.microsoft.com/en-us/windows/wsl/).


# AutoGPT Forge Part 2: The Blueprint of an AI Agent

**Written by Craig Swift & [Ryan Brandt](https://github.com/paperMoose)**

*8 min read*  


---

![Header](../../../docs/content/imgs/quickstart/t2_01.png)





## What are LLM-Based AI Agents?

Before we add logic to our new agent, we have to understand what an agent actually IS. 

Large Language Models (LLMs) are state-of-the-art machine learning models that harness vast amounts of web knowledge. But what happens when you give the LLM the ability to use tools based on it's output? You get LLM-based AI agents — a new breed of artificial intelligence that promises more human-like decision-making in the real world.  

Traditional autonomous agents operated with limited knowledge, often confined to specific tasks or environments. They were like calculators — efficient but limited to predefined functions. LLM-based agents, on the other hand don’t just compute; they understand, reason, and then act, drawing from a vast reservoir of information.  

![AI visualising AI researchers hard at work](../../../docs/content/imgs/quickstart/t2_02.png)


## The Anatomy of an LLM-Based AI Agent

Diving deep into the core of an LLM-based AI agent, we find it’s structured much like a human, with distinct components akin to personality, memory, thought process, and abilities. Let’s break these down:  

![The Github repository](../../../docs/content/imgs/quickstart/t2_03.png)
Anatomy of an Agent from the Agent Landscape Survey  

### **Profile**  
Humans naturally adapt our mindset based on the tasks we're tackling, whether it's writing, cooking, or playing sports. Similarly, agents can be conditioned or "profiled" to specialize in specific tasks.

The profile of an agent is it's personality, mindset, and high-level instructions. Research indicates that merely informing an agent that it's an expert in a certain domain can boost its performance.

| **Potential Applications of Profiling** | **Description**                                                                                          |
|-----------------------------------------|----------------------------------------------------------------------------------------------------------|
| **Prompt Engineering**                  | Tailoring agent prompts for better results.                                                              |
| **Memory Adjustments**                  | Modifying how an agent recalls or prioritizes information.                                               |
| **Action Selection**                    | Influencing the set of actions an agent might consider.                                                  |
| **Driving Mechanism**                   | Potentially tweaking the underlying large language model (LLM) that powers the agent.                    |

#### Example Agent Profile: Weather Expert

- **Profile Name:** Weather Specialist
- **Purpose:** Provide detailed and accurate weather information.
- **Preferred Memory Sources:** Meteorological databases, recent weather news, and scientific journals.
- **Action Set:** Fetching weather data, analyzing weather patterns, and providing forecasts.
- **Base Model Tweaks:** Prioritize meteorological terminology and understanding.

### **Memory**  
Just as our memories shape our decisions, reactions, and identities, an agent's memory is the cornerstone of its identity and capabilities. Memory is fundamental for an agent to learn and adapt. At a high level, agents possess two core types of memories: long-term and short-term.

|                   | **Long-Term Memory**                                                                                         | **Short-Term (Working) Memory**                                                                                 |
|-------------------|-------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| **Purpose**       | Serves as the agent's foundational knowledge base.                                                           | Handles recent or transient memories, much like our recollection of events from the past few days.              |
| **What it Stores**| Historical data and interactions that have taken place over extended periods.                                | Immediate experiences and interactions.                                                                         |
| **Role**          | Guides the agent's core behaviors and understanding, acting as a vast reservoir of accumulated knowledge.   | Essential for real-time tasks and decision-making. Not all these memories transition into long-term storage.     |


### **Planning**  
Planning is essential for agents to systematically tackle challenges, mirroring how humans break down complex problems into smaller tasks.
#### **1. What is Planning?**

- **Concept:** It's the agent's strategy for problem-solving, ensuring solutions are both comprehensive and systematic.
- **Human Analogy:** Just like humans split challenges into smaller, more manageable tasks, agents adopt a similar methodical approach.

#### **2. Key Planning Strategies**

| **Strategy**               | **Description**                                                                                           |
|----------------------------|----------------------------------------------------------------------------------------------------------|
| **Planning with Feedback** | An adaptive approach where agents refine their strategy based on outcomes, similar to iterative design processes.|
| **Planning without Feedback** | The agent acts as a strategist, using only its existing knowledge. It's like playing chess, anticipating challenges and planning several moves ahead. |

### **Action**  
After the introspection of memory and the strategizing of planning, comes the finale: Action. This is where the agent’s cognitive processes manifest into tangible outcomes using the agents Abilities. Every decision, every thought, culminates in the action phase, translating abstract concepts into definitive results.  
Whether it’s penning a response, saving a file, or initiating a new process, the action component is the culmination of the agent’s decision-making journey. It’s the bridge between digital cognition and real-world impact, turning the agent’s electronic impulses into meaningful and purposeful outcomes.  

![t2_agent_flow.png](..%2F..%2F..%2Fdocs%2Fcontent%2Fimgs%2Fquickstart%2Ft2_agent_flow.png)
An example of how a basic agent works
## The Agent Protocol: The Linguistics of AI Communication

After diving deep into the anatomy of an agent, understanding its core components, there emerges a pivotal question: How do we effectively communicate with these diverse, intricately-designed agents? The answer lies in the Agent Protocol.  

### Understanding the Agent Protocol

At its essence, the Agent Protocol is a standardized communication interface, a universal “language” that every AI agent, regardless of its underlying structure or design, can comprehend. Think of it as the diplomatic envoy that ensures smooth conversations between agents and their developers, tools, or even other agents.  

In an ecosystem where every developer might have their unique approach to crafting agents, the Agent Protocol acts as a unifying bridge. It’s akin to a standardized plug fitting into any socket or a universal translator decoding myriad languages.  

## AutoGPT Forge: A Peek Inside the LLM Agent Template

Now we understand the architecture of an agent lets look inside the Forge. It’s a well-organized template, meticulously architected to cater to the needs of agent developers.

#### Forge’s Project Structure: A Bird’s-Eye View
![t2_diagram.png](..%2F..%2F..%2Fdocs%2Fcontent%2Fimgs%2Fquickstart%2Ft2_diagram.png)

The Forge's agent directory structure consists of three parts:
- **agent.py**: The heart of the Forge, where the agent's actual business logic is.  
- **prompts**: A directory of prompts used in agent.py's LLM logic.  
- **sdk**: The boilerplate code and the lower level APIs of the Forge.  

Let’s break them down.

#### Understanding the SDK

The SDK is the main directory for the Forge. Here's a breakdown:

- **Core Components**: These are key parts of the Forge including Memory, Abilities, and Planning. They help the agent think and act.
- **Agent Protocol Routes**: In the routes sub-directory, you'll see the Agent Protocol. This is how the agent communicates.
- **Database (db.py)**: This is where the agent stores its data like experiences and learnings.
- **Prompting Engine (prompting.py)**: This tool uses templates to ask questions to the LLM for consistent interactions.
- **Agent Class**: This connects the agent's actions with the Agent Protocol routes.

#### Configurations and Environment

Configuration is key to ensuring our agent runs seamlessly. The .env.example file provides a template for setting up the necessary environment variables. Before diving into the Forge, developers need to copy this to a new .env file and adjust the settings:  
- **API Key**: `OPENAI_API_KEY` is where you plug in your OpenAI API key.  
- **Log Level**: With `LOG_LEVEL`, control the verbosity of the logs.  
- **Database Connection**: `DATABASE_STRING` determines where and how the agent's data gets stored.  
- **Port**: `PORT` specifies the listening port for the agent's server.  
- **Workspace**: `AGENT_WORKSPACE` points to the agent's working directory.  

## To Recap

- **LLM-Based AI Agents**: 
  - LLMs are machine learning models with vast knowledge. When equipped with tools to utilize their outputs, they evolve into LLM-based AI agents, enabling human-like decision-making.

- **Anatomy of an Agent**: 
  - **Profile**: Sets an agent's personality and specialization.
  - **Memory**: Encompasses the agent's long-term and short-term memory, storing both historical data and recent interactions.
  - **Planning**: The strategy the agent employs to tackle problems.
  - **Action**: The stage where the agent's decisions translate to tangible results.
  
- **Agent Protocol**: 
  - A uniform communication interface ensuring smooth interactions between agents and their developers.

- **AutoGPT Forge**: 
  - A foundational template for creating agents. Components include:
    - **agent.py**: Houses the agent's core logic.
    - **prompts**: Directory of templates aiding LLM logic.
    - **sdk**: Boilerplate code and essential APIs.

Let's put this blueprint into practice in part 3!