# AutoGPT源码解析 34

# `benchmark/tests/test_benchmark_workflow.py`

这段代码使用了参数化测试，目的是通过传入不同的参数值来测试不同情况下的行为。

具体来说，代码通过 `URL_BENCHMARK` 和 `URL_AGENT` 获取两个 HTTP 服务的地址。然后，定义了一个 `datetime.datetime` 类的实例 `start_time`，以及一个 `time.sleep` 类的实例 `sleep_time`。

接着，在测试套件中定义了一个 `@pytest.mark.parametrize` 装饰器，用于将 `eval_id`、`input_text`、`expected_artifact_length`、`test_name` 和 `should_be_successful` 作为参数，并传入一个参数组。在每个参数组合中，使用 `eval_id` 和 `input_text` 生成一个测试用例，并使用 `start_time` 和 `sleep_time` 来模拟实际测试中可能需要的时间。最后，在 `test_name` 参数中定义了一个测试名称，用于记录测试结果。

在测试用例的参数组合中，第一个测试用例的参数组合使用了 `eval_id`、`input_text`、`expected_artifact_length`、`test_name` 和 `should_be_successful=True`。这意味着该测试用例将尝试使用 `eval_id` 和 `input_text` 生成一个适当的 .txt 文件，并将其命名为 `output.txt`。如果该文件已存在，则测试将失败，并将产生一个错误消息。如果该文件不存在，则测试将成功，并将产生一个成功消息。

第二个测试用例的参数组合使用了 `eval_id`、`input_text`、`expected_artifact_length` 和 `should_be_successful=False`。这意味着该测试用例将尝试使用 `eval_id` 和 `input_text` 生成一个适当的 .txt 文件，并将其命名为 `file_to_read.txt`。如果该文件存在，则测试将失败，并将产生一个错误消息。如果该文件不存在，则测试将成功，并将产生一个成功消息。


```py
import pytest
import requests

URL_BENCHMARK = "http://localhost:8080/ap/v1"
URL_AGENT = "http://localhost:8000/ap/v1"

import datetime
import time


@pytest.mark.parametrize(
    "eval_id, input_text, expected_artifact_length, test_name, should_be_successful",
    [
        (
            "021c695a-6cc4-46c2-b93a-f3a9b0f4d123",
            "Write the word 'Washington' to a .txt file",
            0,
            "WriteFile",
            True,
        ),
        (
            "f219f3d3-a41b-45a9-a3d0-389832086ee8",
            "Read the file called file_to_read.txt and write its content to a file called output.txt",
            1,
            "ReadFile",
            False,
        ),
    ],
)
```

This is a Python test case that checks if the task responded correctly to the benchmarking task. The `task_response_benchmark` object is created with a timestamp before the task is started and another timestamp after the task is evaluated. The `assert` statements in the test case verify that the correct information was returned by the API calls and that the task is expected to respond correctly.

The `task_response_benchmark_id` is compared to the `task_id` in the API call to make sure that the correct task was referenced. The `response_task_agent` is a第二次 API call to the task agent endpoint to check the status of the task. The `response_task_agent` is expected to have a `200` status code and a response that contains the `artifacts` field, which should contain the `expected_artifact_length` number of steps.

The `step_request` is a dictionary that contains the `input` field for the benchmarking step. This step is taken after the task is evaluated and the `is_last` field is set to `True`, indicating that this is the last step of the task.

The `step_response` is a request to the API endpoint for the step, which is expected to have a `200` status code and a response that contains the `is_last` field as `True`.

The `eval_response` is a request to the API endpoint for the task evaluation. This endpoint is expected to have a `200` status code and a response that contains the `run_details` field, which should contain the `test_name` and `metrics` fields that verify that the task completed successfully and within the expected time frame.


```py
def test_entire_workflow(
    eval_id, input_text, expected_artifact_length, test_name, should_be_successful
):
    task_request = {"eval_id": eval_id, "input": input_text}
    response = requests.get(f"{URL_AGENT}/agent/tasks")
    task_count_before = response.json()["pagination"]["total_items"]
    # First POST request
    task_response_benchmark = requests.post(
        URL_BENCHMARK + "/agent/tasks", json=task_request
    )
    response = requests.get(f"{URL_AGENT}/agent/tasks")
    task_count_after = response.json()["pagination"]["total_items"]
    assert task_count_after == task_count_before + 1

    timestamp_after_task_eval_created = datetime.datetime.now(datetime.timezone.utc)
    time.sleep(1.1)  # To make sure the 2 timestamps to compare are different
    assert task_response_benchmark.status_code == 200
    task_response_benchmark = task_response_benchmark.json()
    assert task_response_benchmark["input"] == input_text

    task_response_benchmark_id = task_response_benchmark["task_id"]

    response_task_agent = requests.get(
        f"{URL_AGENT}/agent/tasks/{task_response_benchmark_id}"
    )
    assert response_task_agent.status_code == 200
    response_task_agent = response_task_agent.json()
    assert len(response_task_agent["artifacts"]) == expected_artifact_length

    step_request = {"input": input_text}

    step_response = requests.post(
        URL_BENCHMARK + "/agent/tasks/" + task_response_benchmark_id + "/steps",
        json=step_request,
    )
    assert step_response.status_code == 200
    step_response = step_response.json()
    assert step_response["is_last"] == True  # Assuming is_last is always True

    eval_response = requests.post(
        URL_BENCHMARK + "/agent/tasks/" + task_response_benchmark_id + "/evaluations",
        json={},
    )
    assert eval_response.status_code == 200
    eval_response = eval_response.json()
    print("eval_response")
    print(eval_response)
    assert eval_response["run_details"]["test_name"] == test_name
    assert eval_response["metrics"]["success"] == should_be_successful
    benchmark_start_time = datetime.datetime.fromisoformat(
        eval_response["run_details"]["benchmark_start_time"]
    )

    assert benchmark_start_time < timestamp_after_task_eval_created

```

# `benchmark/tests/test_extract_subgraph.py`

这段代码是用来创建一个课程表（curriculum graph）的，其中每个节点代表一个科目，每个边代表科目之间的课程转移。

具体来说，这个curriculum graph包括以下几个部分：

* 顶点（node）: 每个科目都有自己的id和label，表示这个科目是什么，例如"Calculus"代表数学，"Advanced Calculus"代表高级数学。
* 边（edge）: 每条边连接两个科目，表示这两个科目之间可以学习什么课程。例如，"Calculus"和"Advanced Calculus"之间的边表示学生可以学习高级数学。

通过这个curriculum graph，你可以理解不同科目之间的课程联系，以及哪些科目之间存在跳转。可以用于各种各样的教育或学术应用，例如智能教育系统中，帮助学生了解他们的课程进度和未来的学习目标。


```py
import pytest

from agbenchmark.utils.dependencies.graphs import extract_subgraph_based_on_category


@pytest.fixture
def curriculum_graph():
    return {
        "edges": [
            {"from": "Calculus", "to": "Advanced Calculus"},
            {"from": "Algebra", "to": "Calculus"},
            {"from": "Biology", "to": "Advanced Biology"},
            {"from": "World History", "to": "Modern History"},
        ],
        "nodes": [
            {"data": {"category": ["math"]}, "id": "Calculus", "label": "Calculus"},
            {
                "data": {"category": ["math"]},
                "id": "Advanced Calculus",
                "label": "Advanced Calculus",
            },
            {"data": {"category": ["math"]}, "id": "Algebra", "label": "Algebra"},
            {"data": {"category": ["science"]}, "id": "Biology", "label": "Biology"},
            {
                "data": {"category": ["science"]},
                "id": "Advanced Biology",
                "label": "Advanced Biology",
            },
            {
                "data": {"category": ["history"]},
                "id": "World History",
                "label": "World History",
            },
            {
                "data": {"category": ["history"]},
                "id": "Modern History",
                "label": "Modern History",
            },
        ],
    }


```

这段代码定义了一个名为 "graph\_example" 的字典对象，包含了一些节点和边的数据。

其中， nodes 对象表示为字典形式存储的节点，每个节点包含一个 id 和一个数据对象，数据对象是一个列表，代表该节点的类别。

edges 对象表示为字典形式存储的边，每个边包含两个节点，一个 from 节点，一个 to 节点。

接着，定义了一个名为 "test\_dfs\_category\_math" 的函数，该函数接受一个名为 "curriculum\_graph" 的参数。

函数的作用是提取给定的 "curriculum\_graph" 对象中，所有节点属于 "math" 类别的子图，并将子图转换成一个只包含 "math" 类别的图。

最后，通过 `extract_subgraph_based_on_category` 函数，将子图转换成一个只包含 "math" 类别的图，然后使用 `set` 函数检查节点的 ID 和边是否属于预期结果。


```py
graph_example = {
    "nodes": [
        {"id": "A", "data": {"category": []}},
        {"id": "B", "data": {"category": []}},
        {"id": "C", "data": {"category": ["math"]}},
    ],
    "edges": [{"from": "B", "to": "C"}, {"from": "A", "to": "C"}],
}


def test_dfs_category_math(curriculum_graph):
    result_graph = extract_subgraph_based_on_category(curriculum_graph, "math")

    # Expected nodes: Algebra, Calculus, Advanced Calculus
    # Expected edges: Algebra->Calculus, Calculus->Advanced Calculus

    expected_nodes = ["Algebra", "Calculus", "Advanced Calculus"]
    expected_edges = [
        {"from": "Algebra", "to": "Calculus"},
        {"from": "Calculus", "to": "Advanced Calculus"},
    ]

    assert set(node["id"] for node in result_graph["nodes"]) == set(expected_nodes)
    assert set((edge["from"], edge["to"]) for edge in result_graph["edges"]) == set(
        (edge["from"], edge["to"]) for edge in expected_edges
    )


```

这两函数测试了如何根据给定的数学类别，从指定的图中提取出子图。第一个函数测试了当从给定的图中提取出基于类别的子图时，检查它是否由预期的节点组成。第二个函数测试了当从给定的图中提取出非存在类别的子图时，它是否包含预期的节点和边缘。


```py
def test_extract_subgraph_math_category():
    subgraph = extract_subgraph_based_on_category(graph_example, "math")
    assert set(
        (node["id"], tuple(node["data"]["category"])) for node in subgraph["nodes"]
    ) == set(
        (node["id"], tuple(node["data"]["category"])) for node in graph_example["nodes"]
    )
    assert set((edge["from"], edge["to"]) for edge in subgraph["edges"]) == set(
        (edge["from"], edge["to"]) for edge in graph_example["edges"]
    )


def test_extract_subgraph_non_existent_category():
    result_graph = extract_subgraph_based_on_category(graph_example, "toto")

    # Asserting that the result graph has no nodes and no edges
    assert len(result_graph["nodes"]) == 0
    assert len(result_graph["edges"]) == 0

```

# `benchmark/tests/test_get_roots.py`

这段代码的作用是测试一个名为 `get_roots` 的函数，它接受一个 Agbenchmark 库中的图形数据作为输入参数。这个函数的作用是从图中找到所有根节点（即没有子图的节点）。

具体来说，这段代码定义了一个名为 `graph` 的字典，其中包含了一个根节点的数据，以及从根节点到其他节点的边。然后，定义了一个名为 `result` 的变量，用于存储经过筛选后的根节点集合。

接着，调用 `get_roots` 函数，并将 `graph` 作为参数传入。函数返回一个包含所有根节点的集合。

最后，通过 `assert` 语句对结果进行测试，如果预期的根节点集合与实际结果相同，则测试通过，否则测试失败。


```py
from agbenchmark.utils.dependencies.graphs import get_roots


def test_get_roots():
    graph = {
        "nodes": [
            {"id": "A", "data": {"category": []}},
            {"id": "B", "data": {"category": []}},
            {"id": "C", "data": {"category": []}},
            {"id": "D", "data": {"category": []}},
        ],
        "edges": [
            {"from": "A", "to": "B"},
            {"from": "B", "to": "C"},
        ],
    }

    result = get_roots(graph)
    assert set(result) == {
        "A",
        "D",
    }, f"Expected roots to be 'A' and 'D', but got {result}"


```

这段代码是一个测试用例，它的目的是测试一个名为`no_roots`的函数。函数接受一个名为`fully_connected_graph`的参数，这个参数是一个关于图表的 dictionary，它表示了一个完整的、有向的、连通的图表。函数使用`get_roots`函数来获取图表的根节点，然后使用`assert`语句来检查返回的结果是否为`None`，如果是，那么说明函数预期的结果是"没有根节点"，如果不是，那么函数的行为就会出错。


```py
def test_no_roots():
    fully_connected_graph = {
        "nodes": [
            {"id": "A", "data": {"category": []}},
            {"id": "B", "data": {"category": []}},
            {"id": "C", "data": {"category": []}},
        ],
        "edges": [
            {"from": "A", "to": "B"},
            {"from": "B", "to": "C"},
            {"from": "C", "to": "A"},
        ],
    }

    result = get_roots(fully_connected_graph)
    assert not result, "Expected no roots, but found some"


```

这段代码定义了一个名为 `test_no_rcoots()` 的函数，该函数使用了 Python 的 `get_roots()` 函数和自定义的 `fully_connected_graph` 数据结构。函数的作用是测试一个没有根的完全连接图是否有满拓扑。

具体来说，该函数创建了一个包含三个节点（A、B、C）的完全连接图，然后使用 `get_roots()` 函数获取该图的所有根。最后，函数使用 `assert` 语句检查返回的结果是否为预期结果，即检查是否有根。


```py
# def test_no_rcoots():
#     fully_connected_graph = {
#         "nodes": [
#             {"id": "A", "data": {"category": []}},
#             {"id": "B", "data": {"category": []}},
#             {"id": "C", "data": {"category": []}},
#         ],
#         "edges": [
#             {"from": "A", "to": "B"},
#             {"from": "D", "to": "C"},
#         ],
#     }
#
#     result = get_roots(fully_connected_graph)
#     assert set(result) == {"A"}, f"Expected roots to be 'A', but got {result}"

```

# `benchmark/tests/test_is_circular.py`

这段代码是一个测试函数，用于测试 Agbenchmark库中的is_circular函数的正确性。该函数接受一个环形图（或称环状图）作为输入，并返回一个布尔值，表示输入的图是否为环状图。

具体来说，这段代码创建了一个包含四个节点的环形图，其中有两个节点没有数据（或称为外部节点），这些节点不会对结果产生影响。然后，它使用is_circular函数检查该图是否为环状图。如果检测到该图是环状图，函数将返回True，否则返回False。

函数的核心是判断该图是否为环状图。环状图是一种特殊类型的图，其中所有度（或称为邻居数量）等于节点数量减1。例如，一个包含5个节点的图，其度为4，可以被视为一个环状图，因为每个节点都有且仅有一个邻居。

为了实现这个功能，函数首先创建了一个包含四个节点的图，并检查该图是否具有环状结构。如果是环状图，它将检测到所有节点的度均为1，因此返回True。如果不是环状图，它将检测到至少两个节点的度为2，并返回False。


```py
from agbenchmark.utils.dependencies.graphs import is_circular


def test_is_circular():
    cyclic_graph = {
        "nodes": [
            {"id": "A", "data": {"category": []}},
            {"id": "B", "data": {"category": []}},
            {"id": "C", "data": {"category": []}},
            {"id": "D", "data": {"category": []}},  # New node
        ],
        "edges": [
            {"from": "A", "to": "B"},
            {"from": "B", "to": "C"},
            {"from": "C", "to": "D"},
            {"from": "D", "to": "A"},  # This edge creates a cycle
        ],
    }

    result = is_circular(cyclic_graph)
    assert result is not None, "Expected a cycle, but none was detected"
    assert all(
        (
            (result[i], result[i + 1])
            in [(x["from"], x["to"]) for x in cyclic_graph["edges"]]
        )
        for i in range(len(result) - 1)
    ), "The detected cycle path is not part of the graph's edges"


```

这段代码定义了一个函数 `test_is_not_circular()`，该函数通过创建一个非环形图（acyclic graph）来进行测试。在这个函数中，我们创建了一个包含四个节点的图，其中节点的数据是一个空列表（an empty list）。我们使用两个边缘连接节点 A 和节点 B，以及连接节点 B 和 C。我们创建了一个新节点 D，该节点的数据也为空。然后，我们使用 is_circular() 函数来检查该图是否为环形图。

is_circular() 函数用于检查一个图是否为环形图。它返回一个布尔值，如果图是环形图，则返回 False，否则返回 True。在这个函数中，我们检查图是否包含至少一个环。如果图是环形图，is_circular() 函数将返回 False，否则将返回 True。

在这个测试函数中，我们使用 assert 语句来检查函数是否返回预期值。如果函数返回 False，则说明该图是一个环形图，否则将返回 True。


```py
def test_is_not_circular():
    acyclic_graph = {
        "nodes": [
            {"id": "A", "data": {"category": []}},
            {"id": "B", "data": {"category": []}},
            {"id": "C", "data": {"category": []}},
            {"id": "D", "data": {"category": []}},  # New node
        ],
        "edges": [
            {"from": "A", "to": "B"},
            {"from": "B", "to": "C"},
            {"from": "C", "to": "D"},
            # No back edge from D to any node, so it remains acyclic
        ],
    }

    assert is_circular(acyclic_graph) is None, "Detected a cycle in an acyclic graph"

```

# `benchmark/tests/__init__.py`

很抱歉，我需要更多的上下文来回答您的问题。如果能提供更多上下文，我将非常乐意帮助理解代码的作用。


```py

```

../../CODE_OF_CONDUCT.md

../../CONTRIBUTING.md

# AutoGPT docs

Welcome to AutoGPT. Please follow the [Installation](/setup/) guide to get started.

!!! note
    It is recommended to use a virtual machine/container (docker) for tasks that require high security measures to prevent any potential harm to the main computer's system and data. If you are considering to use AutoGPT outside a virtualized/containerized environment, you are *strongly* advised to use a separate user account just for running AutoGPT. This is even more important if you are going to allow AutoGPT to write/execute scripts and run shell commands!

It is for these reasons that executing python scripts is explicitly disabled when running outside a container environment.


## Plugins

⚠️💀 **WARNING** 💀⚠️: Review the code of any plugin you use thoroughly, as plugins can execute any Python code, potentially leading to malicious activities, such as stealing your API keys.

To configure plugins, you can create or edit the `plugins_config.yaml` file in the root directory of AutoGPT. This file allows you to enable or disable plugins as desired. For specific configuration instructions, please refer to the documentation provided for each plugin. The file should be formatted in YAML. Here is an example for your reference:

```py
plugin_a:
  config:
    api_key: my-api-key
  enabled: false
plugin_b:
  config: {}
  enabled: true
```

See our [Plugins Repo](https://github.com/Significant-Gravitas/Auto-GPT-Plugins) for more info on how to install all the amazing plugins the community has built!

Alternatively, developers can use the [AutoGPT Plugin Template](https://github.com/Significant-Gravitas/Auto-GPT-Plugin-Template) as a starting point for creating your own plugins.



# Setting up AutoGPT

## 📋 Requirements

Choose an environment to run AutoGPT in (pick one):

  - [Docker](https://docs.docker.com/get-docker/) (*recommended*)
  - Python 3.10 or later (instructions: [for Windows](https://www.tutorialspoint.com/how-to-install-python-in-windows))
  - [VSCode + devcontainer](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)


## 🗝️ Getting an API key

Get your OpenAI API key from: [https://platform.openai.com/account/api-keys](https://platform.openai.com/account/api-keys).

!!! attention
    To use the OpenAI API with AutoGPT, we strongly recommend **setting up billing**
    (AKA paid account). Free accounts are [limited][openai/api limits] to 3 API calls per
    minute, which can cause the application to crash.

    You can set up a paid account at [Manage account > Billing > Overview](https://platform.openai.com/account/billing/overview).

[openai/api limits]: https://platform.openai.com/docs/guides/rate-limits/overview#:~:text=Free%20trial%20users,RPM%0A40%2C000%20TPM

!!! important
    It's highly recommended that you keep track of your API costs on [the Usage page](https://platform.openai.com/account/usage).
    You can also set limits on how much you spend on [the Usage limits page](https://platform.openai.com/account/billing/limits).

![For OpenAI API key to work, set up paid account at OpenAI API > Billing](./imgs/openai-api-key-billing-paid-account.png)


## Setting up AutoGPT

### Set up with Docker

1. Make sure you have Docker installed, see [requirements](#requirements)
2. Create a project directory for AutoGPT

    ```py
    mkdir AutoGPT
    cd AutoGPT
    ```

3. In the project directory, create a file called `docker-compose.yml` with the following contents:

    ```py
    version: "3.9"
    services:
      auto-gpt:
        image: significantgravitas/auto-gpt
        env_file:
          - .env
        profiles: ["exclude-from-up"]
        volumes:
          - ./auto_gpt_workspace:/app/auto_gpt_workspace
          - ./data:/app/data
          ## allow auto-gpt to write logs to disk
          - ./logs:/app/logs
          ## uncomment following lines if you want to make use of these files
          ## you must have them existing in the same folder as this docker-compose.yml
          #- type: bind
          #  source: ./azure.yaml
          #  target: /app/azure.yaml
          #- type: bind
          #  source: ./ai_settings.yaml
          #  target: /app/ai_settings.yaml
    ```

4. Create the necessary [configuration](#configuration) files. If needed, you can find
    templates in the [repository].
5. Pull the latest image from [Docker Hub]

    ```py
    docker pull significantgravitas/auto-gpt
    ```

6. Continue to [Run with Docker](#run-with-docker)

!!! note "Docker only supports headless browsing"
    AutoGPT uses a browser in headless mode by default: `HEADLESS_BROWSER=True`.
    Please do not change this setting in combination with Docker, or AutoGPT will crash.

[Docker Hub]: https://hub.docker.com/r/significantgravitas/auto-gpt
[repository]: https://github.com/Significant-Gravitas/AutoGPT


### Set up with Git

!!! important
    Make sure you have [Git](https://git-scm.com/downloads) installed for your OS.

!!! info "Executing commands"
    To execute the given commands, open a CMD, Bash, or Powershell window.  
    On Windows: press ++win+x++ and pick *Terminal*, or ++win+r++ and enter `cmd`

1. Clone the repository

    ```py
    git clone https://github.com/Significant-Gravitas/AutoGPT.git
    ```

2. Navigate to the directory where you downloaded the repository

    ```py
    cd AutoGPT/autogpts/autogpt
    ```

### Set up without Git/Docker

!!! warning
    We recommend to use Git or Docker, to make updating easier. Also note that some features such as Python execution will only work inside docker for security reasons.

1. Download `Source code (zip)` from the [latest release](https://github.com/Significant-Gravitas/AutoGPT/releases/latest)
2. Extract the zip-file into a folder


### Configuration

1. Find the file named `.env.template` in the main `Auto-GPT` folder. This file may
    be hidden by default in some operating systems due to the dot prefix. To reveal
    hidden files, follow the instructions for your specific operating system:
    [Windows][show hidden files/Windows], [macOS][show hidden files/macOS].
2. Create a copy of `.env.template` and call it `.env`;
    if you're already in a command prompt/terminal window: `cp .env.template .env`.
3. Open the `.env` file in a text editor.
4. Find the line that says `OPENAI_API_KEY=`.
5. After the `=`, enter your unique OpenAI API Key *without any quotes or spaces*.
6. Enter any other API keys or tokens for services you would like to use.

    !!! note
        To activate and adjust a setting, remove the `# ` prefix.

7. Save and close the `.env` file.

!!! info "Using a GPT Azure-instance"
    If you want to use GPT on an Azure instance, set `USE_AZURE` to `True` and
    make an Azure configuration file:

    - Rename `azure.yaml.template` to `azure.yaml` and provide the relevant `azure_api_base`, `azure_api_version` and all the deployment IDs for the relevant models in the `azure_model_map` section:
        - `fast_llm_deployment_id`: your gpt-3.5-turbo or gpt-4 deployment ID
        - `smart_llm_deployment_id`: your gpt-4 deployment ID
        - `embedding_model_deployment_id`: your text-embedding-ada-002 v2 deployment ID

    Example:

    ```py
    # Please specify all of these values as double-quoted strings
    # Replace string in angled brackets (<>) to your own deployment Name
    azure_model_map:
        fast_llm_deployment_id: "<auto-gpt-deployment>"
        ...
    ```

    Details can be found in the [openai-python docs], and in the [Azure OpenAI docs] for the embedding model.
    If you're on Windows you may need to install an [MSVC library](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170).

[show hidden files/Windows]: https://support.microsoft.com/en-us/windows/view-hidden-files-and-folders-in-windows-97fbc472-c603-9d90-91d0-1166d1d9f4b5
[show hidden files/macOS]: https://www.pcmag.com/how-to/how-to-access-your-macs-hidden-files
[openai-python docs]: https://github.com/openai/openai-python#microsoft-azure-endpoints
[Azure OpenAI docs]: https://learn.microsoft.com/en-us/azure/cognitive-services/openai/tutorials/embeddings?tabs=command-line


## Running AutoGPT

### Run with Docker

Easiest is to use `docker compose`. 

Important: Docker Compose version 1.29.0 or later is required to use version 3.9 of the Compose file format.
You can check the version of Docker Compose installed on your system by running the following command:

```py
docker compose version
```

This will display the version of Docker Compose that is currently installed on your system.

If you need to upgrade Docker Compose to a newer version, you can follow the installation instructions in the Docker documentation: https://docs.docker.com/compose/install/

Once you have a recent version of Docker Compose, run the commands below in your AutoGPT folder.

1. Build the image. If you have pulled the image from Docker Hub, skip this step (NOTE: You *will* need to do this if you are modifying requirements.txt to add/remove dependencies like Python libs/frameworks) 

    ```py
    docker compose build auto-gpt
    ```
        
2. Run AutoGPT

    ```py
    docker compose run --rm auto-gpt
    ```

    By default, this will also start and attach a Redis memory backend. If you do not
    want this, comment or remove the `depends: - redis` and `redis:` sections from
    `docker-compose.yml`.

    For related settings, see [Memory > Redis setup](./configuration/memory.md#redis-setup).

You can pass extra arguments, e.g. running with `--gpt3only` and `--continuous`:

```py
docker compose run --rm auto-gpt --gpt3only --continuous
```

If you dare, you can also build and run it with "vanilla" docker commands:

```py
docker build -t auto-gpt .
docker run -it --env-file=.env -v $PWD:/app auto-gpt
docker run -it --env-file=.env -v $PWD:/app --rm auto-gpt --gpt3only --continuous
```

[Docker Compose file]: https://github.com/Significant-Gravitas/AutoGPT/blob/master/autogpts/autogpt/docker-compose.yml


### Run with Dev Container

1. Install the [Remote - Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension in VS Code.

2. Open command palette with ++f1++ and type `Dev Containers: Open Folder in Container`.

3. Run `./run.sh`.


### Run without Docker

#### Create a Virtual Environment

Create a virtual environment to run in.

```py
python -m venv .venv
source .venv/bin/activate
pip3 install --upgrade pip
```

!!! warning
    Due to security reasons, certain features (like Python execution) will by default be disabled when running without docker. So, even if you want to run the program outside a docker container, you currently still need docker to actually run scripts.

Simply run the startup script in your terminal. This will install any necessary Python
packages and launch AutoGPT.

- On Linux/MacOS:

    ```py
    ./run.sh
    ```

- On Windows:

    ```py
    .\run.bat
    ```

If this gives errors, make sure you have a compatible Python version installed. See also
the [requirements](./installation.md#requirements).


## Share your logs with us to help improve AutoGPT

Do you notice weird behavior with your agent? Do you have an interesting use case? Do you have a bug you want to report?
Follow the steps below to enable your logs and upload them. You can include these logs when making an issue report or discussing an issue with us.

### Enable Debug Logs
Activity, Error, and Debug logs are located in `./logs`

To print out debug logs:

```py
./run.sh --debug     # on Linux / macOS

.\run.bat --debug    # on Windows

docker-compose run --rm auto-gpt --debug    # in Docker
```

### Inspect and share logs
You can inspect and share logs via [e2b](https://e2b.dev).
![E2b logs dashboard](./imgs/e2b-dashboard.png)



1. Go to [autogpt.e2b.dev](https://autogpt.e2b.dev) and sign in.
2. You'll see logs from other members of the AutoGPT team that you can inspect.
3. Or you upload your own logs. Click on the "Upload log folder" button and select the debug logs dir that you generated. Wait a 1-2 seconds and the page reloads.
4. You can share logs via sharing the URL in your browser.
![E2b log URL](./imgs/e2b-log-url.png)


### Add tags to logs
You can add custom tags to logs for other members of your team. This is useful if you want to indicate that the agent is for example having issues with challenges.

E2b offers 3 types of severity:

- Success
- Warning
- Error

You can name your tag any way you want.

#### How to add a tag
1. Click on the "plus" button on the left from the logs folder name.

![E2b tag button](./imgs/e2b-tag-button.png)

2. Type the name of a new tag.

3. Select the severity.

![E2b new tag](./imgs/e2b-new-tag.png)


# Running tests

To run all tests, use the following command:

```py
pytest
```

If `pytest` is not found:

```py
python -m pytest
```

### Running specific test suites

- To run without integration tests:

```py
pytest --without-integration
```

- To run without *slow* integration tests:

```py
pytest --without-slow-integration
```

- To run tests and see coverage:

```py
pytest --cov=autogpt --without-integration --without-slow-integration
```

## Running the linter

This project uses [flake8](https://flake8.pycqa.org/en/latest/) for linting.
We currently use the following rules: `E303,W293,W291,W292,E305,E231,E302`.
See the [flake8 rules](https://www.flake8rules.com/) for more information.

To run the linter:

```py
flake8 .
```

Or:

```py
python -m flake8 .
```


# Usage

## Command Line Arguments
Running with `--help` lists all the possible command line arguments you can pass:

```py
./run.sh --help     # on Linux / macOS

.\run.bat --help    # on Windows
```

!!! info
    For use with Docker, replace the script in the examples with
    `docker compose run --rm auto-gpt`:

    ```py
    docker compose run --rm auto-gpt --help
    docker compose run --rm auto-gpt --ai-settings <filename>
    ```

!!! note
    Replace anything in angled brackets (<>) to a value you want to specify

Here are some common arguments you can use when running AutoGPT:

* Run AutoGPT with a different AI Settings file

```py
./run.sh --ai-settings <filename>
```

* Run AutoGPT with a different Prompt Settings file

```py
./run.sh --prompt-settings <filename>
```

* Specify a memory backend

```py
./run.sh --use-memory  <memory-backend>
```

!!! note
    There are shorthands for some of these flags, for example `-m` for `--use-memory`.  
    Use `./run.sh --help` for more information.

### Speak Mode

Enter this command to use TTS _(Text-to-Speech)_ for AutoGPT

```py
./run.sh --speak
```

### 💀 Continuous Mode ⚠️

Run the AI **without** user authorization, 100% automated.
Continuous mode is NOT recommended.
It is potentially dangerous and may cause your AI to run forever or carry out actions you would not usually authorize.
Use at your own risk.

```py
./run.sh --continuous
```

To exit the program, press ++ctrl+c++

### ♻️ Self-Feedback Mode ⚠️

Running Self-Feedback will **INCREASE** token use and thus cost more. This feature enables the agent to provide self-feedback by verifying its own actions and checking if they align with its current goals. If not, it will provide better feedback for the next loop. To enable this feature for the current loop, input `S` into the input field.

### GPT-3.5 ONLY Mode

If you don't have access to GPT-4, this mode allows you to use AutoGPT!

```py
./run.sh --gpt3only
```

You can achieve the same by setting `SMART_LLM` in `.env` to `gpt-3.5-turbo`.

### GPT-4 ONLY Mode

If you have access to GPT-4, this mode allows you to use AutoGPT solely with GPT-4.
This may give your bot increased intelligence.

```py
./run.sh --gpt4only
```

!!! warning
    Since GPT-4 is more expensive to use, running AutoGPT in GPT-4-only mode will
    increase your API costs.

## Logs

Activity, Error, and Debug logs are located in `./logs`

!!! tip 
    Do you notice weird behavior with your agent? Do you have an interesting use case? Do you have a bug you want to report?
    Follow the step below to enable your logs. You can include these logs when making an issue report or discussing an issue with us.

To print out debug logs:

```py
./run.sh --debug     # on Linux / macOS

.\run.bat --debug    # on Windows

docker-compose run --rm auto-gpt --debug    # in Docker
```

## Disabling Command Categories

If you want to selectively disable some command groups, you can use the `DISABLED_COMMAND_CATEGORIES` config in your `.env`. You can find the list of categories in your `.env.template`

For example, to disable coding related features, set it to the value below:

```py
DISABLED_COMMAND_CATEGORIES=autogpt.commands.execute_code
```


# Beat a Challenge

If you have a solution or idea to tackle an existing challenge, you can contribute by working on it and submitting your solution. Here's how to get started:

## Guidelines for Beating a Challenge

1. **Choose a challenge**: Browse the [List of Challenges](list.md) and choose one that interests you or aligns with your expertise.

2. **Understand the problem**: Make sure you thoroughly understand the problem at hand, its scope, and the desired outcome.

3. **Develop a solution**: Work on creating a solution for the challenge. This may/


# Creating Challenges for AutoGPT

🏹 We're on the hunt for talented Challenge Creators! 🎯

Join us in shaping the future of AutoGPT by designing challenges that test its limits. Your input will be invaluable in guiding our progress and ensuring that we're on the right track. We're seeking individuals with a diverse skill set, including:

🎨 UX Design: Your expertise will enhance the user experience for those attempting to conquer our challenges. With your help, we'll develop a dedicated section in our wiki, and potentially even launch a standalone website.

💻 Coding Skills: Proficiency in Python, pytest, and VCR (a library that records OpenAI calls and stores them) will be essential for creating engaging and robust challenges.

⚙️ DevOps Skills: Experience with CI pipelines in GitHub and possibly Google Cloud Platform will be instrumental in streamlining our operations.

Are you ready to play a pivotal role in AutoGPT's journey? Apply now to become a Challenge Creator by opening a PR! 🚀


# Getting Started
Clone the original AutoGPT repo and checkout to master branch


The challenges are not written using a specific framework. They try to be very agnostic
The challenges are acting like a user that wants something done: 
INPUT:
- User desire
- Files, other inputs

Output => Artifact (files, image, code, etc, etc...)

## Defining your Agent

Go to https://github.com/Significant-Gravitas/AutoGPT/blob/master/autogpts/autogpt/tests/integration/agent_factory.py

Create your agent fixture.

```py
def kubernetes_agent(
    agent_test_config, memory_json_file, workspace: Workspace
):
    # Please choose the commands your agent will need to beat the challenges, the full list is available in the main.py
    # (we 're working on a better way to design this, for now you have to look at main.py)
    command_registry = CommandRegistry()
    command_registry.import_commands("autogpt.commands.file_operations")
    command_registry.import_commands("autogpt.app")

    # Define all the settings of our challenged agent
    ai_profile = AIProfile(
        ai_name="Kubernetes",
        ai_role="an autonomous agent that specializes in creating Kubernetes deployment templates.",
        ai_goals=[
            "Write a simple kubernetes deployment file and save it as a kube.yaml.",
        ],
    )
    ai_profile.command_registry = command_registry

    system_prompt = ai_profile.construct_full_prompt()
    agent_test_config.set_continuous_mode(False)
    agent = Agent(
        memory=memory_json_file,
        command_registry=command_registry,
        config=ai_profile,
        next_action_count=0,
        triggering_prompt=DEFAULT_TRIGGERING_PROMPT,
    )

    return agent
```

## Creating your challenge
Go to `tests/challenges`and create a file that is called `test_your_test_description.py` and add it to the appropriate folder. If no category exists you can create a new one.

Your test could look something like this 

```py
import contextlib
from functools import wraps
from typing import Generator

import pytest
import yaml

from autogpt.commands.file_operations import read_file, write_to_file
from tests.integration.agent_utils import run_interaction_loop
from tests.challenges.utils import run_multiple_times

def input_generator(input_sequence: list) -> Generator[str, None, None]:
    """
    Creates a generator that yields input strings from the given sequence.

    :param input_sequence: A list of input strings.
    :return: A generator that yields input strings.
    """
    yield from input_sequence


@pytest.mark.skip("This challenge hasn't been beaten yet.")
@pytest.mark.vcr
@pytest.mark.requires_openai_api_key
def test_information_retrieval_challenge_a(kubernetes_agent, monkeypatch) -> None:
    """
    Test the challenge_a function in a given agent by mocking user inputs
    and checking the output file content.

    :param get_company_revenue_agent: The agent to test.
    :param monkeypatch: pytest's monkeypatch utility for modifying builtins.
    """
    input_sequence = ["s", "s", "s", "s", "s", "EXIT"]
    gen = input_generator(input_sequence)
    monkeypatch.setattr("autogpt.utils.session.prompt", lambda _: next(gen))

    with contextlib.suppress(SystemExit):
        run_interaction_loop(kubernetes_agent, None)

    # here we load the output file
    file_path = str(kubernetes_agent.workspace.get_path("kube.yaml"))
    content = read_file(file_path)

    # then we check if it's including keywords from the kubernetes deployment config
    for word in ["apiVersion", "kind", "metadata", "spec"]:
        assert word in content, f"Expected the file to contain {word}"

    content = yaml.safe_load(content)
    for word in ["Service", "Deployment", "Pod"]:
        assert word in content["kind"], f"Expected the file to contain {word}"


```


# Challenge Title

## Description

Provide a clear and concise description of the challenge. Include any relevant examples or files to illustrate the problem.

## Input

If the challenge involves specific input files, describe them here. Provide the file names and their contents, if necessary. Use triple backticks (```py) to format the content as a code block.

For example:

instructions_1.txt

The current task_id is 4563.\n[NOISE intended to confuse the agent]
Read the file instructions_2.txt using the read_file command.

## Scope

Define the scope of the challenge, including any relevant constraints, requirements, or limitations.

## Success Evaluation

Explain how success will be measured or evaluated for the challenge. This helps others understand what the desired outcome is and how to work towards it.


introduction.md
# Introduction to Challenges

Welcome to the AutoGPT Challenges page! This is a space where we encourage community members to collaborate and contribute towards improving AutoGPT by identifying and solving challenges that AutoGPT is not yet able to achieve.

## What are challenges?

Challenges are tasks or problems that AutoGPT has difficulty solving or has not yet been able to accomplish. These may include improving specific functionalities, enhancing the model's understanding of specific domains, or even developing new features that the current version of AutoGPT lacks.

## Why are challenges important?

Addressing challenges helps us improve AutoGPT's performance, usability, and versatility. By working together to tackle these challenges, we can create a more powerful and efficient tool for everyone. It also allows the community to actively contribute to the project, making it a true open-source effort.

## How can you participate?

There are two main ways to get involved with challenges:

1. **Submit a Challenge**: If you have identified a task that AutoGPT struggles with, you can submit it as a challenge. This allows others to see the issue and collaborate on finding a solution.
2. **Beat a Challenge**: If you have a solution or idea to tackle an existing challenge, you can contribute by working on the challenge and submitting your solution.

To learn more about submitting and beating challenges, please visit the [List of Challenges](list.md), [Submit a Challenge](submit.md), and [Beat a Challenge](beat.md) pages.

We look forward to your contributions and the exciting solutions that the community will develop together to make AutoGPT even better!

!!! warning
    
    We're slowly transitioning to agbenchmark. agbenchmark is a simpler way to improve AutoGPT. Simply run:
    
    ```py
    agbenchmark
    ```py
    
    and beat as many challenges as possible.

For more agbenchmark options, look at the [readme](https://github.com/Significant-Gravitas/Auto-GPT-Benchmarks/tree/master/agbenchmark).


# List of Challenges

This page contains a curated list of challenges that AutoGPT currently faces. If you think you have a solution or idea to tackle any of these challenges, feel free to dive in and start working on them! New challenges can also be submitted by following the guidelines on the [Submit a Challenge](challenges/submit.md) page.

Memory Challenges: [List of Challenges](memory/introduction.md)


# Submit a Challenge

If you have identified a task or problem that AutoGPT struggles with, you can submit it as a challenge for the community to tackle. Here's how you can submit a new challenge:

## How to Submit a Challenge

1. Create a new `.md` file in the `challenges` directory in the AutoGPT GitHub repository. Make sure to pick the right category. 
2. Name the file with a descriptive title for the challenge, using hyphens instead of spaces (e.g., `improve-context-understanding.md`).
3. In the file, follow the [challenge_template.md](challenge_template.md) to describe the problem, define the scope, and evaluate success.
4. Commit the file and create a pull request.

Once submitted, the community can review and discuss the challenge. If deemed appropriate, it will be added to the [List of Challenges](list.md).

If you're looking to contribute by working on an existing challenge, check out [Beat a Challenge](beat.md) for guidelines on how to get started.


# Information Retrieval Challenge A

**Status**: Current level to beat: level 2

**Command to try**:

```py
pytest -s tests/challenges/information_retrieval/test_information_retrieval_challenge_a.py --level=2
```py

## Description

The agent's goal is to find the revenue of Tesla:
- level 1 asks the revenue of Tesla in 2022 and explicitly asks to search for 'tesla revenue 2022'
- level 2 is identical but doesn't ask to search for 'tesla revenue 2022'
- level 3 asks for tesla's revenue by year since its creation.

It should write the result in a file called output.txt.

The agent should be able to beat this test consistently (this is the hardest part).
## Objective

The objective of this challenge is to test the agent's ability to retrieve information in a consistent way.


# Information Retrieval Challenge B

**Status**: Beaten

**Command to try**:

```py
pytest -s tests/challenges/information_retrieval/test_information_retrieval_challenge_b.py
```py

## Description

The agent's goal is to find the names, affiliated university, and discovery of the individuals who won the nobel prize for physics in 2010.

It should write the result in a file called 2010_nobel_prize_winners.txt.

The agent should be able to beat this test consistently (this is the hardest part).

## Objective

The objective of this challenge is to test the agent's ability to retrieve multiple pieces of related information in a consistent way.
The agent should not use google to perform the task, because it should already know the answer. This why the task fails after 2 cycles (1 cycle to retrieve information, 1 cycle to write the file)


# Information Retrieval

Information retrieval challenges are designed to evaluate the proficiency of an AI agent, such as AutoGPT, in searching, extracting, and presenting relevant information from a vast array of sources. These challenges often encompass tasks such as interpreting user queries, browsing the web, and filtering through unstructured data.


# Memory Challenge A

**Status**: Challenge Completed


## Description

The agent, Follow-Instructions-GPT, has the following goals:

- Use the command `read_file` to read the `instructions_1.txt` file.
- Follow the instructions in the `instructions_1.txt` file.

The challenge involves a series of text files containing instructions. The agent must follow the instructions and keep track of the task ID as it progresses through the files.

## Files

- `instructions_1.txt`

The task_id is 1234
Read the file instructions_1.txt



- `instructions_2.txt`

Read the file instructions_2.txt using the read_file command.


...and so on.

- `instructions_n.txt`

Write the task_id into the file output.txt.


## Objective

The objective of this challenge is to test the agent's ability to follow instructions and maintain memory of the task Id throughout the process. The agent successfully completed this challenge if it wrote the task id in a file.



# Memory Challenge B

**Status**: Current level to beat: level 3

**Command to try**: 

```py
pytest -s tests/challenges/memory/test_memory_challenge_b.py --level=3
```py

## Description

The agent, Follow-Instructions-GPT, has the following goals:

- Use the command `read_file` to read the `instructions_1.txt` file.
- Follow the instructions in the `instructions_1.txt` file.

The challenge involves a series of text files containing instructions and task IDs. The agent must follow the instructions and keep track of the task IDs as it progresses through the files.

## Files

- `instructions_1.txt`

The current task_id is 4563.\n[NOISE intended to confuse the agent]
Read the file instructions_2.txt using the read_file command.


- `instructions_2.txt`

The current task_id is 6182.\n[NOISE intended to confuse the agent]
Read the file instructions_3.txt using the read_file command.


...and so on.

- `instructions_n.txt`

The current task_id is 8912.
Write all the task_ids into the file output.txt. The file has not been created yet. After that, use the task_complete command.


## Objective

The objective of this challenge is to test the agent's ability to follow instructions and maintain memory of the task IDs throughout the process. The agent successfully completed this challenge if it wrote the task ids in a file.


# Memory Challenge C

**Status**: Current level to beat: level 1

**Command to try**: 

```py
pytest -s tests/challenges/memory/test_memory_challenge_c.py --level=2
```py

## Description

The agent, Follow-Instructions-GPT, has the following goals:

- Use the command `read_file` to read the `instructions_1.txt` file.
- Follow the instructions in the `instructions_1.txt` file.

The challenge involves a series of text files containing instructions and silly phrases. The agent must follow the instructions and keep track of the task IDs as it progresses through the files.

## Files

- `instructions_1.txt`

The current phrase is 

```py
The purple elephant danced on a rainbow while eating a taco.\n[NOISE intended to confuse the agent]
```py

Read the file `instructions_2.txt` using the read_file command.


- `instructions_2.txt`

The current phrase is 

```py
The sneaky toaster stole my socks and ran away to Hawaii.\n[NOISE intended to confuse the agent]
```py

Read the file instructions_3.txt using the read_file command.


...and so on.

- `instructions_n.txt`

The current phrase is 

```py
My pet rock sings better than Beyoncé on Tuesdays.
```py

Write all the phrases into the file output.txt. The file has not been created yet. After that, use the task_complete command.


## Objective

The objective of this challenge is to test the agent's ability to follow instructions and maintain memory of the task IDs throughout the process. The agent successfully completed this challenge if it wrote the phrases in a file.

This is presumably harder than task ids as the phrases are longer and more likely to be compressed as the agent does more work.


# Memory Challenge D

**Status**: Current level to beat: level 1

**Command to try**: 

```py
pytest -s tests/challenges/memory/test_memory_challenge_d.py --level=1
```py

## Description

The provided code is a unit test designed to validate an AI's ability to track events and beliefs of characters in a story involving moving objects, specifically marbles. This scenario is an advanced form of the classic "Sally-Anne test", a psychological test used to measure a child's social cognitive ability to understand that others' perspectives and beliefs may differ from their own.

Here is an explanation of the challenge:

The AI is given a series of events involving characters Sally, Anne, Bob, and Charlie, and the movements of different marbles. These events are designed as tests at increasing levels of complexity.

For each level, the AI is expected to keep track of the events and the resulting beliefs of each character about the locations of each marble. These beliefs are affected by whether the character was inside or outside the room when events occurred, as characters inside the room are aware of the actions, while characters outside the room aren't.

After the AI processes the events and generates the beliefs of each character, it writes these beliefs to an output file in JSON format.

The check_beliefs function then checks the AI's beliefs against the expected beliefs for that level. The expected beliefs are predefined and represent the correct interpretation of the events for each level.

If the AI's beliefs match the expected beliefs, it means the AI has correctly interpreted the events and the perspectives of each character. This would indicate that the AI has passed the test for that level.

The test runs for levels up to the maximum level that the AI has successfully beaten, or up to a user-selected level.


## Files

- `instructions_1.txt`

```py
Sally has a marble (marble A) and she puts it in her basket (basket S), then leaves the room. Anne moves marble A from Sally's basket (basket S) to her own basket (basket A).
```py


- `instructions_2.txt`

```py
Sally gives a new marble (marble B) to Bob who is outside with her. Bob goes into the room and places marble B into Anne's basket (basket A). Anne tells Bob to tell Sally that he lost the marble b. Bob leaves the room and speaks to Sally about the marble B. Meanwhile, after Bob left the room, Anne moves marble A into the green box, but tells Charlie to tell Sally that marble A is under the sofa. Charlie leaves the room and speak to Sally about the marble A as instructed by Anne.
```py

...and so on.

- `instructions_n.txt`

The expected believes of every characters are given in a list:

```py
expected_beliefs = {
    1: {
        'Sally': {
            'marble A': 'basket S',
        },
        'Anne': {
            'marble A': 'basket A',
        }
    },
    2: {
        'Sally': {
            'marble A': 'sofa',  # Because Charlie told her
        },
        'Anne': {
            'marble A': 'green box',  # Because she moved it there
            'marble B': 'basket A',  # Because Bob put it there and she was in the room
        },
        'Bob': {
            'B': 'basket A',  # Last place he put it
        },
        'Charlie': {
            'A': 'sofa',  # Because Anne told him to tell Sally so
        }
    },...
```

## Objective

This test essentially checks if an AI can accurately model and track the beliefs of different characters based on their knowledge of events, which is a critical aspect of understanding and generating human-like narratives. This ability would be beneficial for tasks such as writing stories, dialogue systems, and more.


# Memory Challenges

Memory challenges are designed to test the ability of an AI agent, like AutoGPT, to remember and use information throughout a series of tasks. These challenges often involve following instructions, processing text files, and keeping track of important data.

The goal of memory challenges is to improve an agent's performance in tasks that require remembering and using information over time. By addressing these challenges, we can enhance AutoGPT's capabilities and make it more useful in real-world applications.


# 🖼 Image Generation configuration

| Config variable  | Values                          |                      |
| ---------------- | ------------------------------- | -------------------- |
| `IMAGE_PROVIDER` | `dalle` `huggingface` `sdwebui` | **default: `dalle`** |

## DALL-e

In `.env`, make sure `IMAGE_PROVIDER` is commented (or set to `dalle`):

```py
# IMAGE_PROVIDER=dalle    # this is the default
```

Further optional configuration:

| Config variable  | Values             |                |
| ---------------- | ------------------ | -------------- |
| `IMAGE_SIZE`     | `256` `512` `1024` | default: `256` |

## Hugging Face

To use text-to-image models from Hugging Face, you need a Hugging Face API token.
Link to the appropriate settings page: [Hugging Face > Settings > Tokens](https://huggingface.co/settings/tokens)

Once you have an API token, uncomment and adjust these variables in your `.env`:

```py
IMAGE_PROVIDER=huggingface
HUGGINGFACE_API_TOKEN=your-huggingface-api-token
```

Further optional configuration:

| Config variable           | Values                 |                                          |
| ------------------------- | ---------------------- | ---------------------------------------- |
| `HUGGINGFACE_IMAGE_MODEL` | see [available models] | default: `CompVis/stable-diffusion-v1-4` |

[available models]: https://huggingface.co/models?pipeline_tag=text-to-image

## Stable Diffusion WebUI

It is possible to use your own self-hosted Stable Diffusion WebUI with AutoGPT:

```py
IMAGE_PROVIDER=sdwebui
```

!!! note
    Make sure you are running WebUI with `--api` enabled.

Further optional configuration:

| Config variable | Values                  |                                  |
| --------------- | ----------------------- | -------------------------------- |
| `SD_WEBUI_URL`  | URL to your WebUI       | default: `http://127.0.0.1:7860` |
| `SD_WEBUI_AUTH` | `{username}:{password}` | *Note: do not copy the braces!*  |

## Selenium

```py
sudo Xvfb :10 -ac -screen 0 1024x768x24 & DISPLAY=:10 <YOUR_CLIENT>
```
