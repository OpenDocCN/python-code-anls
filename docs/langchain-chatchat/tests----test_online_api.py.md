# `.\Langchain-Chatchat\tests\test_online_api.py`

```py
# 导入必要的模块
import sys
from pathlib import Path
# 获取当前文件的父目录的父目录路径
root_path = Path(__file__).parent.parent
# 将路径转换为字符串并添加到系统路径中
sys.path.append(str(root_path))

# 从configs模块中导入ONLINE_LLM_MODEL变量
from configs import ONLINE_LLM_MODEL
# 从server.model_workers.base模块中导入所有内容
from server.model_workers.base import *
# 从server.utils模块中导入get_model_worker_config和list_config_llm_models函数
from server.utils import get_model_worker_config, list_config_llm_models
# 导入pprint模块中的pprint函数
from pprint import pprint
# 导入pytest模块

# 创建一个空列表workers
workers = []
# 遍历list_config_llm_models()["online"]中的元素
for x in list_config_llm_models()["online"]:
    # 如果x在ONLINE_LLM_MODEL中且不在workers列表中，则将x添加到workers列表中
    if x in ONLINE_LLM_MODEL and x not in workers:
        workers.append(x)
# 打印所有要测试的workers列表
print(f"all workers to test: {workers}")

# 定义一个参数化测试函数test_chat，参数为worker
@pytest.mark.parametrize("worker", workers)
def test_chat(worker):
    # 创建ApiChatParams对象params，包含一个消息字典
    params = ApiChatParams(
        messages = [
            {"role": "user", "content": "你是谁"},
        ],
    )
    # 打印当前测试的worker
    print(f"\nchat with {worker} \n")

    # 如果worker_class存在，则执行以下代码块
    if worker_class := get_model_worker_config(worker).get("worker_class"):
        # 遍历worker_class().do_chat(params)的结果并打印
        for x in worker_class().do_chat(params):
            pprint(x)
            # 断言x是字典类型
            assert isinstance(x, dict)
            # 断言x中的error_code为0
            assert x["error_code"] == 0

# 定义一个参数化测试函数test_embeddings，参数为worker
@pytest.mark.parametrize("worker", workers)
def test_embeddings(worker):
    # 创建ApiEmbeddingsParams对象params，包含两个文本
    params = ApiEmbeddingsParams(
        texts = [
            "LangChain-Chatchat (原 Langchain-ChatGLM): 基于 Langchain 与 ChatGLM 等大语言模型的本地知识库问答应用实现。",
            "一种利用 langchain 思想实现的基于本地知识库的问答应用，目标期望建立一套对中文场景与开源模型支持友好、可离线运行的知识库问答解决方案。",
        ]
    )

    # 如果worker_class存在，则执行以下代码块
    if worker_class := get_model_worker_config(worker).get("worker_class"):
        # 如果worker_class支持embedding，则执行以下代码块
        if worker_class.can_embedding():
            # 打印当前测试的worker
            print(f"\embeddings with {worker} \n")
            # 调用worker_class().do_embeddings(params)方法并获取返回值
            resp = worker_class().do_embeddings(params)

            # 打印resp的内容，限制打印深度为2
            pprint(resp, depth=2)
            # 断言resp中的code为200
            assert resp["code"] == 200
            # 断言resp中包含"data"字段
            assert "data" in resp
            # 获取embeddings列表
            embeddings = resp["data"]
            # 断言embeddings是列表且长度大于0
            assert isinstance(embeddings, list) and len(embeddings) > 0
            # 断言embeddings中的第一个元素是列表且长度大于0
            assert isinstance(embeddings[0], list) and len(embeddings[0]) > 0
            # 断言embeddings中的第一个元素的第一个元素是浮点数
            assert isinstance(embeddings[0][0], float)
            # 打印向量长度
            print("向量长度：", len(embeddings[0]))

# 定义一个参数化测试函数test_completion，参数为worker
# 该函数目前被注释掉，暂时不执行
# 创建一个包含指定提示的ApiCompletionParams对象
params = ApiCompletionParams(prompt="五十六个民族")

# 打印带有worker的完成
print(f"\completion with {worker} \n")

# 获取worker的模型工作配置中的worker_class
worker_class = get_model_worker_config(worker)["worker_class"]
# 调用worker_class的do_completion方法，传入params参数
resp = worker_class().do_completion(params)
# 打印resp的内容
pprint(resp)
```