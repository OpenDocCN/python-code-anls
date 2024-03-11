# `.\Langchain-Chatchat\tests\api\test_llm_api.py`

```
# 导入requests库，用于发送HTTP请求
import requests
# 导入json库，用于处理JSON数据
import json
# 导入sys库，用于访问Python解释器的变量和函数
import sys
# 从pathlib模块中导入Path类，用于处理文件路径
from pathlib import Path

# 获取当前文件的父目录的父目录的父目录
root_path = Path(__file__).parent.parent.parent
# 将根路径添加到系统路径中
sys.path.append(str(root_path))
# 从configs.server_config模块中导入FSCHAT_MODEL_WORKERS变量
from configs.server_config import FSCHAT_MODEL_WORKERS
# 从server.utils模块中导入api_address和get_model_worker_config函数

# 从pprint模块中导入pprint函数，用于美化打印输出
from pprint import pprint
# 导入random模块，用于生成随机数
import random
# 从typing模块中导入List类
from typing import List

# 定义一个函数，返回配置的模型列表
def get_configured_models() -> List[str]:
    # 将FSCHAT_MODEL_WORKERS转换为列表
    model_workers = list(FSCHAT_MODEL_WORKERS)
    # 如果"default"在模型列表中，移除"default"
    if "default" in model_workers:
        model_workers.remove("default")
    return model_workers

# 获取API的基础URL
api_base_url = api_address()

# 定义一个函数，获取当前正在运行的模型
def get_running_models(api="/llm_model/list_models"):
    # 拼接完整的API地址
    url = api_base_url + api
    # 发送POST请求
    r = requests.post(url)
    # 如果响应状态码为200，返回数据部分
    if r.status_code == 200:
        return r.json()["data"]
    return []

# 定义一个函数，测试当前正在运行的模型
def test_running_models(api="/llm_model/list_running_models"):
    # 拼接完整的API地址
    url = api_base_url + api
    # 发送POST请求
    r = requests.post(url)
    # 断言响应状态码为200
    assert r.status_code == 200
    # 打印提示信息
    print("\n获取当前正在运行的模型列表：")
    # 美化打印JSON数据
    pprint(r.json())
    # 断言返回数据为列表
    assert isinstance(r.json()["data"], list)
    # 断言返回数据长度大于0
    assert len(r.json()["data"]) > 0

# 定义一个函数，测试切换模型
def test_change_model(api="/llm_model/change_model"):
    # 拼接完整的API地址
    url = api_base_url + api

    # 获取当前正在运行的模型列表
    running_models = get_running_models()
    # 断言正在运行的模型数量大于0
    assert len(running_models) > 0

    # 获取配置的模型列表
    model_workers = get_configured_models()

    # 计算可用的新模型列表
    availabel_new_models = list(set(model_workers) - set(running_models))
    # 断言可用的新模型数量大于0
    assert len(availabel_new_models) > 0
    # 打印可用的新模型列表
    print(availabel_new_models)

    # 获取本地模型列表
    local_models = [x for x in running_models if not get_model_worker_config(x).get("online_api")]
    # 从本地模型中随机选择一个模型
    model_name = random.choice(local_models)
    # 从可用的新模型中随机选择一个新模型
    new_model_name = random.choice(availabel_new_models)
    # 打印切换模型的信息
    print(f"\n尝试将模型从 {model_name} 切换到 {new_model_name}")
    # 发送POST请求，切换模型
    r = requests.post(url, json={"model_name": model_name, "new_model_name": new_model_name})
    # 断言响应状态码为200
    assert r.status_code == 200

    # 更新当前正在运行的模型列表
    running_models = get_running_models()
    # 断言新模型名称是否在运行中的模型列表中，如果不在则抛出异常
    assert new_model_name in running_models
```