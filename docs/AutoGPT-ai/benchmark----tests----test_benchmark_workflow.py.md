# `.\AutoGPT\benchmark\tests\test_benchmark_workflow.py`

```py
# 导入 pytest 和 requests 模块
import pytest
import requests

# 定义两个 URL 常量
URL_BENCHMARK = "http://localhost:8080/ap/v1"
URL_AGENT = "http://localhost:8000/ap/v1"

# 导入 datetime 和 time 模块
import datetime
import time

# 使用 pytest.mark.parametrize 装饰器定义参数化测试用例
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
# 定义测试函数 test_entire_workflow，接受多个参数
def test_entire_workflow(
    eval_id, input_text, expected_artifact_length, test_name, should_be_successful
):
    # 构建任务请求字典
    task_request = {"eval_id": eval_id, "input": input_text}
    # 发送 GET 请求获取任务数量
    response = requests.get(f"{URL_AGENT}/agent/tasks")
    task_count_before = response.json()["pagination"]["total_items"]
    # 发送第一个 POST 请求创建任务
    task_response_benchmark = requests.post(
        URL_BENCHMARK + "/agent/tasks", json=task_request
    )
    # 再次发送 GET 请求获取任务数量
    response = requests.get(f"{URL_AGENT}/agent/tasks")
    task_count_after = response.json()["pagination"]["total_items"]
    # 断言任务数量增加了一个
    assert task_count_after == task_count_before + 1

    # 获取当前时间戳
    timestamp_after_task_eval_created = datetime.datetime.now(datetime.timezone.utc)
    # 等待 1.1 秒，确保两个时间戳不同
    time.sleep(1.1)
    # 断言任务响应状态码为 200
    assert task_response_benchmark.status_code == 200
    # 将任务响应转换为 JSON 格式
    task_response_benchmark = task_response_benchmark.json()
    # 断言任务输入与预期输入一致
    assert task_response_benchmark["input"] == input_text

    # 获取任务 ID
    task_response_benchmark_id = task_response_benchmark["task_id"]

    # 发送 GET 请求获取任务信息
    response_task_agent = requests.get(
        f"{URL_AGENT}/agent/tasks/{task_response_benchmark_id}"
    )
    # 断言任务响应状态码为 200
    assert response_task_agent.status_code == 200
    # 将任务响应转换为 JSON 格式
    response_task_agent = response_task_agent.json()
    # 断言任务代理的工件数量是否等于预期的工件数量
    assert len(response_task_agent["artifacts"]) == expected_artifact_length

    # 创建步骤请求，包含输入文本
    step_request = {"input": input_text}

    # 发送 POST 请求，创建步骤并获取响应
    step_response = requests.post(
        URL_BENCHMARK + "/agent/tasks/" + task_response_benchmark_id + "/steps",
        json=step_request,
    )
    # 断言步骤响应状态码为 200
    assert step_response.status_code == 200
    # 将步骤响应转换为 JSON 格式
    step_response = step_response.json()
    # 断言 is_last 属性为 True，假设 is_last 总是为 True
    assert step_response["is_last"] == True  

    # 发送 POST 请求，创建评估并获取响应
    eval_response = requests.post(
        URL_BENCHMARK + "/agent/tasks/" + task_response_benchmark_id + "/evaluations",
        json={},
    )
    # 断言评估响应状态码为 200
    assert eval_response.status_code == 200
    # 将评估响应转换为 JSON 格式
    eval_response = eval_response.json()
    # 打印评估响应
    print("eval_response")
    print(eval_response)
    # 断言评估运行详情中的测试名称与 test_name 相符
    assert eval_response["run_details"]["test_name"] == test_name
    # 断言评估指标中的成功率与应为成功的值相符
    assert eval_response["metrics"]["success"] == should_be_successful
    # 将评估运行开始时间转换为日期时间格式
    benchmark_start_time = datetime.datetime.fromisoformat(
        eval_response["run_details"]["benchmark_start_time"]
    )

    # 断言基准开始时间早于任务评估创建后的时间戳
    assert benchmark_start_time < timestamp_after_task_eval_created
```