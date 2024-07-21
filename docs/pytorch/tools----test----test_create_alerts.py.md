# `.\pytorch\tools\test\test_create_alerts.py`

```
# 从未来导入类型注释的支持，允许在类声明之前使用类型注释
from __future__ import annotations

# 导入需要的类型
from typing import Any
# 导入单元测试相关的主要功能
from unittest import main, TestCase

# 从工具包中导入需要的函数和类
from tools.alerts.create_alerts import filter_job_names, JobStatus

# 定义一个作业名称常量
JOB_NAME = "periodic / linux-xenial-cuda10.2-py3-gcc7-slow-gradcheck / test (default, 2, 2, linux.4xlarge.nvidia.gpu)"

# 模拟的测试数据
MOCK_TEST_DATA = [
    {
        "sha": "f02f3046571d21b48af3067e308a1e0f29b43af9",
        "id": 7819529276,
        "conclusion": "failure",
        "htmlUrl": "https://github.com/pytorch/pytorch/runs/7819529276?check_suite_focus=true",
        "logUrl": "https://ossci-raw-job-status.s3.amazonaws.com/log/7819529276",
        "durationS": 14876,
        "failureLine": "##[error]The action has timed out.",
        "failureContext": "",
        "failureCaptures": ["##[error]The action has timed out."],
        "failureLineNumber": 83818,
        "repo": "pytorch/pytorch",
    },
    {
        "sha": "d0d6b1f2222bf90f478796d84a525869898f55b6",
        "id": 7818399623,
        "conclusion": "failure",
        "htmlUrl": "https://github.com/pytorch/pytorch/runs/7818399623?check_suite_focus=true",
        "logUrl": "https://ossci-raw-job-status.s3.amazonaws.com/log/7818399623",
        "durationS": 14882,
        "failureLine": "##[error]The action has timed out.",
        "failureContext": "",
        "failureCaptures": ["##[error]The action has timed out."],
        "failureLineNumber": 72821,
        "repo": "pytorch/pytorch",
    },
]

# 测试类，继承自 TestCase 类
class TestGitHubPR(TestCase):

    # 测试用例：验证是否能够正确触发警报
    def test_alert(self) -> None:
        # 创建修改后的数据列表，初始为空字典
        modified_data: list[Any] = [{}]
        # 添加空字典作为元素
        modified_data.append({})
        # 将模拟测试数据追加到列表中
        modified_data.extend(MOCK_TEST_DATA)
        # 创建作业状态对象
        status = JobStatus(JOB_NAME, modified_data)
        # 断言应该触发警报
        self.assertTrue(status.should_alert())

    # 测试用例：验证作业名称过滤函数的行为
    def test_job_filter(self) -> None:
        # 定义作业名称列表
        job_names = [
            "pytorch_linux_xenial_py3_6_gcc5_4_test",
            "pytorch_linux_xenial_py3_6_gcc5_4_test2",
        ]
        # 断言空正则表达式匹配所有作业名称
        self.assertListEqual(
            filter_job_names(job_names, ""),
            job_names,
            "empty regex should match all jobs",
        )
        # 断言".*"正则表达式匹配所有作业名称
        self.assertListEqual(filter_job_names(job_names, ".*"), job_names)
        # 断言".*xenial.*"正则表达式匹配包含"xenial"的所有作业名称
        self.assertListEqual(filter_job_names(job_names, ".*xenial.*"), job_names)
        # 断言".*xenial.*test2"正则表达式匹配特定作业名称
        self.assertListEqual(
            filter_job_names(job_names, ".*xenial.*test2"),
            ["pytorch_linux_xenial_py3_6_gcc5_4_test2"],
        )
        # 断言".*xenial.*test3"正则表达式不匹配任何作业名称
        self.assertListEqual(filter_job_names(job_names, ".*xenial.*test3"), [])
        # 断言异常应该被抛出，因为正则表达式格式错误
        self.assertRaises(
            Exception,
            lambda: filter_job_names(job_names, "["),
            msg="malformed regex should throw exception",
        )

# 如果当前脚本作为主程序运行，则执行单元测试
if __name__ == "__main__":
    main()
```