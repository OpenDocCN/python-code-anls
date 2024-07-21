# `.\pytorch\test\inductor\opinfo_harness.py`

```
import os
import subprocess

# 从 torch.testing._internal.common_methods_invocations 导入操作数据库
from torch.testing._internal.common_methods_invocations import op_db

# 如果这个脚本是作为主程序运行
if __name__ == "__main__":
    # 初始化循环计数器
    i = 0
    # 循环直到遍历完操作数据库 op_db 中的所有元素
    while i < len(op_db):
        # 设置每个测试范围的起始和结束索引
        start = i
        end = i + 20
        # 设置环境变量，指定 pytest 测试的起始和结束范围
        os.environ["PYTORCH_TEST_RANGE_START"] = f"{start}"
        os.environ["PYTORCH_TEST_RANGE_END"] = f"{end}"
        # 执行 pytest 命令来运行特定的测试文件，并捕获其输出
        popen = subprocess.Popen(
            ["pytest", "test/inductor/test_torchinductor_opinfo.py"],
            stdout=subprocess.PIPE,
        )
        # 逐行读取 pytest 命令输出，并打印出来
        for line in popen.stdout:
            print(line.decode(), end="")
        # 关闭 subprocess 的标准输出流
        popen.stdout.close()
        # 等待 pytest 命令执行结束并获取其返回码
        return_code = popen.wait()
        # 如果 pytest 命令返回非零状态码，抛出 CalledProcessError 异常
        if return_code:
            raise subprocess.CalledProcessError(
                return_code, ["pytest", "test/inductor/test_torchinductor_opinfo.py"]
            )
        # 更新循环计数器，准备下一个测试范围
        i = end + 1
```