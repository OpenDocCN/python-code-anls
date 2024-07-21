# `.\pytorch\torch\csrc\jit\tensorexpr\scripts\bisect.py`

```
# 忽略类型检查错误，这里是为了避免 mypy 对代码进行类型检查时报错
# import subprocess 模块用于执行外部命令
import subprocess
# import click 模块用于处理命令行参数
import click


# 定义函数 test，用于测试给定命令 cmd 在指定限制 limit 下的执行情况
def test(cmd, limit):
    # 打印测试信息，显示当前测试的命令和限制值
    print(f"Testing PYTORCH_JIT_OPT_LIMIT=tensorexpr_fuser={limit} {cmd}")
    # subprocess.run 函数执行指定命令，捕获其输出
    p = subprocess.run(
        f"PYTORCH_JIT_OPT_LIMIT=tensorexpr_fuser={limit} {cmd}",  # 构建执行命令字符串
        shell=True,  # 使用 shell 执行命令
        capture_output=True,  # 捕获标准输出和标准错误
        encoding="utf-8",  # 输出解码为 UTF-8
        check=False,  # 不检查命令的返回码
    )
    # 打印命令执行的标准输出
    print(p.stdout)
    # 定义字符串 f 用于判断是否出现内部断言失败的情况
    f = "INTERNAL ASSERT FAILED"
    # 如果标准输出或标准错误中包含 f 字符串，说明命令执行失败
    if f in p.stdout or f in p.stderr:
        # 打印跳过信息，并返回 -1 表示失败
        print("skip")
        return -1
    # 如果命令返回码为 0，说明执行成功
    if p.returncode == 0:
        # 打印成功信息，并返回 1
        print("good")
        return 1
    # 否则打印失败信息，并返回 0
    print("bad")
    return 0


# 使用 click.command() 创建命令行接口
@click.command()
# 使用 click.option() 定义命令行参数 --cmd，用于接收测试命令
@click.option("--cmd")
# 定义函数 bisect，用于执行二分查找测试
def bisect(cmd):
    # 初始化最后一个好的版本号为 0
    last_good = 0
    # 初始化第一个坏的版本号为 10000
    first_bad = 10000
    # 使用集合 skips 记录已跳过的版本号
    skips = set()

    # 定义内部函数 keep_going，用于检查是否还有未跳过的版本号需要测试
    def keep_going():
        # 遍历从 last_good+1 到 first_bad 的版本号范围
        for limit in range(last_good + 1, first_bad):
            # 如果 limit 不在 skips 集合中，返回 True
            if limit not in skips:
                return True
        # 如果所有版本号都已跳过，返回 False
        return False

    # 当还有未跳过的版本号需要测试时执行循环
    while keep_going():
        # 设置测试的限制为中间值
        test_limit = test_mid = (last_good + first_bad) // 2
        # 初始化测试结果为 -1
        val = -1

        # 从中间值向后扫描到坏的版本
        while test_limit <= first_bad and val == -1:
            # 执行测试命令，并获取返回值
            val = test(cmd, test_limit)
            # 如果测试失败，将当前限制加入 skips 集合，并继续向后测试
            if val == -1:
                skips.add(test_limit)
                test_limit = test_limit + 1

        # 如果 [中间值, 坏的版本] 区间内所有版本都跳过了，则向前扫描到好的版本
        if val == -1:
            test_limit = test_mid - 1
            while test_limit >= last_good and val == -1:
                val = test(cmd, test_limit)
                if val == -1:
                    skips.add(test_limit)
                    test_limit = test_limit - 1

        # 根据测试结果更新 first_bad 或 last_good
        if val == 0:
            first_bad = test_limit
        elif val == 1:
            last_good = test_limit

    # 打印最后一个好版本和第一个坏版本的版本号
    print(f"last good: {last_good}, first bad: {first_bad}")


# 如果当前脚本作为主程序运行，则执行 bisect 函数
if __name__ == "__main__":
    bisect()
```