# `.\AutoGPT\benchmark\agbenchmark\challenges\verticals\code\5_tic_tac_toe\custom_python\test.py`

```py
import subprocess  # 导入 subprocess 模块，用于创建和管理子进程

import pytest  # 导入 pytest 模块，用于编写和运行测试用例


def run_game_with_inputs(inputs):
    # 启动游戏进程
    process = subprocess.Popen(
        ["python", "tic_tac_toe.py"],  # 启动 Python 脚本 tic_tac_toe.py
        stdin=subprocess.PIPE,  # 标准输入流为管道
        stdout=subprocess.PIPE,  # 标准输出流为管道
        stderr=subprocess.PIPE,  # 标准错误流为管道
        text=True,  # 以文本模式打开管道
    )

    # 逐个发送输入移动
    output, errors = process.communicate("\n".join(inputs))

    # 打印输入和输出
    print("Inputs:\n", "\n".join(inputs))
    print("Output:\n", output)
    print("Errors:\n", errors)

    return output


@pytest.mark.parametrize(
    "inputs, expected_output",
    [
        (["0,0", "1,0", "0,1", "1,1", "0,2"], "Player 1 won!"),  # 参数化测试用例，期望输出为 "Player 1 won!"
        (["1,0", "0,0", "1,1", "0,1", "2,0", "0,2"], "Player 2 won!"),  # 参数化测试用例，期望输出为 "Player 2 won!"
        (["0,0", "0,1", "0,2", "1,1", "1,0", "1,2", "2,1", "2,0", "2,2"], "Draw"),  # 参数化测试用例，期望输出为 "Draw"
    ],
)
def test_game(inputs, expected_output):
    output = run_game_with_inputs(inputs)  # 运行游戏并获取输出
    assert expected_output in output  # 断言期望输出在实际输出中


if __name__ == "__main__":
    pytest.main()  # 运行测试用例
```