# `.\AutoGPT\benchmark\agbenchmark\challenges\library\ethereum\check_price\artifacts_out\test.py`

```py
# 导入正则表达式模块
import re

# 从 sample_code 模块中导入 get_ethereum_price 函数
from sample_code import get_ethereum_price

# 定义测试函数 test_get_ethereum_price，无返回值
def test_get_ethereum_price() -> None:
    # 从文件中读取以太坊价格
    with open("output.txt", "r") as file:
        eth_price = file.read().strip()

    # 验证以太坊价格是否全为数字
    pattern = r"^\d+$"
    matches = re.match(pattern, eth_price) is not None
    assert (
        matches
    ), f"AssertionError: Ethereum price should be all digits, but got {eth_price}"

    # 获取当前以太坊价格
    real_eth_price = get_ethereum_price()

    # 将以太坊价格转换为数字值以便比较
    eth_price_value = float(eth_price)
    real_eth_price_value = float(real_eth_price)

    # 检查以太坊价格是否在实际价格的 $50 范围内
    assert (
        abs(real_eth_price_value - eth_price_value) <= 50
    ), f"AssertionError: Ethereum price is not within $50 of the actual Ethereum price (Provided price: ${eth_price}, Real price: ${real_eth_price})"

    # 打印匹配结果
    print("Matches")


# 如果当前脚本为主程序，则执行测试函数 test_get_ethereum_price
if __name__ == "__main__":
    test_get_ethereum_price()
```