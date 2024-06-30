# `D:\src\scipysrc\sympy\sympy\external\tests\__init__.py`

```
# 定义一个名为 `parse_data` 的函数，接受一个字符串参数 `data`
def parse_data(data):
    # 利用正则表达式查找数据中的数字，并以列表形式返回
    return [int(x) for x in re.findall(r'\d+', data)]

# 定义一个名为 `process_data` 的函数，接受一个字符串参数 `data`
def process_data(data):
    # 调用 `parse_data` 函数解析输入数据，得到一个整数列表
    numbers = parse_data(data)
    # 计算整数列表中的总和并返回结果
    return sum(numbers)

# 定义一个名为 `main` 的函数
def main():
    # 提供一个测试用例字符串
    data = "Python is 25 years old"
    # 调用 `process_data` 函数处理测试用例字符串，并打印结果
    print(process_data(data))

# 如果该脚本作为主程序运行，则执行 `main` 函数
if __name__ == "__main__":
    main()
```