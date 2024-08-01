# `.\DB-GPT-src\dbgpt\app\scene\chat_knowledge\refine_summary\out_parser.py`

```py
        # 打印清理后的提示响应内容
        print("clean prompt response:", response)

        # 如果响应以"Triplets:"开头
        # 将"Triplets:"后的部分从响应中截取出来
        # 定义匹配模式，用于匹配括号内的内容
        # 从截取的响应中找到所有匹配模式的部分
        # 打印解析后的提示响应内容
        # 初始化结果列表
        # 遍历响应中的每一行文本
        # 跳过空行和非三元组的行
        # 去掉括号并按逗号分割文本，得到三个部分
        # 如果分割后的部分不是三个，跳过此行
        # 检查每个部分是否超过最大长度限制（按 UTF-8 编码计算字节长度）
        # 跳过长度超过限制的三元组
        # 去除每个部分的空白字符并转换为小写
        # 如果任意部分为空，则跳过此三元组
        # 将处理后的三元组添加到结果列表中
        # 返回结果列表
        return response

    ### 将工具输出的数据转换成表格视图
    def parse_view_response(self, speak, data) -> str:
        return data
```