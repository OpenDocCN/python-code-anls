# `.\pytorch\test\mobile\model_test\builtin_ops.py`

```py
import torch  # 导入 PyTorch 库


# https://pytorch.org/docs/stable/jit_builtin_functions.html#builtin-functions
# 定义一个继承自 torch.nn.Module 的类，用于展示 PyTorch 内置函数的示例
class TSBuiltinOpsModule(torch.nn.Module):
    def forward(self):
        x = torch.tensor(1)  # 创建一个值为 1 的 PyTorch 张量 x
        y = torch.tensor(0.5)  # 创建一个值为 0.5 的 PyTorch 张量 y
        b = float(1)  # 创建一个值为 1.0 的浮点数 b
        s = "abcde"  # 创建一个字符串 s
        l = ["1", "2", "test", "a{}b"]  # 创建一个字符串列表 l
        d = {"key": 1}  # 创建一个键为 'key'，值为 1 的字典 d
        d2 = {0: 100}  # 创建一个键为 0，值为 100 的字典 d2
        return len(
            # type 类型转换操作
            bool(x),  # 将 x 转换为布尔值
            bool(x.item()),  # 将 x 的值转换为布尔值
            int(y),  # 将 y 转换为整数
            int(y.item()),  # 将 y 的值转换为整数
            float(x),  # 将 x 转换为浮点数
            float(x.item()),  # 将 x 的值转换为浮点数
            # math 数学操作
            x & x,  # x 与 x 的按位与操作
            bool(x) & bool(x),  # x 的布尔值与 x 的布尔值的按位与操作
            int(x) & int(x),  # x 的整数值与 x 的整数值的按位与操作
            x | x,  # x 与 x 的按位或操作
            bool(x) | bool(x),  # x 的布尔值与 x 的布尔值的按位或操作
            int(x) | int(x),  # x 的整数值与 x 的整数值的按位或操作
            x << x,  # x 左移 x 位
            int(x) << int(x),  # x 的整数值左移 x 的整数值位
            x >> x,  # x 右移 x 位
            int(x) >> int(x),  # x 的整数值右移 x 的整数值位
            x ^ x,  # x 与 x 的按位异或操作
            bool(x) ^ bool(x),  # x 的布尔值与 x 的布尔值的按位异或操作
            int(x) ^ int(x),  # x 的整数值与 x 的整数值的按位异或操作
            b * float(x),  # b 乘以 x 的浮点值
            b * int(x),  # b 乘以 x 的整数值
            b + float(x),  # b 加上 x 的浮点值
            b - float(x),  # b 减去 x 的浮点值
            x.item() + y.item(),  # x 的值加上 y 的值
            x.item() - y.item(),  # x 的值减去 y 的值
            x.item() * y.item(),  # x 的值乘以 y 的值
            x.item() / y.item(),  # x 的值除以 y 的值
            float(x) < float(y),  # x 的浮点值是否小于 y 的浮点值
            float(x) <= float(y),  # x 的浮点值是否小于等于 y 的浮点值
            float(x) > float(y),  # x 的浮点值是否大于 y 的浮点值
            float(x) > int(y),  # x 的浮点值是否大于 y 的整数值
            float(x) >= float(y),  # x 的浮点值是否大于等于 y 的浮点值
            float(x) >= int(y),  # x 的浮点值是否大于等于 y 的整数值
            float(x) == float(y),  # x 的浮点值是否等于 y 的浮点值
            float(x) == int(y),  # x 的浮点值是否等于 y 的整数值
            float(x) != float(y),  # x 的浮点值是否不等于 y 的浮点值
            int(x) != float(y),  # x 的整数值是否不等于 y 的浮点值
            float(x) / float(y),  # x 的浮点值除以 y 的浮点值
            int(x) / int(y),  # x 的整数值除以 y 的整数值
            max(x),  # x 的最大值
            max(x.item(), y.item()),  # x 的值和 y 的值的最大值
            max(int(x), int(y)),  # x 的整数值和 y 的整数值的最大值
            max(float(x), float(y)),  # x 的浮点值和 y 的浮点值的最大值
            min(x),  # x 的最小值
            min(x.item(), y.item()),  # x 的值和 y 的值的最小值
            min(int(x), int(y)),  # x 的整数值和 y 的整数值的最小值
            min(float(x), float(y)),  # x 的浮点值和 y 的浮点值的最小值
            int(l[0]),  # 列表 l 中第一个元素转换为整数
            float(l[0]),  # 列表 l 中第一个元素转换为浮点数
            # string 字符串操作
            str(torch.tensor(1)),  # 创建一个值为 1 的 PyTorch 张量，并将其转换为字符串
            l[2].find("t"),  # 在字符串列表 l 的第三个元素中查找字符 "t" 的位置
            l[2].replace("t", "x"),  # 将字符串列表 l 的第三个元素中的 "t" 替换为 "x"
            l[2].lower(),  # 将字符串列表 l 的第三个元素转换为小写
            l[2].startswith("t"),  # 检查字符串列表 l 的第三个元素是否以 "t" 开头
            l[2].split("t"),  # 使用字符 "t" 分割字符串列表 l 的第三个元素
            l[2].strip(),  # 去除字符串列表 l 的第三个元素两侧的空白字符
            l[2].rstrip(),  # 去除字符串列表 l 的第三个元素右侧的空白字符
            l[2].lstrip(),  # 去除字符串列表 l 的第三个元素左侧的空白字符
            l[2][slice(2)],  # 获取字符串列表 l 的第三个元素的前两个字符
            l[3].format("x"),  # 使用字符串 "x" 格式化字符串列表 l 的第四个元素
            ord(l[2][0]),  # 获取字符串列表 l 的第三个元素的第一个字符的 ASCII 值
            len(torch.randn(3)),  # 创建一个包含 3 个元素的 PyTorch 随机张量，并返回其长度
            len(l),  # 获取字符串列表 l 的长度
            len(l[2]),  # 获取字符串列表 l 的第三个元素的长度
            len(d),  # 获取字典 d 的长度
            len(d2),  # 获取字典 d2 的长度
        )


# 定义另一个继承自 torch.nn.Module 的类，用于展示集合操作的示例
class TSCollectionOpsModule(torch.nn.Module):
    def forward(self):
        s = "abcde"  # 创建一个字符串 s
        # list 列表操作
        l = ["1", "2", "test"]  # 创建一个字符串列表 l
        l.reverse()  # 将列表 l 反转
        l.reverse()  # 再次将列表 l 反转（恢复原顺序）
        l[1] = "3"  # 将列表 l 中第二个元素改为 "3"
        l.extend(["4"])  # 将字符串 "4" 添加到列表 l 的末尾
        # str dict 字符串字典操作
        d = {"key": 1}  # 创建一个键为 '
```