# `d:/src/tocomm/basic-computer-games\57_Literature_Quiz\python\litquiz.py`

```
"""
LITQUIZ

A children's literature quiz

Ported by Dave LeCompte
"""

from typing import List, NamedTuple  # 导入 List 和 NamedTuple 类型

PAGE_WIDTH = 64  # 设置页面宽度为 64


class Question(NamedTuple):  # 定义 Question 类，继承自 NamedTuple
    question: str  # 问题字符串
    answer_list: List[str]  # 答案列表
    correct_number: int  # 正确答案的编号
    incorrect_message: str  # 答错时的提示信息
    correct_message: str  # 答对时的提示信息
    # 定义一个方法，用于询问用户问题并返回布尔值
    def ask(self) -> bool:
        # 打印问题
        print(self.question)

        # 创建选项列表，包括问题的所有答案
        options = [f"{i+1}){self.answer_list[i]}" for i in range(len(self.answer_list))]
        # 打印选项
        print(", ".join(options))

        # 获取用户输入的响应
        response = int(input())

        # 如果用户输入的响应与正确答案编号相同，则打印正确消息并返回True
        if response == self.correct_number:
            print(self.correct_message)
            return True
        # 否则打印错误消息并返回False
        else:
            print(self.incorrect_message)
            return False


questions = [
    Question(
        "IN PINOCCHIO, WHAT WAS THE NAME OF THE CAT?",
        ["TIGGER", "CICERO", "FIGARO", "GUIPETTO"],
抱歉，给定的代码片段似乎是Python中的一个函数，而不是需要注释的代码。请提供需要解释的实际代码片段，以便我可以为您提供帮助。
# 定义一个包含多个元组的列表，每个元组包含问题、选项、正确答案和提示
questions = [
    ("WHAT IS THE NAME OF THE PRINCESS IN THE STORY?", 
        ["SLEEPING BEAUTY", "CINDERELLA", "SNOW WHITE", "WENDY"],
        3,
        "OH, COME ON NOW...IT WAS SNOW WHITE.",
        "GOOD MEMORY!",
    ),
]

# 定义一个函数，用于将给定的字符串居中打印
def print_centered(msg: str) -> None:
    # 计算需要添加的空格数，使得字符串在64个字符的宽度中居中
    spaces = " " * ((64 - len(msg)) // 2)
    # 打印居中的字符串
    print(spaces + msg)

# 定义一个函数，用于打印游戏的说明
def print_instructions() -> None:
    print("TEST YOUR KNOWLEDGE OF CHILDREN'S LITERATURE.")
    print()
    print("THIS IS A MULTIPLE-CHOICE QUIZ.")
    print("TYPE A 1, 2, 3, OR 4 AFTER THE QUESTION MARK.")
    print()
    print("GOOD LUCK!")
    print()  # 打印空行
    print()  # 打印空行


def main() -> None:
    print_centered("LITERATURE QUIZ")  # 调用打印居中文本的函数，打印题目名称
    print_centered("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")  # 调用打印居中文本的函数，打印创意计算机的地点
    print()  # 打印空行
    print()  # 打印空行
    print()  # 打印空行

    print_instructions()  # 调用打印指令的函数

    score = 0  # 初始化得分为0

    for q in questions:  # 遍历问题列表
        if q.ask():  # 调用问题的ask方法，如果回答正确
            score += 1  # 得分加1
        print()  # 打印空行
        print()  # 打印空行
# 如果得分等于问题的数量，打印“WOW！真棒！你真的很了解你的幼儿园”
# 打印“你下一次的测验将是关于2世纪的中国文学（哈哈哈）”
elif得分小于问题数量的一半：
    # 打印“噢。那绝对不太聪明。回到幼儿园吧，我的朋友。”
否则：
    # 打印“还不错，但你可能需要多花一点时间阅读幼儿园的经典。”

如果程序作为主程序运行：
    运行主函数main()
```