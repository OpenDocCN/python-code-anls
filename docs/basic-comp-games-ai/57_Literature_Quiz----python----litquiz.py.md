# `basic-computer-games\57_Literature_Quiz\python\litquiz.py`

```py
"""
LITQUIZ

A children's literature quiz

Ported by Dave LeCompte
"""

# 导入必要的类型
from typing import List, NamedTuple

# 设置页面宽度
PAGE_WIDTH = 64


# 定义问题类
class Question(NamedTuple):
    question: str
    answer_list: List[str]
    correct_number: int
    incorrect_message: str
    correct_message: str

    # 定义问题类的方法，用于提问并返回是否回答正确
    def ask(self) -> bool:
        print(self.question)

        # 打印选项
        options = [f"{i+1}){self.answer_list[i]}" for i in range(len(self.answer_list))]
        print(", ".join(options))

        # 获取用户输入
        response = int(input())

        # 判断用户输入是否正确
        if response == self.correct_number:
            print(self.correct_message)
            return True
        else:
            print(self.incorrect_message)
            return False


# 定义问题列表
questions = [
    Question(
        "IN PINOCCHIO, WHAT WAS THE NAME OF THE CAT?",
        ["TIGGER", "CICERO", "FIGARO", "GUIPETTO"],
        3,
        "SORRY...FIGARO WAS HIS NAME.",
        "VERY GOOD!  HERE'S ANOTHER.",
    ),
    Question(
        "FROM WHOSE GARDEN DID BUGS BUNNY STEAL THE CARROTS?",
        ["MR. NIXON'S", "ELMER FUDD'S", "CLEM JUDD'S", "STROMBOLI'S"],
        2,
        "TOO BAD...IT WAS ELMER FUDD'S GARDEN.",
        "PRETTY GOOD!",
    ),
    Question(
        "IN THE WIZARD OF OS, DOROTHY'S DOG WAS NAMED?",
        ["CICERO", "TRIXIA", "KING", "TOTO"],
        4,
        "BACK TO THE BOOKS,...TOTO WAS HIS NAME.",
        "YEA!  YOU'RE A REAL LITERATURE GIANT.",
    ),
    Question(
        "WHO WAS THE FAIR MAIDEN WHO ATE THE POISON APPLE?",
        ["SLEEPING BEAUTY", "CINDERELLA", "SNOW WHITE", "WENDY"],
        3,
        "OH, COME ON NOW...IT WAS SNOW WHITE.",
        "GOOD MEMORY!",
    ),
]


# 定义居中打印函数
def print_centered(msg: str) -> None:
    spaces = " " * ((64 - len(msg)) // 2)
    print(spaces + msg)


# 打印说明
def print_instructions() -> None:
    print("TEST YOUR KNOWLEDGE OF CHILDREN'S LITERATURE.")
    print()
    print("THIS IS A MULTIPLE-CHOICE QUIZ.")
    print("TYPE A 1, 2, 3, OR 4 AFTER THE QUESTION MARK.")
    print()
    print("GOOD LUCK!")
    print()
    print()
# 定义主函数，不返回任何结果
def main() -> None:
    # 居中打印标题
    print_centered("LITERATURE QUIZ")
    # 居中打印副标题
    print_centered("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
    # 打印空行
    print()
    print()
    print()

    # 打印游戏说明
    print_instructions()

    # 初始化得分
    score = 0

    # 遍历问题列表
    for q in questions:
        # 如果问题被回答正确，得分加一
        if q.ask():
            score += 1
        # 打印空行
        print()
        print()

    # 根据得分情况打印不同的结果
    if score == len(questions):
        print("WOW!  THAT'S SUPER!  YOU REALLY KNOW YOUR NURSERY")
        print("YOUR NEXT QUIZ WILL BE ON 2ND CENTURY CHINESE")
        print("LITERATURE (HA, HA, HA)")
    elif score < len(questions) / 2:
        print("UGH.  THAT WAS DEFINITELY NOT TOO SWIFT.  BACK TO")
        print("NURSERY SCHOOL FOR YOU, MY FRIEND.")
    else:
        print("NOT BAD, BUT YOU MIGHT SPEND A LITTLE MORE TIME")
        print("READING THE NURSERY GREATS.")


# 如果当前脚本被直接执行，则调用主函数
if __name__ == "__main__":
    main()
```