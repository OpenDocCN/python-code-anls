# `d:/src/tocomm/basic-computer-games\63_Name\python\name.py`

```
"""
NAME

simple string manipulations on the user's name

Ported by Dave LeCompte
"""

# 定义函数，接受一个字符串参数，返回布尔值
def is_yes_ish(answer: str) -> bool:
    # 去除字符串两端的空格并转换为大写
    cleaned = answer.strip().upper()
    # 如果清理后的字符串在列表["Y", "YES"]中，则返回True，否则返回False
    if cleaned in ["Y", "YES"]:
        return True
    return False

# 定义主函数，不返回任何值
def main() -> None:
    # 打印字符串
    print(" " * 34 + "NAME")
    # 打印字符串
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")
    # 打印字符串
    print("HELLO.")
    print("MY NAME iS CREATIVE COMPUTER.")  # 打印字符串
    name = input("WHAT'S YOUR NAME (FIRST AND LAST)?")  # 获取用户输入的名字
    print()
    name_as_list = list(name)  # 将名字转换为列表
    reversed_name = "".join(name_as_list[::-1])  # 将名字倒序排列
    print(f"THANK YOU, {reversed_name}.\n")  # 打印倒序排列后的名字
    print("OOPS!  I GUESS I GOT IT BACKWARDS.  A SMART")
    print("COMPUTER LIKE ME SHOULDN'T MAKE A MISTAKE LIKE THAT!\n\n")
    print("BUT I JUST NOTICED YOUR LETTERS ARE OUT OF ORDER.")

    sorted_name = "".join(sorted(name_as_list))  # 将名字按字母顺序排列
    print(f"LET'S PUT THEM IN ORDER LIKE THIS: {sorted_name}\n\n")  # 打印按字母顺序排列后的名字

    print("DON'T YOU LIKE THAT BETTER?")
    like_answer = input()  # 获取用户输入
    print()
    if is_yes_ish(like_answer):  # 判断用户是否同意
        print("I KNEW YOU'D AGREE!!")  # 如果用户同意，打印相应信息
    else:
        print("I'M SORRY YOU DON'T LIKE IT THAT WAY.")  # 如果用户不同意，打印相应信息
    print()  # 打印空行
    print(f"I REALLY ENJOYED MEETING YOU, {name}.")  # 打印包含变量 name 的欢迎信息
    print("HAVE A NICE DAY!")  # 打印祝愿信息


if __name__ == "__main__":
    main()  # 调用主函数
```