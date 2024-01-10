# `basic-computer-games\63_Name\python\name.py`

```
"""
NAME

simple string manipulations on the user's name

Ported by Dave LeCompte
"""


# 检查用户输入是否为“是”的形式
def is_yes_ish(answer: str) -> bool:
    # 去除输入字符串两端的空格并转换为大写
    cleaned = answer.strip().upper()
    # 判断清理后的字符串是否在["Y", "YES"]中
    if cleaned in ["Y", "YES"]:
        return True
    return False


# 主函数
def main() -> None:
    # 打印标题
    print(" " * 34 + "NAME")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")
    print("HELLO.")
    print("MY NAME iS CREATIVE COMPUTER.")
    # 获取用户输入的名字
    name = input("WHAT'S YOUR NAME (FIRST AND LAST)?")
    print()
    # 将名字转换为列表
    name_as_list = list(name)
    # 将名字列表反转并拼接成字符串
    reversed_name = "".join(name_as_list[::-1])
    print(f"THANK YOU, {reversed_name}.\n")
    print("OOPS!  I GUESS I GOT IT BACKWARDS.  A SMART")
    print("COMPUTER LIKE ME SHOULDN'T MAKE A MISTAKE LIKE THAT!\n\n")
    print("BUT I JUST NOTICED YOUR LETTERS ARE OUT OF ORDER.")

    # 将名字列表排序并拼接成字符串
    sorted_name = "".join(sorted(name_as_list))
    print(f"LET'S PUT THEM IN ORDER LIKE THIS: {sorted_name}\n\n")

    print("DON'T YOU LIKE THAT BETTER?")
    like_answer = input()
    print()
    # 判断用户输入是否为“是”的形式
    if is_yes_ish(like_answer):
        print("I KNEW YOU'D AGREE!!")
    else:
        print("I'M SORRY YOU DON'T LIKE IT THAT WAY.")
    print()
    print(f"I REALLY ENJOYED MEETING YOU, {name}.")
    print("HAVE A NICE DAY!")


# 如果当前脚本为主程序，则执行主函数
if __name__ == "__main__":
    main()
```