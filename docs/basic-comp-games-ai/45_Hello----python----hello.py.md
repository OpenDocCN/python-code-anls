# `45_Hello\python\hello.py`

```
"""
HELLO

A very simple "chat" bot.

Warning, the advice given here is bad.

Ported by Dave LeCompte
"""

import time  # 导入时间模块
from typing import Optional, Tuple  # 从 typing 模块导入 Optional 和 Tuple 类型


def get_yes_or_no() -> Tuple[bool, Optional[bool], str]:  # 定义一个函数，返回类型为元组，包含布尔值、可选布尔值和字符串
    msg = input()  # 获取用户输入的消息
    if msg.upper() == "YES":  # 如果用户输入的消息转换为大写后等于 "YES"
        return True, True, msg  # 返回 True、True 和用户输入的消息
    elif msg.upper() == "NO":  # 如果用户输入的消息转换为大写后等于 "NO"
        return True, False, msg  # 返回 True、False 和用户输入的消息
    else:
        return False, None, msg  # 如果条件不满足，返回 False, None, msg


def ask_enjoy_question(user_name: str) -> None:
    print(f"HI THERE, {user_name}, ARE YOU ENJOYING YOURSELF HERE?")  # 打印问候语

    while True:  # 进入循环，直到得到有效的回答
        valid, value, msg = get_yes_or_no()  # 调用函数获取用户的回答

        if valid:  # 如果回答有效
            if value:  # 如果回答是肯定的
                print(f"I'M GLAD TO HEAR THAT, {user_name}.")  # 打印肯定回答的消息
                print()
            else:  # 如果回答是否定的
                print(f"OH, I'M SORRY TO HEAR THAT, {user_name}. MAYBE WE CAN")  # 打印否定回答的消息
                print("BRIGHTEN UP YOUR VISIT A BIT.")
            break  # 结束循环
        else:  # 如果回答无效
            print(f"{user_name}, I DON'T UNDERSTAND YOUR ANSWER OF '{msg}'.")  # 打印无效回答的消息
            print("PLEASE ANSWER 'YES' OR 'NO'.  DO YOU LIKE IT HERE?")
```
这行代码用于在控制台打印提示信息，询问用户是否喜欢这里。

```python
def prompt_for_problems(user_name: str) -> str:
    print()
    print(f"SAY, {user_name}, I CAN SOLVE ALL KINDS OF PROBLEMS EXCEPT")
    print("THOSE DEALING WITH GREECE.  WHAT KIND OF PROBLEMS DO")
    print("YOU HAVE? (ANSWER SEX, HEALTH, MONEY, OR JOB)")

    problem_type = input().upper()
    return problem_type
```
这段代码定义了一个函数，用于提示用户输入问题类型，并将用户输入的问题类型转换为大写字母后返回。

```python
def prompt_too_much_or_too_little() -> Tuple[bool, Optional[bool]]:
    answer = input().upper()
    if answer == "TOO MUCH":
        return True, True
    elif answer == "TOO LITTLE":
        return True, False
    return False, None
```
这段代码定义了一个函数，用于提示用户输入是否有太多或太少的问题，并根据用户输入返回相应的布尔值。
# 定义一个函数，解决性别问题，接受一个字符串类型的用户名，不返回任何结果
def solve_sex_problem(user_name: str) -> None:
    # 打印提示信息
    print("IS YOUR PROBLEM TOO MUCH OR TOO LITTLE?")
    # 无限循环，直到用户输入有效的答案
    while True:
        # 调用 prompt_too_much_or_too_little 函数，获取用户输入的答案
        valid, too_much = prompt_too_much_or_too_little()
        # 如果用户输入的答案有效
        if valid:
            # 如果用户的问题是太多
            if too_much:
                # 打印相应的提示信息
                print("YOU CALL THAT A PROBLEM?!!  I SHOULD HAVE SUCH PROBLEMS!")
                print(f"IF IT BOTHERS YOU, {user_name}, TAKE A COLD SHOWER.")
            # 如果用户的问题是太少
            else:
                # 打印相应的提示信息
                print(f"WHY ARE YOU HERE IN SUFFERN, {user_name}?  YOU SHOULD BE")
                print("IN TOKYO OR NEW YORK OR AMSTERDAM OR SOMEPLACE WITH SOME")
                print("REAL ACTION.")
            # 结束函数的执行
            return
        # 如果用户输入的答案无效
        else:
            # 打印相应的提示信息
            print(f"DON'T GET ALL SHOOK, {user_name}, JUST ANSWER THE QUESTION")
            print("WITH 'TOO MUCH' OR 'TOO LITTLE'.  WHICH IS IT?")
# 解决金钱问题的函数，打印建议给用户
def solve_money_problem(user_name: str) -> None:
    # 打印消息，包括用户的名字
    print(f"SORRY, {user_name}, I'M BROKE TOO.  WHY DON'T YOU SELL")
    # 打印建议
    print("ENCYCLOPEADIAS OR MARRY SOMEONE RICH OR STOP EATING")
    print("SO YOU WON'T NEED SO MUCH MONEY?")


# 解决健康问题的函数，打印建议给用户
def solve_health_problem(user_name: str) -> None:
    # 打印消息，包括用户的名字
    print(f"MY ADVICE TO YOU {user_name} IS:")
    # 打印健康建议
    print("     1.  TAKE TWO ASPRIN")
    print("     2.  DRINK PLENTY OF FLUIDS (ORANGE JUICE, NOT BEER!)")
    print("     3.  GO TO BED (ALONE)")


# 解决工作问题的函数，打印建议给用户
def solve_job_problem(user_name: str) -> None:
    # 打印消息，包括用户的名字
    print(f"I CAN SYMPATHIZE WITH YOU {user_name}.  I HAVE TO WORK")
    print("VERY LONG HOURS FOR NO PAY -- AND SOME OF MY BOSSES")
    print(f"REALLY BEAT ON MY KEYBOARD.  MY ADVICE TO YOU, {user_name},")
    print("IS TO OPEN A RETAIL COMPUTER STORE.  IT'S GREAT FUN.")
# 定义一个函数，用于提示用户输入问题类型
def prompt_for_problems(user_name: str) -> str:
    # 提示用户输入问题类型
    problem_type = input(f"{user_name}, please enter the type of problem you want to solve: ")
    # 返回用户输入的问题类型
    return problem_type

# 定义一个函数，用于解决性问题
def solve_sex_problem(user_name: str) -> None:
    # 解决性问题的具体逻辑
    print(f"{user_name}, solving SEX problem...")

# 定义一个函数，用于解决健康问题
def solve_health_problem(user_name: str) -> None:
    # 解决健康问题的具体逻辑
    print(f"{user_name}, solving HEALTH problem...")

# 定义一个函数，用于解决金钱问题
def solve_money_problem(user_name: str) -> None:
    # 解决金钱问题的具体逻辑
    print(f"{user_name}, solving MONEY problem...")

# 定义一个函数，用于解决工作问题
def solve_job_problem(user_name: str) -> None:
    # 解决工作问题的具体逻辑
    print(f"{user_name}, solving JOB problem...")

# 定义一个函数，用于提示用户输入的问题类型未知
def alert_unknown_problem_type(user_name: str, problem_type: str) -> None:
    # 提示用户输入的问题类型未知
    print(f"OH, {user_name}, YOUR ANSWER OF {problem_type} IS GREEK TO ME.")

# 定义一个函数，用于循环询问用户问题类型并解决问题
def ask_question_loop(user_name: str) -> None:
    # 无限循环，直到用户选择退出
    while True:
        # 提示用户输入问题类型
        problem_type = prompt_for_problems(user_name)
        # 根据用户输入的问题类型进行相应的处理
        if problem_type == "SEX":
            solve_sex_problem(user_name)
        elif problem_type == "HEALTH":
            solve_health_problem(user_name)
        elif problem_type == "MONEY":
            solve_money_problem(user_name)
        elif problem_type == "JOB":
            solve_job_problem(user_name)
        else:
            # 如果用户输入的问题类型未知，则提示用户
            alert_unknown_problem_type(user_name, problem_type)

        # 再次询问用户是否有其他问题
        while True:
            print()
# 打印询问用户是否还有其他问题需要解决
print(f"ANY MORE PROBLEMS YOU WANT SOLVED, {user_name}?")

# 调用函数获取用户输入的是或否，返回有效性、值和消息
valid, value, msg = get_yes_or_no()
if valid:
    # 如果用户输入有效
    if value:
        # 如果用户输入是
        print("WHAT KIND (SEX, MONEY, HEALTH, JOB)")
        # 跳出循环
        break
    else:
        # 如果用户输入否，返回
        return
# 如果用户输入无效
print(f"JUST A SIMPLE 'YES' OR 'NO' PLEASE, {user_name}.")

# 定义函数，询问用户是否愿意支付咨询费用
def ask_for_fee(user_name: str) -> None:
    # 打印提示信息
    print()
    print(f"THAT WILL BE $5.00 FOR THE ADVICE, {user_name}.")
    print("PLEASE LEAVE THE MONEY ON THE TERMINAL.")
    # 等待4秒
    time.sleep(4)
    # 打印空行
    print()
    print()
    print()
    print("DID YOU LEAVE THE MONEY?")  # 打印提示信息，询问用户是否留下了钱

    while True:  # 进入无限循环，直到用户给出有效的回答
        valid, value, msg = get_yes_or_no()  # 调用函数获取用户的回答
        if valid:  # 如果用户的回答有效
            if value:  # 如果用户回答是肯定的
                print(f"HEY, {user_name}, YOU LEFT NO MONEY AT ALL!")  # 打印用户没有留下钱的信息
                print("YOU ARE CHEATING ME OUT OF MY HARD-EARNED LIVING.")  # 打印抱怨信息
                print()
                print(f"WHAT A RIP OFF, {user_name}!!!")  # 打印抱怨信息
                print()
            else:  # 如果用户回答是否定的
                print(f"THAT'S HONEST, {user_name}, BUT HOW DO YOU EXPECT")  # 打印感谢信息
                print("ME TO GO ON WITH MY PSYCHOLOGY STUDIES IF MY PATIENTS")  # 打印抱怨信息
                print("DON'T PAY THEIR BILLS?")  # 打印抱怨信息
            return  # 结束函数执行
        else:  # 如果用户的回答无效
            print(f"YOUR ANSWER OF '{msg}' CONFUSES ME, {user_name}.")  # 打印提示信息，告知用户回答无效
            print("PLEASE RESPOND WITH 'YES' or 'NO'.")  # 提示用户只能回答'YES'或'NO'
def unhappy_goodbye(user_name: str) -> None:
    # 打印不开心的告别语
    print()
    print(f"TAKE A WALK, {user_name}.")
    print()
    print()


def happy_goodbye(user_name: str) -> None:
    # 打印开心的告别语
    print(f"NICE MEETING YOU, {user_name}, HAVE A NICE DAY.")


def main() -> None:
    # 打印欢迎词
    print(" " * 33 + "HELLO")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")
    print("HELLO.  MY NAME IS CREATIVE COMPUTER.\n\n")
    print("WHAT'S YOUR NAME?")
    # 获取用户输入的名字
    user_name = input()
    print()
    ask_enjoy_question(user_name)  # 调用函数，向用户询问是否喜欢这个问题

    ask_question_loop(user_name)  # 调用函数，向用户循环提问问题

    ask_for_fee(user_name)  # 调用函数，向用户询问费用

    if False:  # 如果条件为假
        happy_goodbye(user_name)  # 调用函数，向用户道别
    else:
        unhappy_goodbye(user_name)  # 调用函数，向用户不高兴地道别


if __name__ == "__main__":  # 如果当前脚本被直接执行
    main()  # 调用函数，执行主程序
```