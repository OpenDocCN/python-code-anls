# `basic-computer-games\45_Hello\python\hello.py`

```py
"""
HELLO

A very simple "chat" bot.

Warning, the advice given here is bad.

Ported by Dave LeCompte
"""

import time
from typing import Optional, Tuple


def get_yes_or_no() -> Tuple[bool, Optional[bool], str]:
    # 获取用户输入
    msg = input()
    # 判断用户输入是否为"YES"，如果是则返回True, True, msg
    if msg.upper() == "YES":
        return True, True, msg
    # 判断用户输入是否为"NO"，如果是则返回True, False, msg
    elif msg.upper() == "NO":
        return True, False, msg
    # 如果用户输入既不是"YES"也不是"NO"，则返回False, None, msg
    else:
        return False, None, msg


def ask_enjoy_question(user_name: str) -> None:
    # 打印问候语
    print(f"HI THERE, {user_name}, ARE YOU ENJOYING YOURSELF HERE?")

    while True:
        # 调用get_yes_or_no函数获取用户输入
        valid, value, msg = get_yes_or_no()

        if valid:
            if value:
                # 如果用户输入是"YES"，则打印积极回应
                print(f"I'M GLAD TO HEAR THAT, {user_name}.")
                print()
            else:
                # 如果用户输入是"NO"，则打印消极回应
                print(f"OH, I'M SORRY TO HEAR THAT, {user_name}. MAYBE WE CAN")
                print("BRIGHTEN UP YOUR VISIT A BIT.")
            break
        else:
            # 如果用户输入既不是"YES"也不是"NO"，则提示用户重新输入
            print(f"{user_name}, I DON'T UNDERSTAND YOUR ANSWER OF '{msg}'.")
            print("PLEASE ANSWER 'YES' OR 'NO'.  DO YOU LIKE IT HERE?")


def prompt_for_problems(user_name: str) -> str:
    # 打印提示信息
    print()
    print(f"SAY, {user_name}, I CAN SOLVE ALL KINDS OF PROBLEMS EXCEPT")
    print("THOSE DEALING WITH GREECE.  WHAT KIND OF PROBLEMS DO")
    print("YOU HAVE? (ANSWER SEX, HEALTH, MONEY, OR JOB)")

    # 获取用户输入并转换为大写
    problem_type = input().upper()
    return problem_type


def prompt_too_much_or_too_little() -> Tuple[bool, Optional[bool]]:
    # 获取用户输入并转换为大写
    answer = input().upper()
    # 判断用户输入是否为"TOO MUCH"，如果是则返回True, True
    if answer == "TOO MUCH":
        return True, True
    # 判断用户输入是否为"TOO LITTLE"，如果是则返回True, False
    elif answer == "TOO LITTLE":
        return True, False
    # 如果用户输入既不是"TOO MUCH"也不是"TOO LITTLE"，则返回False, None
    return False, None


def solve_sex_problem(user_name: str) -> None:
    # 打印提示信息
    print("IS YOUR PROBLEM TOO MUCH OR TOO LITTLE?")
    # 无限循环，直到条件满足才退出
    while True:
        # 调用函数询问用户是否有太多或太少问题
        valid, too_much = prompt_too_much_or_too_little()
        # 如果用户输入有效
        if valid:
            # 如果用户有太多问题
            if too_much:
                # 打印相应的消息
                print("YOU CALL THAT A PROBLEM?!!  I SHOULD HAVE SUCH PROBLEMS!")
                print(f"IF IT BOTHERS YOU, {user_name}, TAKE A COLD SHOWER.")
            # 如果用户有太少问题
            else:
                # 打印相应的消息
                print(f"WHY ARE YOU HERE IN SUFFERN, {user_name}?  YOU SHOULD BE")
                print("IN TOKYO OR NEW YORK OR AMSTERDAM OR SOMEPLACE WITH SOME")
                print("REAL ACTION.")
            # 退出循环
            return
        # 如果用户输入无效
        else:
            # 提示用户重新输入
            print(f"DON'T GET ALL SHOOK, {user_name}, JUST ANSWER THE QUESTION")
            print("WITH 'TOO MUCH' OR 'TOO LITTLE'.  WHICH IS IT?")
# 定义解决金钱问题的函数，接受用户名称作为参数，不返回数值
def solve_money_problem(user_name: str) -> None:
    # 打印抱歉信息和建议
    print(f"SORRY, {user_name}, I'M BROKE TOO.  WHY DON'T YOU SELL")
    print("ENCYCLOPEADIAS OR MARRY SOMEONE RICH OR STOP EATING")
    print("SO YOU WON'T NEED SO MUCH MONEY?")


# 定义解决健康问题的函数，接受用户名称作为参数，不返回数值
def solve_health_problem(user_name: str) -> None:
    # 打印建议
    print(f"MY ADVICE TO YOU {user_name} IS:")
    print("     1.  TAKE TWO ASPRIN")
    print("     2.  DRINK PLENTY OF FLUIDS (ORANGE JUICE, NOT BEER!)")
    print("     3.  GO TO BED (ALONE)")


# 定义解决工作问题的函数，接受用户名称作为参数，不返回数值
def solve_job_problem(user_name: str) -> None:
    # 打印建议
    print(f"I CAN SYMPATHIZE WITH YOU {user_name}.  I HAVE TO WORK")
    print("VERY LONG HOURS FOR NO PAY -- AND SOME OF MY BOSSES")
    print(f"REALLY BEAT ON MY KEYBOARD.  MY ADVICE TO YOU, {user_name},")
    print("IS TO OPEN A RETAIL COMPUTER STORE.  IT'S GREAT FUN.")


# 定义处理未知问题类型的函数，接受用户名称和问题类型作为参数，不返回数值
def alert_unknown_problem_type(user_name: str, problem_type: str) -> None:
    # 打印未知问题类型的提示信息
    print(f"OH, {user_name}, YOUR ANSWER OF {problem_type} IS GREEK TO ME.")


# 定义循环询问问题类型的函数，接受用户名称作为参数，不返回数值
def ask_question_loop(user_name: str) -> None:
    # 进入无限循环
    while True:
        # 调用函数询问用户问题类型
        problem_type = prompt_for_problems(user_name)
        # 根据问题类型调用相应的解决问题函数
        if problem_type == "SEX":
            solve_sex_problem(user_name)
        elif problem_type == "HEALTH":
            solve_health_problem(user_name)
        elif problem_type == "MONEY":
            solve_money_problem(user_name)
        elif problem_type == "JOB":
            solve_job_problem(user_name)
        else:
            # 如果问题类型未知，则调用处理未知问题类型的函数
            alert_unknown_problem_type(user_name, problem_type)

        # 进入内部循环
        while True:
            print()
            print(f"ANY MORE PROBLEMS YOU WANT SOLVED, {user_name}?")

            # 调用函数获取用户输入的是或否
            valid, value, msg = get_yes_or_no()
            if valid:
                if value:
                    print("WHAT KIND (SEX, MONEY, HEALTH, JOB)")
                    break
                else:
                    return
            print(f"JUST A SIMPLE 'YES' OR 'NO' PLEASE, {user_name}.")


# 定义询问费用的函数，接受用户名称作为参数，不返回数值
def ask_for_fee(user_name: str) -> None:
    # 打印空行
    print()
    # 打印收费信息，包括用户名
    print(f"THAT WILL BE $5.00 FOR THE ADVICE, {user_name}.")
    # 提示用户在终端上留下钱
    print("PLEASE LEAVE THE MONEY ON THE TERMINAL.")
    # 等待4秒
    time.sleep(4)
    # 打印空行
    print()
    print()
    print()
    # 询问用户是否留下了钱
    print("DID YOU LEAVE THE MONEY?")

    # 无限循环，直到用户给出有效的答复
    while True:
        # 获取用户的是或否的回答
        valid, value, msg = get_yes_or_no()
        # 如果回答有效
        if valid:
            # 如果用户回答是
            if value:
                # 打印用户没有留下钱的信息
                print(f"HEY, {user_name}, YOU LEFT NO MONEY AT ALL!")
                print("YOU ARE CHEATING ME OUT OF MY HARD-EARNED LIVING.")
                print()
                print(f"WHAT A RIP OFF, {user_name}!!!")
                print()
            # 如果用户回答否
            else:
                # 打印用户诚实的信息
                print(f"THAT'S HONEST, {user_name}, BUT HOW DO YOU EXPECT")
                print("ME TO GO ON WITH MY PSYCHOLOGY STUDIES IF MY PATIENTS")
                print("DON'T PAY THEIR BILLS?")
            # 结束函数
            return
        # 如果回答无效
        else:
            # 提示用户回答无效
            print(f"YOUR ANSWER OF '{msg}' CONFUSES ME, {user_name}.")
            print("PLEASE RESPOND WITH 'YES' or 'NO'.")
# 定义一个函数，用于向用户输出不开心的告别信息
def unhappy_goodbye(user_name: str) -> None:
    # 输出空行
    print()
    # 输出带有用户名称的不开心告别信息
    print(f"TAKE A WALK, {user_name}.")
    # 输出空行
    print()


# 定义一个函数，用于向用户输出开心的告别信息
def happy_goodbye(user_name: str) -> None:
    # 输出带有用户名称的开心告别信息
    print(f"NICE MEETING YOU, {user_name}, HAVE A NICE DAY.")


# 定义主函数
def main() -> None:
    # 输出一行空格和"HELLO"
    print(" " * 33 + "HELLO")
    # 输出一行空格和"CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"，并空两行
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")
    # 输出"HELLO.  MY NAME IS CREATIVE COMPUTER."，并空两行
    print("HELLO.  MY NAME IS CREATIVE COMPUTER.\n\n")
    # 输出"WHAT'S YOUR NAME?"
    print("WHAT'S YOUR NAME?")
    # 获取用户输入的名称
    user_name = input()
    # 输出空行
    print()

    # 调用函数，询问用户是否喜欢计算机
    ask_enjoy_question(user_name)

    # 调用函数，循环询问用户问题
    ask_question_loop(user_name)

    # 调用函数，询问用户是否愿意支付费用
    ask_for_fee(user_name)

    # 如果条件为假
    if False:
        # 调用不开心的告别函数
        unhappy_goodbye(user_name)
    else:
        # 否则调用开心的告别函数
        happy_goodbye(user_name)


# 如果当前脚本作为主程序运行，则调用主函数
if __name__ == "__main__":
    main()
```