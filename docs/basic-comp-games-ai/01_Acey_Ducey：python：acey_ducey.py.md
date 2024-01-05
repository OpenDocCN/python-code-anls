# `d:/src/tocomm/basic-computer-games\01_Acey_Ducey\python\acey_ducey.py`

```
    12: "Queen",
    13: "King",
    14: "Ace"
}

def play_acey_ducey():
    """
    Play the Acey-Ducey game
    """
    print("Welcome to Acey-Ducey!")
    print("You will be shown two cards. You must bet whether the next card will have a value between the first two.")
    print("If you win, you get the amount you bet. If you lose, you lose the amount you bet.")
    print("You start with $100.")
    money = 100
    while money > 0:
        print(f"You have ${money}.")
        bet = int(input("How much do you want to bet? "))
        if bet > money:
            print("You don't have enough money to bet that much.")
            continue
        first_card = random.choice(list(cards.keys()))
        second_card = random.choice(list(cards.keys()))
        print(f"The first card is {cards[first_card]}.")
        print(f"The second card is {cards[second_card]}.")
        if first_card < second_card:
            low = first_card
            high = second_card
        else:
            low = second_card
            high = first_card
        guess = int(input(f"What is your bet? Enter a number between {low} and {high}: "))
        next_card = random.choice(list(cards.keys()))
        print(f"The next card is {cards[next_card]}.")
        if low < next_card < high:
            print("You win!")
            money += bet
        else:
            print("You lose!")
            money -= bet
    print("Game over. You are out of money.")

# 主程序
if __name__ == "__main__":
    play_acey_ducey()
    12: "Queen",  # 12号牌对应的值为"Queen"
    13: "King",   # 13号牌对应的值为"King"
    14: "Ace",    # 14号牌对应的值为"Ace"
}


def play_game() -> None:  # 定义一个名为play_game的函数，不返回任何结果
    cash = 100  # 初始化玩家的现金为100
    while cash > 0:  # 当玩家的现金大于0时循环执行下面的代码
        print(f"You now have {cash} dollars\n")  # 打印玩家当前的现金余额
        print("Here are you next two cards")  # 打印提示信息
        round_cards = list(cards.keys())  # 将牌的编号转换为列表
        card_a = random.choice(round_cards)  # 从列表中随机选择一张牌
        card_b = card_a  # 复制第一张牌，以避免第二张牌与第一张相同
        while (card_a == card_b):  # 如果两张牌相同，则重新选择第二张牌
            card_b = random.choice(round_cards)
        card_c = random.choice(round_cards)  # 随机选择第三张牌
        if card_a > card_b:  # 如果第一张牌大于第二张牌，则交换它们的位置
            card_a, card_b = card_b, card_a
        print(f" {cards[card_a]}")  # 打印第一张牌对应的值
        # 打印出卡片B的内容
        print(f" {cards[card_b]}\n")
        # 无限循环，直到用户输入有效的赌注
        while True:
            try:
                # 获取用户输入的赌注
                bet = int(input("What is your bet? "))
                # 如果赌注小于0，则抛出数值错误
                if bet < 0:
                    raise ValueError("Bet must be more than zero")
                # 如果赌注为0，则打印"CHICKEN!!"并继续循环
                if bet == 0:
                    print("CHICKEN!!\n")
                # 如果赌注大于现金数额，则打印错误信息并继续循环
                if bet > cash:
                    print("Sorry, my friend but you bet too much")
                    print(f"You only have {cash} dollars to bet")
                    continue
                # 从现金中扣除赌注金额
                cash -= bet
                # 跳出循环
                break

            # 捕获数值错误，提示用户输入正数
            except ValueError:
                print("Please enter a positive number")
        # 打印出卡片C的内容
        print(f" {cards[card_c]}")
        # 如果赌注大于0且卡片C在卡片A和卡片B之间
        if bet > 0:
            if card_a <= card_c <= card_b:
                print("You win!!!")  # 打印出玩家赢了的消息
                cash += bet * 2  # 如果玩家赢了，将赌注翻倍加到现金中
            else:
                print("Sorry, you lose")  # 打印出玩家输了的消息

    print("Sorry, friend, but you blew your wad")  # 打印出玩家输光了所有的钱


def main() -> None:
    print(
        """
Acey-Ducey is played in the following manner
The dealer (computer) deals two cards face up
You have an option to bet or not bet depending
on whether or not you feel the card will have
a value between the first two.
If you do not want to bet, input a 0
  """
    )  # 打印出游戏规则的说明
    keep_playing = True  # 设置一个变量来表示玩家是否继续游戏
    while keep_playing:  # 当 keep_playing 为 True 时循环执行下面的代码
        play_game()  # 调用 play_game() 函数来进行游戏
        keep_playing = input("Try again? (yes or no) ").lower().startswith("y")  # 获取用户输入，如果以 "y" 开头则将 keep_playing 设置为 True，否则设置为 False
    print("Ok hope you had fun")  # 打印消息，表示游戏结束

if __name__ == "__main__":  # 如果当前脚本被直接执行，则执行下面的代码
    random.seed()  # 初始化随机数生成器
    main()  # 调用 main() 函数来执行程序的主要逻辑
```