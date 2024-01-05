# `d:/src/tocomm/basic-computer-games\00_Alternate_Languages\01_Acey_Ducey\c++\source\Aceyducey.cpp`

```
#include <iostream>
#include <time.h>
#include "Aceyducey.h"  # 包含自定义的头文件 "Aceyducey.h"

int main()
{
    //Setting Seed for the Random Generator  # 设置随机数生成器的种子
    srand((unsigned int)time(NULL));  # 使用当前时间作为种子初始化随机数生成器
    bool isPlaying(true);  # 初始化变量 isPlaying 为 true
    Money = 100;  # 初始化 Money 变量为 100
    WelcomeMessage();  # 调用自定义函数 WelcomeMessage()
    while (isPlaying)  # 进入循环，条件为 isPlaying 为 true
    {
        Play(isPlaying);  # 调用自定义函数 Play()，并传入 isPlaying 变量
    }
    printf("O.K., HOPE YOU HAD FUN!\n");  # 打印消息 "O.K., HOPE YOU HAD FUN!"
}

void WelcomeMessage()  # 定义自定义函数 WelcomeMessage()
{
    // 打印空格
    for (int i = 0; i < 25; i++)
    {
        printf(" ");
    }
    // 打印游戏标题
    printf("ACEY DUCEY CARD GAME\n");
    // 打印空格
    for (int i = 0; i < 14; i++)
    {
        printf(" ");
    }
    // 打印游戏信息
    printf("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\nACEY-DUCEY IS PLAYED IN THE FOLLOWING MANNER \n");
    printf("THE DEALER (COMPUTER) DEALS TWO CARDS FACE UP\nYOU HAVE AN OPTION TO BET OR NOT BET DEPENDING\n");
    printf("ON WHETHER OR NOT YOU FEEL THE CARD WILL HAVE\nA VALUE BETWEEN THE FIRST TWO.\n");
    printf("IF YOU DO NOT WANT TO BET, INPUT A 0\n");
}

void Play(bool& isPlaying)
{
    // 创建一个包含两个元素的整型数组，用于存储庄家的两张牌
    short int DealerCards[2];
    // 玩家的赌注
    int Bet;
    // 声明一个短整型变量 CurrentCard
    short int CurrentCard;
    // 打印玩家当前的金额
    printf("YOU NOW HAVE %d DOLLARS.\n\n", Money);
    // 打印玩家接下来的两张牌
    printf("HERE ARE YOUR NEXT TWO CARDS: \n");

    // 绘制庄家的牌
    DrawCard(DealerCards[0]);
    printf("\n");
    DrawCard(DealerCards[1]);
    printf("\n\n\n");

    // 检查下注是否有效
    do {
        // 提示玩家输入下注金额
        printf("WHAT IS YOUR BET: ");
        std::cin >> Bet;
        // 如果下注金额为0，则打印"CHICKEN!!"
        if (Bet == 0)
        {
            printf("CHICKEN!!\n\n");
        }
    } while (Bet > Money || Bet < 0);  // 当下注金额大于玩家金额或者小于0时，继续循环
    // 绘制玩家的卡片
    DrawCard(CurrentCard);
    printf("\n");
    // 如果当前卡片大于庄家的第一张卡片并且小于庄家的第二张卡片
    if (CurrentCard > DealerCards[0] && CurrentCard < DealerCards[1])
    {
        printf("YOU WIN!!!\n");
        Money += Bet;
        return;
    }
    else
    {
        printf("SORRY, YOU LOSE\n");
        Money -= Bet;
    }
    // 如果游戏结束
    if (isGameOver())
    {
        printf("TRY AGAIN (YES OR NO)\n\n");
        std::string response;
        std::cin >> response;
        // 如果回答不是"YES"
        {
            isPlaying = false;  # 将变量 isPlaying 设置为 false，表示游戏结束
        }
        Money = 100;  # 初始化变量 Money 为 100
    }
}

bool isGameOver()  # 定义函数 isGameOver，用于判断游戏是否结束
{
    if (Money <= 0)  # 如果 Money 小于等于 0
    {
        printf("\n\n");  # 输出两个换行符
        printf("SORRY, FRIEND, BUT YOU BLEW YOUR WAD.\n\n");  # 输出提示信息
        return true;  # 返回 true，表示游戏结束
    }
    return false;  # 返回 false，表示游戏未结束
}

void DrawCard(short int& Card)  # 定义函数 DrawCard，用于抽取一张卡片
    # 生成两个随机数，第一个在2-11之间，第二个在0-3之间
    RandomNum1 = (rand() % 10) + 2
    RandomNum2 = rand() % 4
    Card = RandomNum1 + RandomNum2

    # 根据生成的随机数判断卡片的类型并打印出来
    switch (Card):
        case 11:
            printf("JACK")
            break
        case 12:
            printf("QUEEN")
            break
        case 13:
            printf("KING")
            break
        case 14:
            printf("ACE")
            break
        default:
        printf("%d", Card);  # 打印变量 Card 的值
    }  # 结束内层循环
}  # 结束外层循环
```