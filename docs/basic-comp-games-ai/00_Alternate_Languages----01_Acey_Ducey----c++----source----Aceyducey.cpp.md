# `basic-computer-games\00_Alternate_Languages\01_Acey_Ducey\c++\source\Aceyducey.cpp`

```

#include <iostream> // 包含输入输出流库
#include <time.h> // 包含时间库
#include "Aceyducey.h" // 包含自定义的Aceyducey头文件

int main()
{
    // 设置随机数生成器的种子
    srand((unsigned int)time(NULL));
    bool isPlaying(true); // 初始化游戏状态为正在进行
    Money = 100; // 初始化玩家的初始资金为100
    WelcomeMessage(); // 调用欢迎消息函数
    while (isPlaying) // 当游戏正在进行时
    {
        Play(isPlaying); // 调用游戏进行函数
    }
    printf("O.K., HOPE YOU HAD FUN!\n"); // 打印游戏结束消息
}

void WelcomeMessage()
{
    for (int i = 0; i < 25; i++)
    {
        printf(" "); // 打印空格
    }
    printf("ACEY DUCEY CARD GAME\n"); // 打印游戏标题
    for (int i = 0; i < 14; i++)
    {
        printf(" "); // 打印空格
    }
    printf("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\nACEY-DUCEY IS PLAYED IN THE FOLLOWING MANNER \n");
    printf("THE DEALER (COMPUTER) DEALS TWO CARDS FACE UP\nYOU HAVE AN OPTION TO BET OR NOT BET DEPENDING\n");
    printf("ON WHETHER OR NOT YOU FEEL THE CARD WILL HAVE\nA VALUE BETWEEN THE FIRST TWO.\n");
    printf("IF YOU DO NOT WANT TO BET, INPUT A 0\n");
}

void Play(bool& isPlaying)
{
    short int DealerCards[2]; // 创建两张庄家的牌
    int Bet; // 玩家的赌注
    short int CurrentCard; // 玩家当前的牌
    printf("YOU NOW HAVE %d DOLLARS.\n\n", Money); // 打印玩家当前的资金
    printf("HERE ARE YOUR NEXT TWO CARDS: \n");

    //Draw Dealers Cards
    DrawCard(DealerCards[0]); // 绘制庄家的第一张牌
    printf("\n");
    DrawCard(DealerCards[1]); // 绘制庄家的第二张牌
    printf("\n\n\n");

    //Check if Bet is Valid
    do {
        printf("WHAT IS YOUR BET: "); // 提示玩家输入赌注
        std::cin >> Bet; // 读取玩家输入的赌注
        if (Bet == 0)
        {
            printf("CHICKEN!!\n\n"); // 如果玩家输入0，则打印提示信息
        }
    } while (Bet > Money || Bet < 0); // 当赌注大于玩家资金或小于0时，继续循环

    //Draw Players Card
    DrawCard(CurrentCard); // 绘制玩家的牌
    printf("\n");
    if (CurrentCard > DealerCards[0] && CurrentCard < DealerCards[1]) // 如果玩家的牌在庄家的两张牌之间
    {
        printf("YOU WIN!!!\n"); // 打印玩家获胜消息
        Money += Bet; // 玩家赢得赌注
        return;
    }
    else
    {
        printf("SORRY, YOU LOSE\n"); // 打印玩家失败消息
        Money -= Bet; // 玩家失去赌注
    }
    if (isGameOver()) // 如果游戏结束
    {
        printf("TRY AGAIN (YES OR NO)\n\n"); // 提示玩家是否再玩一次
        std::string response; // 创建字符串变量用于存储玩家的回答
        std::cin >> response; // 读取玩家的回答
        if (response != "YES") // 如果玩家回答不是YES
        {
            isPlaying = false; // 设置游戏状态为结束
        }
        Money = 100; // 重置玩家的资金为100
    }
}

bool isGameOver()
{
    if (Money <= 0) // 如果玩家的资金小于等于0
    {
        printf("\n\n");
        printf("SORRY, FRIEND, BUT YOU BLEW YOUR WAD.\n\n"); // 打印玩家破产消息
        return true; // 返回游戏结束
    }
    return false; // 返回游戏未结束
}

void DrawCard(short int& Card)
{
    //Basically generate 2 numbers first one is between 2-11 and second one 0-3
    short int RandomNum1 = (rand() % 10) + 2; // 生成2-11之间的随机数
    short int RandomNum2 = rand() % 4; // 生成0-3之间的随机数
    Card = RandomNum1 + RandomNum2; // 计算牌的点数

    switch (Card)
    {
    case 11:
        printf("JACK"); // 如果牌点数为11，打印JACK
        break;
    case 12:
        printf("QUEEN"); // 如果牌点数为12，打印QUEEN
        break;
    case 13:
        printf("KING"); // 如果牌点数为13，打印KING
        break;
    case 14:
        printf("ACE"); // 如果牌点数为14，打印ACE
        break;
    default:
        printf("%d", Card); // 否则打印牌的点数
    }
}

```