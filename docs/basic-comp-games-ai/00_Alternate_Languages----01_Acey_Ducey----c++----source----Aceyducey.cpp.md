# `basic-computer-games\00_Alternate_Languages\01_Acey_Ducey\c++\source\Aceyducey.cpp`

```py
#include <iostream> // 包含输入输出流库
#include <time.h> // 包含时间库
#include "Aceyducey.h" // 包含自定义的Aceyducey头文件

int main() // 主函数
{
    srand((unsigned int)time(NULL)); // 设置随机数生成器的种子
    bool isPlaying(true); // 初始化游戏状态为正在进行
    Money = 100; // 初始化玩家的初始资金为100
    WelcomeMessage(); // 调用欢迎消息函数
    while (isPlaying) // 当游戏正在进行时
    {
        Play(isPlaying); // 调用游戏进行函数
    }
    printf("O.K., HOPE YOU HAD FUN!\n"); // 打印结束语
}

void WelcomeMessage() // 欢迎消息函数
{
    for (int i = 0; i < 25; i++) // 打印空格
    {
        printf(" ");
    }
    printf("ACEY DUCEY CARD GAME\n"); // 打印游戏标题
    for (int i = 0; i < 14; i++) // 打印空格
    {
        printf(" ");
    }
    printf("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\nACEY-DUCEY IS PLAYED IN THE FOLLOWING MANNER \n"); // 打印游戏玩法说明
    printf("THE DEALER (COMPUTER) DEALS TWO CARDS FACE UP\nYOU HAVE AN OPTION TO BET OR NOT BET DEPENDING\n"); // 打印游戏规则
    printf("ON WHETHER OR NOT YOU FEEL THE CARD WILL HAVE\nA VALUE BETWEEN THE FIRST TWO.\n");
    printf("IF YOU DO NOT WANT TO BET, INPUT A 0\n");
}

void Play(bool& isPlaying) // 游戏进行函数
{
    short int DealerCards[2]; // 定义存储庄家卡牌的数组
    int Bet; // 玩家下注金额
    short int CurrentCard; // 玩家当前的卡牌
    printf("YOU NOW HAVE %d DOLLARS.\n\n", Money); // 打印玩家当前资金
    printf("HERE ARE YOUR NEXT TWO CARDS: \n"); // 打印玩家下两张卡牌

    //Draw Dealers Cards
    DrawCard(DealerCards[0]); // 抽取庄家的第一张卡牌
    printf("\n");
    DrawCard(DealerCards[1]); // 抽取庄家的第二张卡牌
    printf("\n\n\n");

    //Check if Bet is Valid
    do {
        printf("WHAT IS YOUR BET: "); // 提示玩家下注
        std::cin >> Bet; // 输入玩家下注金额
        if (Bet == 0) // 如果玩家选择不下注
        {
            printf("CHICKEN!!\n\n"); // 提示玩家选择了不下注
        }
    } while (Bet > Money || Bet < 0); // 当下注金额大于玩家资金或小于0时，重新输入下注金额

    //Draw Players Card
    DrawCard(CurrentCard); // 抽取玩家的卡牌
    printf("\n");
    if (CurrentCard > DealerCards[0] && CurrentCard < DealerCards[1]) // 如果玩家的卡牌在庄家两张卡牌之间
    {
        printf("YOU WIN!!!\n"); // 玩家获胜
        Money += Bet; // 玩家获得下注金额
        return;
    }
    else
    {
        printf("SORRY, YOU LOSE\n"); // 玩家失败
        Money -= Bet; // 玩家失去下注金额
    }
    if (isGameOver()) // 如果游戏结束
    {
        printf("TRY AGAIN (YES OR NO)\n\n"); // 提示玩家是否再玩一次
        std::string response; // 定义存储玩家回应的字符串
        std::cin >> response; // 输入玩家回应
        if (response != "YES") // 如果玩家回应不是YES
        {
            isPlaying = false; // 设置游戏状态为结束
        }
        Money = 100; // 重置玩家资金为100
    }
}

bool isGameOver() // 判断游戏是否结束的函数
{
    // 如果玩家的钱小于等于0，则输出提示信息并返回true
    if (Money <= 0)
    {
        printf("\n\n");
        printf("SORRY, FRIEND, BUT YOU BLEW YOUR WAD.\n\n");
        return true;
    }
    // 否则返回false
    return false;
}

void DrawCard(short int& Card)
{
    // 生成两个随机数，第一个在2-11之间，第二个在0-3之间
    short int RandomNum1 = (rand() % 10) + 2;
    short int RandomNum2 = rand() % 4;
    // 将两个随机数相加得到卡片的点数
    Card = RandomNum1 + RandomNum2;

    // 根据卡片的点数输出相应的牌面
    switch (Card)
    {
    case 11:
        printf("JACK");
        break;
    case 12:
        printf("QUEEN");
        break;
    case 13:
        printf("KING");
        break;
    case 14:
        printf("ACE");
        break;
    default:
        printf("%d", Card);
    }
}
```