# `basic-computer-games\08_Batnum\csharp\Program.cs`

```

# 导入 Batnum 模块和相关资源
using Batnum;
using Batnum.Properties;
using System;

# 打印游戏名称
Console.WriteLine(ConsoleUtilities.CenterText(Resources.GAME_NAME));
# 打印游戏介绍标题
Console.WriteLine(ConsoleUtilities.CenterText(Resources.INTRO_HEADER));
# 打印空行
Console.WriteLine();
Console.WriteLine();
Console.WriteLine();
# 打印游戏介绍部分1
ConsoleUtilities.WriteLineWordWrap(Resources.INTRO_PART1);
# 打印空行
Console.WriteLine();
# 打印游戏介绍部分2
ConsoleUtilities.WriteLineWordWrap(Resources.INTRO_PART2);

# 游戏循环
while (true)
{
    # 打印空行
    Console.WriteLine();
    # 询问玩家初始堆大小，要求输入大于1的数字
    int pileSize = ConsoleUtilities.AskNumberQuestion(Resources.START_QUESTION_PILESIZE, (n) => n > 1);
    # 询问玩家胜利条件，要求输入预定义的枚举值
    WinOptions winOption = (WinOptions)ConsoleUtilities.AskNumberQuestion(Resources.START_QUESTION_WINOPTION, (n) => Enum.IsDefined(typeof(WinOptions), n));
    # 询问玩家取石子的范围，要求输入最小值大于等于1，最大值小于堆大小且大于最小值
    (int minTake, int maxTake) = ConsoleUtilities.AskNumberRangeQuestion(Resources.START_QUESTION_DRAWMINMAX, (min,max) => min >= 1 && max < pileSize && max > min);
    # 询问玩家谁先开始，要求输入预定义的枚举值
    Players currentPlayer = (Players)ConsoleUtilities.AskNumberQuestion(Resources.START_QUESTION_WHOSTARTS, (n) => Enum.IsDefined(typeof(Players), n));

    # 创建 BatnumGame 对象，传入游戏参数和询问玩家输入的方法
    BatnumGame game = new BatnumGame(pileSize, winOption, minTake, maxTake, currentPlayer, (question) => ConsoleUtilities.AskNumberQuestion(question, (c) => true));
    # 游戏循环
    while(game.IsRunning)
    {
        # 玩家轮流取石子，打印消息
        string message = game.TakeTurn();
        Console.WriteLine(message);
    }

}

```