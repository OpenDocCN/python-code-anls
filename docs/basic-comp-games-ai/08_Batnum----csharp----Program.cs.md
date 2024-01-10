# `basic-computer-games\08_Batnum\csharp\Program.cs`

```
// 引入 Batnum 命名空间
using Batnum;
// 引入 Batnum.Properties 命名空间
using Batnum.Properties;
// 引入 System 命名空间
using System;

// 居中打印游戏名称
Console.WriteLine(ConsoleUtilities.CenterText(Resources.GAME_NAME));
// 居中打印游戏介绍标题
Console.WriteLine(ConsoleUtilities.CenterText(Resources.INTRO_HEADER));
// 打印空行
Console.WriteLine();
Console.WriteLine();
Console.WriteLine();
// 打印游戏介绍第一部分
ConsoleUtilities.WriteLineWordWrap(Resources.INTRO_PART1);
// 打印空行
Console.WriteLine();
// 打印游戏介绍第二部分
ConsoleUtilities.WriteLineWordWrap(Resources.INTRO_PART2);

// 游戏循环
while (true)
{
    // 打印空行
    Console.WriteLine();
    // 询问并获取初始堆大小
    int pileSize = ConsoleUtilities.AskNumberQuestion(Resources.START_QUESTION_PILESIZE, (n) => n > 1);
    // 询问并获取获胜选项
    WinOptions winOption = (WinOptions)ConsoleUtilities.AskNumberQuestion(Resources.START_QUESTION_WINOPTION, (n) => Enum.IsDefined(typeof(WinOptions), n));
    // 询问并获取取石子范围
    (int minTake, int maxTake) = ConsoleUtilities.AskNumberRangeQuestion(Resources.START_QUESTION_DRAWMINMAX, (min,max) => min >= 1 && max < pileSize && max > min);
    // 询问并获取先手玩家
    Players currentPlayer = (Players)ConsoleUtilities.AskNumberQuestion(Resources.START_QUESTION_WHOSTARTS, (n) => Enum.IsDefined(typeof(Players), n));

    // 创建 BatnumGame 对象
    BatnumGame game = new BatnumGame(pileSize, winOption, minTake, maxTake, currentPlayer, (question) => ConsoleUtilities.AskNumberQuestion(question, (c) => true));
    // 游戏进行中
    while(game.IsRunning)
    {
        // 玩家轮流取石子
        string message = game.TakeTurn();
        // 打印消息
        Console.WriteLine(message);
    }

}
```