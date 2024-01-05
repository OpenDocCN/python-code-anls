# `08_Batnum\csharp\Program.cs`

```
// 使用 Batnum 命名空间
using Batnum;
// 使用 Batnum.Properties 命名空间
using Batnum.Properties;
// 使用 System 命名空间
using System;

// 在控制台中居中打印游戏名称
Console.WriteLine(ConsoleUtilities.CenterText(Resources.GAME_NAME));
// 在控制台中居中打印介绍标题
Console.WriteLine(ConsoleUtilities.CenterText(Resources.INTRO_HEADER));
// 打印空行
Console.WriteLine();
Console.WriteLine();
Console.WriteLine();
// 在控制台中打印介绍部分1
ConsoleUtilities.WriteLineWordWrap(Resources.INTRO_PART1);
// 打印空行
Console.WriteLine();
// 在控制台中打印介绍部分2
ConsoleUtilities.WriteLineWordWrap(Resources.INTRO_PART2);

// 进入游戏循环
while (true)
{
    // 打印空行
    Console.WriteLine();
    // 询问玩家初始堆大小，并要求输入大于1的数字
    int pileSize = ConsoleUtilities.AskNumberQuestion(Resources.START_QUESTION_PILESIZE, (n) => n > 1);
    // 询问玩家胜利条件，并要求输入合法的 WinOptions 枚举值
    WinOptions winOption = (WinOptions)ConsoleUtilities.AskNumberQuestion(Resources.START_QUESTION_WINOPTION, (n) => Enum.IsDefined(typeof(WinOptions), n));
    // 询问玩家每次取石子的范围，并要求输入合法的范围
    (int minTake, int maxTake) = ConsoleUtilities.AskNumberRangeQuestion(Resources.START_QUESTION_DRAWMINMAX, (min,max) => min >= 1 && max < pileSize && max > min);
    // 询问玩家谁先开始，并要求输入合法的 Players 枚举值
    Players currentPlayer = (Players)ConsoleUtilities.AskNumberQuestion(Resources.START_QUESTION_WHOSTARTS, (n) => Enum.IsDefined(typeof(Players), n));
}
    # 创建一个猜数字游戏对象，传入参数包括堆大小、获胜选项、最小取数、最大取数、当前玩家和一个用于询问玩家问题的回调函数
    game = BatnumGame(pileSize, winOption, minTake, maxTake, currentPlayer, (question) => ConsoleUtilities.AskNumberQuestion(question, (c) => true))
    # 当游戏正在运行时循环执行以下操作
    while game.IsRunning:
        # 执行游戏的一个回合，并获取回合信息
        message = game.TakeTurn()
        # 在控制台打印回合信息
        Console.WriteLine(message)
```