# `basic-computer-games\04_Awari\csharp\Program.cs`

```

// 引入 Awari 命名空间
using Awari;

// 打印游戏标题
Console.WriteLine(Tab(34) + "AWARI");
Console.WriteLine(Tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");

// 创建游戏对象
Game game = new();

// 游戏循环
while (true)
{
    // 重置游戏状态
    game.Reset();
    // 显示游戏界面
    DisplayGame();

    // 游戏状态循环
    while (game.State != GameState.Done)
    {
        // 根据游戏状态执行相应的操作
        switch (game.State)
        {
            case GameState.PlayerMove:
                PlayerMove(second: false);
                break;
            case GameState.PlayerSecondMove:
                PlayerMove(second: true);
                break;
            case GameState.ComputerMove:
                ComputerTurn();
                break;
        }

        // 显示游戏界面
        DisplayGame();
    }

    // 获取游戏结果
    var outcome = game.GetOutcome();

    // 根据游戏结果输出对应的信息
    string outcomeLabel =
        outcome.Winner switch
        {
            GameWinner.Computer => $"I WIN BY {outcome.Difference} POINTS",
            GameWinner.Draw => "DRAWN GAME",
            GameWinner.Player => $"YOU WIN BY {outcome.Difference} POINTS",
            _ => throw new InvalidOperationException($"Unexpected winner {outcome.Winner}."),
        };
    Console.WriteLine(outcomeLabel);
    Console.WriteLine();
}

// 显示游戏界面
void DisplayGame()
{
    // 显示计算机的坑
    Console.Write("   ");
    foreach (var pit in game.ComputerPits.Reverse())
        Console.Write($"{pit,2} ");
    Console.WriteLine();

    // 显示双方的家
    Console.WriteLine($"{game.ComputerHome,2}{Tab(19)}{game.PlayerHome,2}");

    // 显示玩家的坑
    Console.Write("   ");
    foreach (var pit in game.PlayerPits)
        Console.Write($"{pit,2} ");
    Console.WriteLine();

    Console.WriteLine();
}

// 玩家移动
void PlayerMove(bool second = false)
{
    int move = GetMove(second);
    game.PlayerMove(move);
}

// 获取玩家移动
int GetMove(bool second)
{
    string prompt = second ? "AGAIN? " : "YOUR MOVE? ";

    while (true)
    {
        Console.Write(prompt);

        string input = Console.ReadLine() ?? "";

        // 输入必须是 1 到 6 之间的数字，并且坑里必须有 > 0 个豆子
        if (int.TryParse(input, out int move)
         && game.IsLegalPlayerMove(move))
            return move;

        Console.WriteLine("ILLEGAL MOVE");
    }
}

// 计算机回合
void ComputerTurn()
{
    var moves = game.ComputerTurn();
    string movesString = string.Join(",", moves);

    Console.WriteLine($"MY MOVE IS {movesString}");
}

// 生成指定数量的空格字符串
string Tab(int n) => new(' ', n);

```