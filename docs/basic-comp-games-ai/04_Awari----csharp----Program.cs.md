# `basic-computer-games\04_Awari\csharp\Program.cs`

```
// 输出游戏标题
Console.WriteLine(Tab(34) + "AWARI");
// 输出游戏信息
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
        // 根据游戏状态执行相应操作
        switch (game.State)
        {
            case GameState.PlayerMove:
                // 玩家进行移动
                PlayerMove(second: false);
                break;
            case GameState.PlayerSecondMove:
                // 玩家进行第二次移动
                PlayerMove(second: true);
                break;
            case GameState.ComputerMove:
                // 电脑进行移动
                ComputerTurn();
                break;
        }

        // 显示游戏界面
        DisplayGame();
    }

    // 获取游戏结果
    var outcome = game.GetOutcome();

    // 根据结果输出相应信息
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
    // 显示电脑的坑
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

// 玩家进行移动
void PlayerMove(bool second = false)
{
    // 获取玩家移动的位置
    int move = GetMove(second);
    // 玩家进行移动
    game.PlayerMove(move);
}

// 获取玩家移动的位置
int GetMove(bool second)
{
    string prompt = second ? "AGAIN? " : "YOUR MOVE? ";

    while (true)
    {
        // 输出提示信息
        Console.Write(prompt);
    
        // 读取用户输入，如果为空则赋值为空字符串
        string input = Console.ReadLine() ?? "";
    
        // 检查输入是否为1到6之间的数字，并且当前位置的豆子数量大于0
        if (int.TryParse(input, out int move) && game.IsLegalPlayerMove(move))
            // 如果是合法移动，则返回移动的位置
            return move;
    
        // 如果移动不合法，则输出提示信息
        Console.WriteLine("ILLEGAL MOVE");
    }
# 定义一个名为 ComputerTurn 的函数，用于执行计算机的回合
void ComputerTurn()
{
    # 调用 game 对象的 ComputerTurn 方法，获取计算机的移动
    var moves = game.ComputerTurn();
    # 将移动列表转换为逗号分隔的字符串
    string movesString = string.Join(",", moves);

    # 在控制台输出计算机的移动
    Console.WriteLine($"MY MOVE IS {movesString}");
}

# 定义一个名为 Tab 的函数，用于生成指定数量空格的字符串
string Tab(int n) => new(' ', n);
```