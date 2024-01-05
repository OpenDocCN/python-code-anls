# `d:/src/tocomm/basic-computer-games\04_Awari\csharp\Program.cs`

```
# 使用 Awari 命名空间中的 Tab 方法打印 AWARI 字样
Console.WriteLine(Tab(34) + "AWARI")
# 使用 Awari 命名空间中的 Tab 方法打印 CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY 字样
Console.WriteLine(Tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")

# 创建一个新的游戏对象
Game game = new()

# 进入游戏循环
while (true):
    # 重置游戏状态
    game.Reset()
    # 显示游戏界面
    DisplayGame()

    # 在游戏未结束的情况下进行循环
    while (game.State != GameState.Done):
        # 根据游戏状态进行不同的操作
        switch (game.State):
            # 当游戏状态为玩家移动时
            case GameState.PlayerMove:
                # 调用 PlayerMove 方法，传入参数为 false
                PlayerMove(second: false)
                break
            # 当游戏状态为玩家第二次移动时
            case GameState.PlayerSecondMove:
                PlayerMove(second: true);  # 调用名为PlayerMove的函数，传入参数second为true
                break;  # 跳出当前的switch语句
            case GameState.ComputerMove:  # 当游戏状态为ComputerMove时
                ComputerTurn();  # 调用名为ComputerTurn的函数
                break;  # 跳出当前的switch语句
        }

        DisplayGame();  # 调用名为DisplayGame的函数

    }

    var outcome = game.GetOutcome();  # 调用名为GetOutcome的函数，将结果赋值给变量outcome

    string outcomeLabel =  # 声明一个名为outcomeLabel的字符串变量
        outcome.Winner switch  # 根据outcome的Winner属性进行switch分支
        {
            GameWinner.Computer => $"I WIN BY {outcome.Difference} POINTS",  # 当Winner为Computer时，赋值特定的字符串
            GameWinner.Draw => "DRAWN GAME",  # 当Winner为Draw时，赋值特定的字符串
            GameWinner.Player => $"YOU WIN BY {outcome.Difference} POINTS",  # 当Winner为Player时，赋值特定的字符串
            _ => throw new InvalidOperationException($"Unexpected winner {outcome.Winner}."),  # 其他情况抛出异常
        };
    // 输出结果标签
    Console.WriteLine(outcomeLabel);
    // 输出空行
    Console.WriteLine();
}

void DisplayGame()
{
    // 显示计算机的坑
    Console.Write("   ");
    // 遍历计算机的坑并输出
    foreach (var pit in game.ComputerPits.Reverse())
        Console.Write($"{pit,2} ");
    Console.WriteLine();

    // 显示两个玩家的家
    Console.WriteLine($"{game.ComputerHome,2}{Tab(19)}{game.PlayerHome,2}");

    // 显示玩家的坑
    Console.Write("   ");
    // 遍历玩家的坑并输出
    foreach (var pit in game.PlayerPits)
        Console.Write($"{pit,2} ");
    Console.WriteLine();
    Console.WriteLine();  # 打印空行

}

void PlayerMove(bool second = false)  # 定义名为PlayerMove的函数，参数为bool类型的second，默认值为false
{
    int move = GetMove(second);  # 调用GetMove函数，将返回值赋给move变量
    game.PlayerMove(move);  # 调用game对象的PlayerMove方法，传入move参数
}

int GetMove(bool second)  # 定义名为GetMove的函数，参数为bool类型的second
{
    string prompt = second ? "AGAIN? " : "YOUR MOVE? ";  # 根据second的值选择不同的提示信息

    while (true)  # 进入无限循环
    {
        Console.Write(prompt);  # 打印提示信息

        string input = Console.ReadLine() ?? "";  # 读取用户输入的内容，如果为空则赋值为空字符串
// 检查输入是否为介于1和6之间的数字，并且坑中必须有大于0的豆子
if (int.TryParse(input, out int move)
 && game.IsLegalPlayerMove(move))
    return move;

Console.WriteLine("ILLEGAL MOVE");
```
这段代码用于检查玩家输入的移动是否合法，首先使用int.TryParse()方法将输入转换为整数，并将结果存储在move变量中，然后使用game.IsLegalPlayerMove()方法检查玩家的移动是否合法，如果合法则返回move，否则打印"ILLEGAL MOVE"。

```
void ComputerTurn()
{
    var moves = game.ComputerTurn();
    string movesString = string.Join(",", moves);

    Console.WriteLine($"MY MOVE IS {movesString}");
}
```
这段代码用于实现计算机的回合，首先调用game.ComputerTurn()方法获取计算机的移动，然后使用string.Join()方法将移动连接成字符串，最后打印出计算机的移动。

```
string Tab(int n) => new(' ', n);
```
这段代码定义了一个名为Tab的函数，用于生成包含n个空格的字符串。
```