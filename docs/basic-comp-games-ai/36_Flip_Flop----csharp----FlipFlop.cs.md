# `basic-computer-games\36_Flip_Flop\csharp\FlipFlop.cs`

```
// 打印游戏信息
PrintGameInfo();

// 是否开始新游戏
bool startNewGame = true;

// 初始化游戏棋盘
string[] board = new string[] { "X", "X", "X", "X", "X", "X", "X", "X", "X", "X" };

// 开始游戏循环
do
{
    // 步数计数
    int stepsCount = 0;
    // 上一步移动的位置
    int lastMove = -1;
    // 移动位置索引
    int moveIndex;
    // 游戏总和
    int gameSum;
    // 游戏熵率
    double gameEntropyRate = Rnd();
    // 是否轮到玩家
    bool toPlay = false;
    // 是否设置新的棋盘
    bool setNewBoard = true;

    // 打印当前棋盘
    Print();
    Print("HERE IS THE STARTING LINE OF X'S.");
    Print();

    // 开始游戏步数循环
    do
    }
    while (stepsCount < 12 && gameSum < 10);

    // 如果轮到玩家
    if (toPlay)
    {
        // 打印游戏结果
        PrintGameResult(gameSum, stepsCount);

        // 询问玩家是否继续
        Write("DO YOU WANT TO TRY ANOTHER PUZZLE ");

        var toContinue = Console.ReadLine();

        // 如果玩家输入不为空且以"N"开头，则不开始新游戏
        if (!string.IsNullOrEmpty(toContinue) && toContinue?.ToUpper()[0] == 'N')
        {
            startNewGame = false;
        }

        Print();
    }
}
while (startNewGame);

// 打印字符串
void Print(string str = "") => Console.WriteLine(str);

// 输出字符串
void Write(string value) => Console.Write(value);

// 生成指定长度的空格字符串
string Tab(int pos) => new(' ', pos);

// 生成随机熵率
double Rnd() => new Random().NextDouble();

// 获取移动位置索引
int GetMoveIndex(int moveIndex, double gameEntropyRate)
{
    // 计算移动位置索引
    double rate = Math.Tan(gameEntropyRate + moveIndex / gameEntropyRate - moveIndex) - Math.Sin(gameEntropyRate / moveIndex) + 336 * Math.Sin(8 * moveIndex);
    return Convert.ToInt32(Math.Floor(10 * (rate - Math.Floor(rate))));
}

// 当移动位置索引等于上一步移动位置时，获取新的移动位置索引
int GetMoveIndexWhenEqualeLastMove(int moveIndex, double gameEntropyRate)
{
    // 计算移动位置索引
    double rate = 0.592 * (1 / Math.Tan(gameEntropyRate / moveIndex + gameEntropyRate)) / Math.Sin(moveIndex * 2 + gameEntropyRate) - Math.Cos(moveIndex);
    return Convert.ToInt32(Math.Floor(10 * (rate - Math.Floor(rate))));
}

// 打印新的棋盘
void PrintNewBoard()
{
    Print("1 2 3 4 5 6 7 8 9 10");
    Print("X X X X X X X X X X");
    Print();
}

// 打印游戏棋盘
void PrintGameBoard(string[] board)
{
    Print("1 2 3 4 5 6 7 8 9 10");

    // 遍历棋盘并打印
    foreach (var item in board)
    {
        Write($"{item} ");
    }

    Print();
    Print();
}

// 打印游戏结果
void PrintGameResult(int gameSum, int stepsCount)
{
    // 如果游戏总和为10
    {
        # 如果猜对了，输出猜对的消息和猜测次数
        Print($"VERY GOOD.  YOU GUESSED IT IN ONLY {stepsCount} GUESSES.");
    }
    else
    {
        # 如果猜错了，输出猜错的消息和猜测次数
        Print($"TRY HARDER NEXT TIME.  IT TOOK YOU {stepsCount} GUESSES.");
    }
# 打印游戏信息
void PrintGameInfo()
{
    # 打印游戏名称和制作信息
    Print(Tab(32) + "FLIPFLOP");
    Print(Tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
    Print();
    # 打印游戏目标
    Print("THE OBJECT OF THIS PUZZLE IS TO CHANGE THIS:");
    Print();
    # 打印示例图案
    Print("X X X X X X X X X X");
    Print();
    # 打印目标图案
    Print("TO THIS:");
    Print();
    Print("O O O O O O O O O O");
    Print();
    # 打印游戏说明
    Print("BY TYPING THE NUMBER CORRESPONDING TO THE POSITION OF THE");
    Print("LETTER ON SOME NUMBERS, ONE POSITION WILL CHANGE, ON");
    Print("OTHERS, TWO WILL CHANGE.  TO RESET LINE TO ALL X'S, TYPE 0");
    Print("(ZERO) AND TO START OVER IN THE MIDDLE OF A GAME, TYPE ");
    Print("11 (ELEVEN).");
}
```