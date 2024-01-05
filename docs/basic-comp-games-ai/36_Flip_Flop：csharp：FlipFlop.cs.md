# `36_Flip_Flop\csharp\FlipFlop.cs`

```
# Flip Flop Game

# 打印游戏信息
PrintGameInfo();

# 设置开始新游戏的标志
bool startNewGame = true;

# 初始化游戏棋盘
string[] board = new string[] { "X", "X", "X", "X", "X", "X", "X", "X", "X", "X" };

# 开始游戏循环
do
{
    # 初始化步数计数
    int stepsCount = 0;
    # 初始化上一步移动的位置
    int lastMove = -1;
    # 初始化移动的索引
    int moveIndex;
    # 初始化游戏总和
    int gameSum;
    # 初始化游戏熵率
    double gameEntropyRate = Rnd();
    # 初始化是否轮到玩家行动的标志
    bool toPlay = false;
    # 初始化是否设置新的棋盘的标志
    bool setNewBoard = true;

    # 打印当前棋盘
    Print();
    # 打印提示信息
    Print("HERE IS THE STARTING LINE OF X'S.");
    # 打印函数
    Print();

    # 开始循环
    do
    {
        # 定义变量，用于检查是否有非法输入和是否与上一步相同
        bool illegalEntry;
        bool equalToLastMove;

        # 如果需要设置新的棋盘
        if (setNewBoard)
        {
            # 打印新的棋盘
            PrintNewBoard();
            # 初始化棋盘
            board = new string[] { "X", "X", "X", "X", "X", "X", "X", "X", "X", "X" };
            # 设置新棋盘标志为假
            setNewBoard = false;
            # 设置下一步为玩家
            toPlay = true;
        }

        # 步数加一
        stepsCount++;
        # 游戏总数清零
        gameSum = 0;

        # 读取用户的移动
        do
        {
            // 提示用户输入数字
            Write("INPUT THE NUMBER? ");
            // 读取用户输入
            var input = Console.ReadLine();
            // 检查用户输入是否合法
            illegalEntry = !int.TryParse(input, out moveIndex);

            // 如果用户输入不合法或者移动索引大于11
            if (illegalEntry || moveIndex > 11)
            {
                // 将illegalEntry设置为true
                illegalEntry = true;
                // 打印提示信息
                Print("ILLEGAL ENTRY--TRY AGAIN.");
            }
        }
        // 当用户输入不合法时重复执行上述步骤
        while (illegalEntry);

        // 如果移动索引为11
        if (moveIndex == 11)
        {
            // 运行新游戏，随时开始新游戏
            toPlay = false;
            stepsCount = 12;
            startNewGame = true;
        }
        if (moveIndex == 0):
            # 如果移动索引为0，表示需要重置游戏盘，将所有位置设为X，并设置为不需要进行下一步移动
            setNewBoard = True
            toPlay = False

        if (toPlay):
            # 如果需要进行下一步移动
            # 将上一步移动的位置设为相反的符号（如果是O则设为X，反之亦然）
            board[moveIndex - 1] = board[moveIndex - 1] == "O" ? "X" : "O"

            if (lastMove == moveIndex):
                # 如果当前移动位置与上一步相同，设置equalToLastMove为True
                equalToLastMove = True
            else:
                # 如果当前移动位置与上一步不同，设置equalToLastMove为False
                equalToLastMove = False
                lastMove = moveIndex;  # 将当前移动的索引赋值给lastMove变量，用于记录上一次的移动位置

            }

            do
            {
                moveIndex = equalToLastMove  # 如果equalToLastMove为真，则调用GetMoveIndexWhenEqualeLastMove函数，否则调用GetMoveIndex函数
                    ? GetMoveIndexWhenEqualeLastMove(moveIndex, gameEntropyRate)
                    : GetMoveIndex(moveIndex, gameEntropyRate);

                board[moveIndex] = board[moveIndex] == "O" ? "X" : "O";  # 根据当前位置的棋子类型进行翻转，如果是"O"则变为"X"，如果是"X"则变为"O"
            }
            while (lastMove == moveIndex && board[moveIndex] == "X");  # 当上一次移动的位置等于当前移动的位置，并且当前位置的棋子类型为"X"时，继续循环

            PrintGameBoard(board);  # 打印游戏棋盘

            foreach (var item in board)  # 遍历棋盘上的每个位置
            {
                if (item == "O")  # 如果当前位置的棋子类型为"O"
                {
                    gameSum++;  # 游戏总数加一
    }
            } // 结束内层循环
        } // 结束外层循环
    } // 结束游戏总和小于10的条件判断

    if (toPlay) // 如果需要继续游戏
    {
        PrintGameResult(gameSum, stepsCount); // 打印游戏结果

        Write("DO YOU WANT TO TRY ANOTHER PUZZLE "); // 提示用户是否想尝试另一个谜题

        var toContinue = Console.ReadLine(); // 读取用户输入

        if (!string.IsNullOrEmpty(toContinue) && toContinue?.ToUpper()[0] == 'N') // 如果用户输入不为空且第一个字符为N
        {
            startNewGame = false; // 设置开始新游戏为false
        }

        Print(); // 打印空行
while (startNewGame);
```
这是一个while循环的开始，它的条件是startNewGame。当startNewGame为true时，循环会一直执行。

```
void Print(string str = "") => Console.WriteLine(str);
```
这是一个名为Print的函数，它接受一个字符串参数并将其打印到控制台上。

```
void Write(string value) => Console.Write(value);
```
这是一个名为Write的函数，它接受一个字符串参数并将其写入到控制台上。

```
string Tab(int pos) => new(' ', pos);
```
这是一个名为Tab的函数，它接受一个整数参数并返回一个由空格组成的字符串，长度为pos。

```
double Rnd() => new Random().NextDouble();
```
这是一个名为Rnd的函数，它创建一个Random对象并调用其NextDouble方法，返回一个随机的双精度浮点数。

```
int GetMoveIndex(int moveIndex, double gameEntropyRate)
{
    double rate = Math.Tan(gameEntropyRate + moveIndex / gameEntropyRate - moveIndex) - Math.Sin(gameEntropyRate / moveIndex) + 336 * Math.Sin(8 * moveIndex);
    return Convert.ToInt32(Math.Floor(10 * (rate - Math.Floor(rate))));
}
```
这是一个名为GetMoveIndex的函数，它接受两个参数并根据一定的数学计算返回一个整数值。

```
int GetMoveIndexWhenEqualeLastMove(int moveIndex, double gameEntropyRate)
{
```
这是一个名为GetMoveIndexWhenEqualeLastMove的函数，它接受两个参数但是没有实现函数体。
    // 计算一个复杂的数学表达式并将结果存储在变量rate中
    double rate = 0.592 * (1 / Math.Tan(gameEntropyRate / moveIndex + gameEntropyRate)) / Math.Sin(moveIndex * 2 + gameEntropyRate) - Math.Cos(moveIndex);
    // 将rate转换为整数并返回
    return Convert.ToInt32(Math.Floor(10 * (rate - Math.Floor(rate))));
}

// 打印一个新的游戏板
void PrintNewBoard()
{
    // 打印游戏板的第一行
    Print("1 2 3 4 5 6 7 8 9 10");
    // 打印游戏板的第二行
    Print("X X X X X X X X X X");
    // 打印空行
    Print();
}

// 打印游戏板的内容
void PrintGameBoard(string[] board)
{
    // 打印游戏板的第一行
    Print("1 2 3 4 5 6 7 8 9 10");

    // 遍历游戏板数组并逐个打印
    foreach (var item in board)
    {
        Write($"{item} ");
    }
}
    # 调用Print函数，但未提供参数
    Print();
    # 调用Print函数，但未提供参数
    Print();
}

# 定义PrintGameResult函数，接受gameSum和stepsCount两个参数
void PrintGameResult(int gameSum, int stepsCount)
{
    # 如果gameSum等于10，打印“VERY GOOD.  YOU GUESSED IT IN ONLY {stepsCount} GUESSES.”
    if (gameSum == 10)
    {
        Print($"VERY GOOD.  YOU GUESSED IT IN ONLY {stepsCount} GUESSES.");
    }
    # 否则，打印“TRY HARDER NEXT TIME.  IT TOOK YOU {stepsCount} GUESSES.”
    else
    {
        Print($"TRY HARDER NEXT TIME.  IT TOOK YOU {stepsCount} GUESSES.");
    }
}

# 定义PrintGameInfo函数
void PrintGameInfo()
{
    # 打印“FLIPFLOP”并在前面加上32个空格
    Print(Tab(32) + "FLIPFLOP");
    # 打印“CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY”并在前面加上15个空格
    Print(Tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
    # 打印空行
    Print();
    # 打印提示信息
    Print("THE OBJECT OF THIS PUZZLE IS TO CHANGE THIS:");
    # 打印空行
    Print();

    # 打印提示信息
    Print("X X X X X X X X X X");
    # 打印空行
    Print();
    # 打印提示信息
    Print("TO THIS:");
    # 打印空行
    Print();
    # 打印提示信息
    Print("O O O O O O O O O O");
    # 打印空行
    Print();

    # 打印提示信息
    Print("BY TYPING THE NUMBER CORRESPONDING TO THE POSITION OF THE");
    # 打印提示信息
    Print("LETTER ON SOME NUMBERS, ONE POSITION WILL CHANGE, ON");
    # 打印提示信息
    Print("OTHERS, TWO WILL CHANGE.  TO RESET LINE TO ALL X'S, TYPE 0");
    # 打印提示信息
    Print("(ZERO) AND TO START OVER IN THE MIDDLE OF A GAME, TYPE ");
    # 打印提示信息
    Print("11 (ELEVEN).");
}
```