# `basic-computer-games\36_Flip_Flop\csharp\FlipFlop.cs`

```

// Flip Flop Game

// 打印游戏信息
PrintGameInfo();

// 是否开始新游戏
bool startNewGame = true;

// 初始化游戏棋盘
string[] board = new string[] { "X", "X", "X", "X", "X", "X", "X", "X", "X", "X" };

// 开始游戏循环
do
{
    int stepsCount = 0; // 步数计数
    int lastMove = -1; // 上一步移动的位置
    int moveIndex; // 移动的位置
    int gameSum; // 游戏总数
    double gameEntropyRate = Rnd(); // 游戏熵率
    bool toPlay = false; // 是否进行游戏
    bool setNewBoard = true; // 是否设置新的游戏棋盘

    // 打印游戏棋盘
    Print();
    Print("HERE IS THE STARTING LINE OF X'S.");
    Print();

    // 游戏进行循环
    do
    {
        bool illegalEntry; // 非法输入
        bool equalToLastMove; // 是否与上一步相同

        if (setNewBoard)
        {
            // 打印新的游戏棋盘
            PrintNewBoard();
            board = new string[] { "X", "X", "X", "X", "X", "X", "X", "X", "X", "X" };
            setNewBoard = false;
            toPlay = true;
        }

        stepsCount++; // 步数加一
        gameSum = 0;

        // 读取用户的移动
        do
        {
            Write("INPUT THE NUMBER? ");
            var input = Console.ReadLine();
            illegalEntry = !int.TryParse(input, out moveIndex);

            if (illegalEntry || moveIndex > 11)
            {
                illegalEntry = true;
                Print("ILLEGAL ENTRY--TRY AGAIN.");
            }
        }
        while (illegalEntry);

        if (moveIndex == 11)
        {
            // 开始新游戏
            toPlay = false;
            stepsCount = 12;
            startNewGame = true;
        }

        if (moveIndex == 0)
        {
            // 重置游戏棋盘
            setNewBoard = true;
            toPlay = false;
        }

        if (toPlay)
        {
            // 根据用户输入改变棋盘状态
            board[moveIndex - 1] = board[moveIndex - 1] == "O" ? "X" : "O";

            if (lastMove == moveIndex)
            {
                equalToLastMove = true;
            }
            else
            {
                equalToLastMove = false;
                lastMove = moveIndex;
            }

            do
            {
                moveIndex = equalToLastMove
                    ? GetMoveIndexWhenEqualeLastMove(moveIndex, gameEntropyRate)
                    : GetMoveIndex(moveIndex, gameEntropyRate);

                board[moveIndex] = board[moveIndex] == "O" ? "X" : "O";
            }
            while (lastMove == moveIndex && board[moveIndex] == "X");

            // 打印游戏棋盘
            PrintGameBoard(board);

            // 计算游戏总数
            foreach (var item in board)
            {
                if (item == "O")
                {
                    gameSum++;
                }
            }
        }
    }
    while (stepsCount < 12 && gameSum < 10);

    if (toPlay)
    {
        // 打印游戏结果
        PrintGameResult(gameSum, stepsCount);

        Write("DO YOU WANT TO TRY ANOTHER PUZZLE ");

        var toContinue = Console.ReadLine();

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

// 创建指定数量的空格字符串
string Tab(int pos) => new(' ', pos);

// 生成随机数
double Rnd() => new Random().NextDouble();

// 获取移动位置
int GetMoveIndex(int moveIndex, double gameEntropyRate)
{
    double rate = Math.Tan(gameEntropyRate + moveIndex / gameEntropyRate - moveIndex) - Math.Sin(gameEntropyRate / moveIndex) + 336 * Math.Sin(8 * moveIndex);
    return Convert.ToInt32(Math.Floor(10 * (rate - Math.Floor(rate))));
}

// 获取移动位置（当与上一步相同）
int GetMoveIndexWhenEqualeLastMove(int moveIndex, double gameEntropyRate)
{
    double rate = 0.592 * (1 / Math.Tan(gameEntropyRate / moveIndex + gameEntropyRate)) / Math.Sin(moveIndex * 2 + gameEntropyRate) - Math.Cos(moveIndex);
    return Convert.ToInt32(Math.Floor(10 * (rate - Math.Floor(rate))));
}

// 打印新的游戏棋盘
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
    if (gameSum == 10)
    {
        Print($"VERY GOOD.  YOU GUESSED IT IN ONLY {stepsCount} GUESSES.");
    }
    else
    {
        Print($"TRY HARDER NEXT TIME.  IT TOOK YOU {stepsCount} GUESSES.");
    }
}

// 打印游戏信息
void PrintGameInfo()
{
    Print(Tab(32) + "FLIPFLOP");
    Print(Tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
    Print();
    Print("THE OBJECT OF THIS PUZZLE IS TO CHANGE THIS:");
    Print();

    Print("X X X X X X X X X X");
    Print();
    Print("TO THIS:");
    Print();
    Print("O O O O O O O O O O");
    Print();

    Print("BY TYPING THE NUMBER CORRESPONDING TO THE POSITION OF THE");
    Print("LETTER ON SOME NUMBERS, ONE POSITION WILL CHANGE, ON");
    Print("OTHERS, TWO WILL CHANGE.  TO RESET LINE TO ALL X'S, TYPE 0");
    Print("(ZERO) AND TO START OVER IN THE MIDDLE OF A GAME, TYPE ");
    Print("11 (ELEVEN).");
}

```