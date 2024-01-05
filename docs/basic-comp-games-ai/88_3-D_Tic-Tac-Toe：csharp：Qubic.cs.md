# `d:/src/tocomm/basic-computer-games\88_3-D_Tic-Tac-Toe\csharp\Qubic.cs`

```
    //  - AI: the AI moved here.
    private const int EMPTY = 0;
    private const int PLAYER = 1;
    private const int AI = 2;

    // The game board is represented as a 4x4x4 array of integers.
    private int[,,] board = new int[4, 4, 4];

    // The current player, either the player or the AI.
    private int currentPlayer;

    // Constructor for the Qubic class, initializes the game board and sets the current player.
    public Qubic()
    {
        // Initialize the game board with EMPTY values.
        for (int x = 0; x < 4; x++)
        {
            for (int y = 0; y < 4; y++)
            {
                for (int z = 0; z < 4; z++)
                {
                    board[x, y, z] = EMPTY;
                }
            }
        }

        // Set the current player to the player.
        currentPlayer = PLAYER;
    }

    // Method to make a move on the game board.
    public void MakeMove(int x, int y, int z)
    {
        // Check if the space is empty and within the bounds of the board.
        if (board[x, y, z] == EMPTY && x >= 0 && x < 4 && y >= 0 && y < 4 && z >= 0 && z < 4)
        {
            // Set the current player's value in the specified space.
            board[x, y, z] = currentPlayer;

            // Switch the current player to the other player.
            currentPlayer = (currentPlayer == PLAYER) ? AI : PLAYER;
        }
    }

    // Method to check if the game has been won by a player.
    public bool CheckForWin()
    {
        // Check for a win in each plane, row, and column.
        for (int i = 0; i < 4; i++)
        {
            if (CheckPlane(i) || CheckRow(i) || CheckColumn(i))
            {
                return true;
            }
        }

        // Check for a win in each diagonal.
        if (CheckDiagonals())
        {
            return true;
        }

        return false;
    }

    // Other methods for checking win conditions, determining AI moves, etc. would go here.
}
        //  - MACHINE: 机器移动到这里。
        //  - POTENTIAL: 机器在移动过程中，可能会在空格中填入潜在的移动标记，这样一来
        //      一旦它最终选择移动的位置，就会优先考虑这个空格。
        //
        // 数值允许程序通过对一行中的值求和来确定已经完成的移动。理论上，这些个别值可以是满足以下条件的任何正数：
        //
        //  - EMPTY = 0
        //  - POTENTIAL * 4 < PLAYER
        //  - PLAYER * 4 < MACHINE
        private const double PLAYER = 1.0;
        private const double MACHINE = 5.0;
        private const double POTENTIAL = 0.125;
        private const double EMPTY = 0.0;

        // 原始BASIC语言中的X变量。这是Qubic棋盘，展开成一维数组。
        private readonly double[] Board = new double[64]; // 创建一个包含64个元素的双精度浮点数数组，用于表示游戏棋盘

        // The L variable in the original BASIC. There are 76 unique winning rows
        //  in the board, so each gets an entry in RowSums. A row sum can be used
        //  to check what moves have been made to that row in the board.
        //
        // Example: if RowSums[i] == PLAYER * 4, the player won with row i!
        private readonly double[] RowSums = new double[76]; // 创建一个包含76个元素的双精度浮点数数组，用于表示棋盘中的不同获胜行的总和

        public Qubic() { } // Qubic类的构造函数

        /// <summary>
        /// Run the Qubic game.
        ///
        /// Show the title, prompt for instructions, then begin the game loop.
        /// </summary>
        public void Run() // 运行Qubic游戏的方法
        {
            Title(); // 调用Title方法，显示游戏标题
            Instructions(); // 调用Instructions方法，提示用户阅读游戏说明，然后开始游戏循环
            Loop();
        }
```
这是一个循环的结束标志，表示循环结束后执行的操作。

```
        /***********************************************************************
        /* Terminal Text/Prompts
        /**********************************************************************/
```
这是一个注释，用于说明下面的代码段是关于终端文本和提示的。

```
        #region TerminalText
```
这是一个代码区域的开始标志，用于标记下面的代码段是关于终端文本的。

```
        /// <summary>
        /// Display title and attribution.
        ///
        /// Original BASIC: 50-120
        /// </summary>
```
这是一个函数的注释，用于说明下面的代码是用来显示标题和归属信息的。同时还提供了原始BASIC代码的行号范围。

```
        private static void Title()
        {
            Console.WriteLine(
                "\n" +
                "                                 QUBIC\n\n" +
                "               CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n"
            );
```
这是一个函数，用于在控制台上显示标题和归属信息。
        }

        # 提示用户是否需要游戏说明
        # 原始BASIC代码：210-313
        def Instructions():
            print("DO YOU WANT INSTRUCTIONS? ")
            # 读取用户输入的yes或no
            yes = ReadYesNo()

            if (yes):
                print(
                    "\n" +
                    "THE GAME IS TIC-TAC-TOE IN A 4 X 4 X 4 CUBE.\n" +
                    "EACH MOVE IS INDICATED BY A 3 DIGIT NUMBER, WITH EACH\n" +
                    "DIGIT BETWEEN 1 AND 4 INCLUSIVE.  THE DIGITS INDICATE THE\n" +
                    "LEVEL, ROW, AND COLUMN, RESPECTIVELY, OF THE OCCUPIED\n"
        /// <summary>
        /// Prompt player for whether they would like to move first, or allow
        ///  the machine to make the first move.
        ///
        /// Original BASIC: 440-490
        /// </summary>
        /// <returns>true if the player wants to move first</returns>
        private static bool PlayerMovePreference()
        {
            // 提示玩家选择是否先手
            Console.WriteLine("DO YOU WANT TO MOVE FIRST (Y/N)? ");
            // 读取玩家输入
            string response = Console.ReadLine();
            // 如果玩家选择先手，则返回 true，否则返回 false
            return (response.ToUpper() == "Y");
        }
        {
            Console.Write("DO YOU WANT TO MOVE FIRST? "); // 输出提示信息，询问用户是否要先行
            var result = ReadYesNo(); // 读取用户输入的是或否
            Console.WriteLine(); // 输出空行
            return result; // 返回用户输入的结果
        }

        /// <summary>
        /// Run the Qubic program loop.
        /// </summary>
        private void Loop()
        {
            // The "retry" loop; ends if player quits or chooses not to retry
            // after game ends.
            while (true) // 进入循环，直到条件不满足
            {
                ClearBoard(); // 清空游戏棋盘
                var playerNext = PlayerMovePreference(); // 获取玩家的移动偏好

                // The "game" loop; ends if player quits, player/machine wins,
                // or game ends in draw.
                // 当游戏没有结束时，进入循环
                while (true)
                {
                    if (playerNext)
                    {
                        // Player makes a move.
                        // 玩家进行移动操作
                        var playerAction = PlayerMove();
                        // 如果玩家选择移动
                        if (playerAction == PlayerAction.Move)
                        {
                            // 切换到下一个玩家
                            playerNext = !playerNext;
                        }
                        // 如果玩家选择结束游戏
                        else
                        {
                            // 结束游戏
                            return;
                        }
                    }
                    else
                    {
                        // Check for wins, if any.
                        // 检查是否有玩家获胜
                        RefreshRowSums();
                        // 检查玩家或机器是否获胜，如果是则跳出循环
                        if (CheckPlayerWin() || CheckMachineWin())
                        {
                            break;
                        }

                        // 机器进行移动
                        var machineAction = MachineMove();
                        // 如果机器动作是移动，则切换到下一个玩家
                        if (machineAction == MachineAction.Move)
                        {
                            playerNext = !playerNext;
                        }
                        // 如果机器动作是结束游戏，则跳出循环
                        else if (machineAction == MachineAction.End)
                        {
                            break;
                        }
                        // 如果机器动作既不是移动也不是结束游戏，则抛出异常
                        else
                        {
                            throw new Exception("unreachable; machine should always move or end game in game loop");
                        }
                    }
                }

                var retry = RetryPrompt();  // 调用RetryPrompt()函数，将返回值赋给变量retry

                if (!retry)  // 如果retry为false
                {
                    return;  // 返回，结束函数
                }
            }
        }

        /// <summary>
        /// Prompt the user to try another game.
        ///
        /// Original BASIC: 1490-1560
        /// </summary>
        /// <returns>true if the user wants to play again</returns>
        private static bool RetryPrompt()  // 定义名为RetryPrompt的私有函数，返回布尔值
        {
            Console.Write("DO YOU WANT TO TRY ANOTHER GAME? ");  // 在控制台上输出提示信息
        /// <summary>
        /// 从终端读取 ZIP 文件名的内容，返回其中文件名到数据的字典
        /// </summary>
        /// <param name="fname">ZIP 文件名</param>
        /// <returns>文件名到数据的字典</returns>
        def read_zip(fname):
            # 根据 ZIP 文件名读取其二进制，封装成字节流
            bio = BytesIO(open(fname, 'rb').read())
            # 使用字节流里面内容创建 ZIP 对象
            zip = zipfile.ZipFile(bio, 'r')
            # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
            fdict = {n:zip.read(n) for n in zip.namelist()}
            # 关闭 ZIP 对象
            zip.close()
            # 返回结果字典
            return fdict
                }
                else
                {
                    Console.Write("INCORRECT ANSWER.  PLEASE TYPE 'YES' OR 'NO'. ");
                }
            }
        }
```
这部分代码是一个条件语句的结尾，如果条件不满足则输出错误信息。

```
        #endregion
```
这部分代码是用来标记代码块的结束。

```
        /***********************************************************************
        /* Player Move
        /**********************************************************************/
```
这部分代码是用来注释标记玩家移动的部分代码。

```
        #region PlayerMove
```
这部分代码是用来标记代码块的开始。

```
        /// <summary>
        /// Possible actions player has taken after ending their move. This
        ///  replaces the `GOTO` logic that allowed the player to jump out of
        ///  the game loop and quit.
        /// </summary>
```
这部分代码是一个函数的注释，解释了函数的作用和替代了之前的逻辑。
        private enum PlayerAction
        {
            /// <summary>
            /// The player ends the game prematurely.
            /// </summary>
            Quit,  // 表示玩家提前结束游戏的动作

            /// <summary>
            /// The player makes a move on the board.
            /// </summary>
            Move,  // 表示玩家在棋盘上进行移动的动作
        }

        /// <summary>
        /// Make the player's move based on their input.
        ///
        /// Original BASIC: 500-620
        /// </summary>
        /// <returns>Whether the player moved or quit the program.</returns>
        private PlayerAction PlayerMove()
        {
            // 循环直到输入有效的移动
            while (true)
            {
                // 读取玩家输入的移动
                var move = ReadMove();
                // 如果输入为1，表示玩家选择退出游戏
                if (move == 1)
                {
                    return PlayerAction.Quit;
                }
                // 如果输入为0，显示游戏棋盘
                else if (move == 0)
                {
                    ShowBoard();
                }
                // 如果输入为其他数字，清除潜在的移动，尝试将坐标转换为索引
                else
                {
                    ClearPotentialMoves();
                    if (TryCoordToIndex(move, out int moveIndex))
                    {
                        // 如果移动的位置为空，将玩家的标记放置在该位置
                        if (Board[moveIndex] == EMPTY)
                        {
                            Board[moveIndex] = PLAYER;
                            return PlayerAction.Move;  # 返回玩家的移动动作
                        }
                        else
                        {
                            Console.WriteLine("THAT SQUARE IS USED, TRY AGAIN.");  # 如果方块已被使用，则打印提示信息
                        }
                    }
                    else
                    {
                        Console.WriteLine("INCORRECT MOVE, TRY AGAIN.");  # 如果移动不正确，则打印提示信息
                    }
                }
            }
        }

        /// <summary>
        /// Read a player move from the terminal. Move can be any integer.
        ///
        /// Original BASIC: 510-520
        /// </summary>
        /// <returns>the move inputted</returns>
        private static int ReadMove()
        {
            // 输出提示信息，要求用户输入移动
            Console.Write("YOUR MOVE? ");
            // 调用 ReadInteger() 方法读取用户输入的移动
            return ReadInteger();
        }

        /// <summary>
        /// Read an integer from the terminal.
        ///
        /// Original BASIC: 520
        ///
        /// Unlike the basic, this code will not accept any string that starts
        ///  with a number; only full number strings are allowed.
        /// </summary>
        /// <returns>the integer inputted</returns>
        private static int ReadInteger()
        {
            // 循环读取用户输入，直到输入合法的整数
            while (true)
            {
                var response = Console.ReadLine() ?? " ";  // 从控制台读取用户输入，如果输入为空则赋值为空字符串

                if (int.TryParse(response, out var move))  // 尝试将用户输入转换为整数，如果成功则将结果赋值给move变量
                {
                    return move;  // 如果转换成功，则返回move变量的值作为移动位置

                }
                else
                {
                    Console.Write("!NUMBER EXPECTED - RETRY INPUT LINE--? ");  // 如果转换失败，则在控制台输出错误提示信息
                }
            }
        }

        /// <summary>
        /// Display the board to the player. Spaces taken by the player are
        ///  marked with "Y", while machine spaces are marked with "M".
        ///
        /// Original BASIC: 2550-2740
        /// </summary>
        // 显示游戏棋盘
        private void ShowBoard()
        {
            // 创建一个包含换行符的字符串构建器
            var s = new StringBuilder(new string('\n', 9));

            // 遍历棋盘的行
            for (int i = 1; i <= 4; i++)
            {
                // 遍历棋盘的列
                for (int j = 1; j <= 4; j++)
                {
                    // 在字符串构建器中添加空格，用于布局棋盘
                    s.Append(' ', 3 * (j + 1));
                    // 遍历每个小格子
                    for (int k = 1; k <= 4; k++)
                    {
                        // 计算当前格子在一维数组中的索引
                        int q = (16 * i) + (4 * j) + k - 21;
                        // 根据棋盘格子的值添加相应的字符到字符串构建器中
                        s.Append(Board[q] switch
                        {
                            // 如果格子为空或者是潜在的下一步位置，添加相应的字符
                            EMPTY or POTENTIAL => "( )      ",
                            // 如果格子是玩家的棋子，添加相应的字符
                            PLAYER => "(Y)      ",
                            // 如果格子是机器的棋子，添加相应的字符
                            MACHINE => "(M)      ",
                            // 如果格子的值不符合预期，抛出异常
                            _ => throw new Exception($"invalid space value {Board[q]}"),
                        });
                    }
                    s.Append("\n\n");
                }
                s.Append("\n\n");
            }
```
这部分代码是在循环结束后，向字符串s中添加两个换行符。

```
            Console.WriteLine(s.ToString());
        }
```
这部分代码是将字符串s的内容打印到控制台上。

```
        #endregion
```
这是一个代码段的结束标记，表示MachineMove部分的代码结束。

```
        /***********************************************************************
        /* Machine Move
        /**********************************************************************/
```
这是一个注释，用于说明下面的代码是关于机器移动的。

```
        #region MachineMove
```
这是一个代码段的开始标记，表示MachineMove部分的代码开始。

```
        /// <summary>
        /// Check all rows for a player win.
        ///
        /// A row indicates a player win if its sum = PLAYER * 4.
        ///
```
这是一个函数的注释，说明了函数的作用和判断条件。
/// Original BASIC: 720-780
/// </summary>
/// <returns>whether the player won in any row</returns>
private bool CheckPlayerWin()
{
    // 遍历每一行
    for (int row = 0; row < 76; row++)
    {
        // 如果某一行的和等于玩家的棋子乘以4，表示玩家获胜
        if (RowSums[row] == (PLAYER * 4))
        {
            // 找到玩家获胜的情况
            Console.WriteLine("YOU WIN AS FOLLOWS");
            // 显示获胜的行
            DisplayRow(row);
            return true;
        }
    }

    // 没有找到玩家获胜的情况
    return false;
}
        /// <summary>
        /// 检查所有行，看是否有一行机器可以立即移动到以赢得比赛。
        ///
        /// 一行表示如果它已经有三次机器移动，即总和 = MACHINE * 3，则玩家可以立即赢得比赛。
        ///
        /// 原始基本代码：790-920
        /// </summary>
        /// <returns></returns>
        private bool CheckMachineWin()
        {
            for (int row = 0; row < 76; row++)
            {
                if (RowSums[row] == (MACHINE * 3))
                {
                    // 找到一行可以赢得比赛！
                    for (int space = 0; space < 4; space++)
                    {
                        int move = RowsByPlane[row, space];
                        if (Board[move] == EMPTY)
                        {
                            // 如果在获胜的行中找到空的位置，则移动到那里。
                            Board[move] = MACHINE;
                            Console.WriteLine($"MACHINE MOVES TO {IndexToCoord(move)} , AND WINS AS FOLLOWS");
                            DisplayRow(row);
                            return true;
                        }
                    }
                }
            }

            // 没有可用的获胜行。
            return false;
        }

        /// <summary>
        /// 显示获胜行的坐标。
        /// </summary>
        /// <param name="row">在RowsByPlane数据中的索引</param>
        // 显示指定行的数据
        private void DisplayRow(int row)
        {
            for (int space = 0; space < 4; space++)
            {
                // 打印指定行和空格对应的坐标
                Console.Write($" {IndexToCoord(RowsByPlane[row, space])} ");
            }
            // 换行
            Console.WriteLine();
        }

        /// <summary>
        /// 机器在一次移动中可以采取的可能动作。这有助于替换原始BASIC中复杂的GOTO逻辑，该逻辑允许程序从机器的动作跳转到游戏的结束。
        /// </summary>
        private enum MachineAction
        {
            /// <summary>
            /// 机器没有采取任何动作。
            /// </summary>
            None,
            /// <summary>
            /// 机器做出了移动。
            /// </summary>
            Move,  // 表示机器做出了移动

            /// <summary>
            /// 机器要么赢了，要么认输了，要么达成了平局。
            /// </summary>
            End,  // 表示游戏结束

        }

        /// <summary>
        /// 机器决定在棋盘上移动的位置，并在适当的情况下结束游戏。
        ///
        /// 机器的人工智能尝试采取以下行动（顺序排列）：
        ///
        ///  1. 如果玩家有一行可以在下一轮获胜，就阻止那一行。
        ///  2. 如果机器可以困住玩家（创建两个不同的三步机器移动的行，只有通过一步移动无法阻止的行）
        # single player move, create such a trap.
        # If the player can create a similar trap for the machine on
        # their next move, block the space where that trap would be
        # created.
        # Find a plane in the board that is well-populated by player
        # moves, and take a space in the first such plane.
        # Find the first open corner or center and move there.
        # Find the first open space and move there.
        #
        # If none of these actions are possible, then the board is entirely
        # full, and the game results in a draw.
        #
        # Original BASIC: start at 930
        # <returns>the action the machine took</returns>
        # The actions the machine attempts to take, in order.
                BlockPlayer,  // 定义了名为BlockPlayer的动作
                MakePlayerTrap,  // 定义了名为MakePlayerTrap的动作
                BlockMachineTrap,  // 定义了名为BlockMachineTrap的动作
                MoveByPlane,  // 定义了名为MoveByPlane的动作
                MoveCornerOrCenter,  // 定义了名为MoveCornerOrCenter的动作
                MoveAnyOpenSpace,  // 定义了名为MoveAnyOpenSpace的动作
            };

            foreach (var action in actions)  // 遍历动作列表
            {
                // 尝试每个动作，如果没有发生任何事情则移动到下一个动作
                var actionResult = action();  // 执行当前动作并获取结果
                if (actionResult != MachineAction.None)  // 如果动作结果不是None
                {
                    // 不是原始BASIC语言中的部分：在每次机器移动后检查是否平局
                    if (CheckDraw())  // 检查是否平局
                    {
                        return DrawGame();  // 返回平局游戏结果
                    }
                    return actionResult;
                }
            }
```
这段代码是在检查是否有玩家已经占据了三个空格，如果是，则机器人会选择在这一行进行阻挡。

```
            // If we got here, all spaces are taken. Draw the game.
            return DrawGame();
        }
```
如果程序执行到这里，说明所有的空格都已经被占据，那么游戏就会以平局结束。

```
        /// <summary>
        /// Block a row with three spaces already taken by the player.
        ///
        /// Original BASIC: 930-1010
        /// </summary>
        /// <returns>
        /// Move if the machine blocked,
        /// None otherwise
        /// </returns>
```
这段代码是一个方法的注释，说明了这个方法的作用和原始的BASIC代码的范围。

```
        private MachineAction BlockPlayer()
        {
            for (int row = 0; row < 76; row++)
```
这段代码是一个名为BlockPlayer的私有方法，用于阻挡玩家已经占据了三个空格的行。接下来是一个for循环，用于遍历游戏棋盘的行。
            {
                // 检查当前行是否有玩家已经占据两个位置，需要阻止玩家获胜
                if (RowSums[row] == (PLAYER * 3))
                {
                    // 找到需要阻止玩家获胜的行
                    for (int space = 0; space < 4; space++)
                    {
                        // 在找到的行中找到空位置
                        if (Board[RowsByPlane[row, space]] == EMPTY)
                        {
                            // 占据剩余的空位置，阻止玩家获胜
                            Board[RowsByPlane[row, space]] = MACHINE;
                            Console.WriteLine($"NICE TRY. MACHINE MOVES TO {IndexToCoord(RowsByPlane[row, space])}");
                            return MachineAction.Move;
                        }
                    }
                }
            }

            // 没有找到需要阻止的行
            return MachineAction.None;
        }
# 创建一个陷阱，如果可能的话。如果在棋盘上移动到一个空格会导致两行有三个不同的机器空格，剩下的空格不在这两行中共享。玩家只能阻止其中一个陷阱，所以机器会赢。
# 如果无法创建玩家的陷阱，但找到了对机器移动特别有利的行，则机器会尝试移动到该行的平面边缘。
# 原始BASIC代码：1300-1480
# BASIC代码的1440/50行调用2360（MovePlaneEdge）。因为它只在找到一个标记为潜在空格的开放空间后才进入这段代码，所以它无法到达该代码的2440行，因为只有在调用该代码的行上找不到开放空间时才会到达该行。
        /// <returns>
        /// 如果创建了陷阱，则移动，
        /// 如果机器让步，则结束，
        /// 否则返回 None
        /// </returns>
        private MachineAction MakePlayerTrap()
        {
            for (int row = 0; row < 76; row++)
            {
                // 刷新行总和，因为新的 POTENTIAL 可能已经改变了它。
                var rowSum = RefreshRowSum(row);

                // 机器在这一行移动了两次，玩家在这一行没有移动。
                if (rowSum >= (MACHINE * 2) && rowSum < (MACHINE * 2) + 1)
                {
                    // 机器在这一行还没有潜在的移动。
                    if (rowSum == (MACHINE * 2))
                    {
                        for (int space = 0; space < 4; space++)
                        {
                            // 如果空格可以用来创建陷阱，则将其标记为潜在的陷阱位置
                            if (Board[RowsByPlane[row, space]] == EMPTY)
                            {
                                Board[RowsByPlane[row, space]] = POTENTIAL;
                            }
                        }
                    }
                    // 机器已经在这一行找到了潜在的移动位置，
                    // 因此可以通过另一行创建陷阱
                    else
                    {
                        return MakeOrBlockTrap(row);
                    }
                }
            }

            // 无法创建玩家的陷阱
            RefreshRowSums();
            for (int row = 0; row < 76; row++)
            {
                // 遍历76行的游戏板，寻找特别有利于机器移动的行
                // 如果一行完全填满了POTENTIAL，或者有一个MACHINE和其他是POTENTIAL，那么这行对于机器移动可能特别有利
                // 这样的行可能有助于设置陷阱机会
                if (RowSums[row] == (POTENTIAL * 4) || RowSums[row] == MACHINE + (POTENTIAL * 3))
                {
                    // 尝试在有利的行中移动到飞机边缘
                    return MovePlaneEdge(row, POTENTIAL);
                }
            }

            // 没有找到特别有利于机器的空格
            ClearPotentialMoves();
            return MachineAction.None;
        }

        /// <summary>
# 阻止玩家在下一回合可能创建的陷阱
#
# 如果没有玩家的陷阱需要阻止，但找到了对机器移动特别有利的行，则机器将尝试移动到该行的平面边缘。
#
# 原始BASIC代码：1030-1190
#
# BASIC代码的1160/1170行调用了2360（MovePlaneEdge）。与MakePlayerTrap一样，因为它只在找到标记为潜在空间的开放空间后才进入此代码，所以它无法到达该代码的2440行，因为只有在该代码调用的行上找不到开放空间时才会到达该行。
# </summary>
# <returns>
# 如果创建了陷阱，则移动，
# 如果机器让步，则结束，
# 否则返回None
# </returns>
        private MachineAction BlockMachineTrap()
        {
            for (int i = 0; i < 76; i++)
            {
                // 刷新行总和，因为新的潜在移动可能已经改变了它。
                var rowSum = RefreshRowSum(i);

                // 玩家在这一行移动了两次，而机器在这一行没有移动。
                if (rowSum >= (PLAYER * 2) && rowSum < (PLAYER * 2) + 1)
                {
                    // 机器在这一行还没有潜在的移动。
                    if (rowSum == (PLAYER * 2))
                    {
                        for (int j = 0; j < 4; j++)
                        {
                            if (Board[RowsByPlane[i, j]] == EMPTY)
                            {
                                Board[RowsByPlane[i, j]] = POTENTIAL;
                            }
// Machine has already found a potential move in this row,
// so a trap can be created with another row by the player.
// Move to block.
// 机器已经在这一行找到了一个潜在的移动位置，
// 所以玩家可以通过在另一行创建陷阱来阻止机器。
// 移动以阻止。

else
{
    return MakeOrBlockTrap(i);
}
// 否则
// 调用MakeOrBlockTrap函数，传入参数i，并返回结果

// No player traps to block found.
// 没有找到需要阻止的玩家陷阱。
RefreshRowSums();
// 刷新行总和

for (int row = 0; row < 76; row++)
{
    // A row may be particularly advantageous for the player to move
    // to at this point, indicated by a row containing all POTENTIAL
    // moves or one PLAYER and rest POTENTIAL. Such rows may aid in
    // 玩家在这一点上可能会特别有利的移动到一行，
    // 通过包含所有潜在的移动或一个玩家和其余的潜在移动来指示。
    // 这样的行可能有助于
                // 在后续陷阱的创建中检查行的总和，如果满足条件则执行下面的操作
                if (RowSums[row] == (POTENTIAL * 4) || RowSums[row] == PLAYER + (POTENTIAL * 3))
                {
                    // 尝试在有利的行中移动到飞机边缘
                    return MovePlaneEdge(row, POTENTIAL);
                }
            }

            // 没有找到特别有利于玩家的空格
            return MachineAction.None;
        }

        /// <summary>
        /// 为玩家制造陷阱或者阻止玩家在下一回合制造陷阱。
        ///
        /// 不清楚这个方法怎么可能以认输结束；似乎只有在行包含潜在移动时才会调用它。
        ///
        /// 原始BASIC代码：2230-2350
        /// </summary>
        /// <param name="row">the row containing the space to move to</param>
        /// <returns>
        /// Move if the machine moved,
        /// End if the machine conceded
        /// </returns>
        private MachineAction MakeOrBlockTrap(int row)
        {
            // 遍历指定行的四个空格
            for (int space = 0; space < 4; space++)
            {
                // 如果指定位置是潜在的空格
                if (Board[RowsByPlane[row, space]] == POTENTIAL)
                {
                    // 将指定位置标记为机器的棋子
                    Board[RowsByPlane[row, space]] = MACHINE;

                    // 如果行总和小于机器的值，表示我们正在阻止玩家的陷阱
                    if (RowSums[row] < MACHINE)
                    {
                        // 输出信息表示正在阻止玩家的陷阱
                        Console.Write("YOU FOX.  JUST IN THE NICK OF TIME, ");
                    }
                    // 如果行总和表示我们正在完成机器的陷阱
                    else
                    {
                        Console.Write("LET'S SEE YOU GET OUT OF THIS:  ");
                    }
                    // 如果条件不满足，则输出提示信息

                    Console.WriteLine($"MACHINE MOVES TO {IndexToCoord(RowsByPlane[row, space])}");
                    // 输出机器移动到的位置信息

                    return MachineAction.Move;
                    // 返回机器移动的动作
                }
            }

            // Unclear how this can be reached.
            // 不清楚如何到达这里的情况
            Console.WriteLine("MACHINE CONCEDES THIS GAME.");
            // 输出机器认输的信息
            return MachineAction.End;
            // 返回机器结束游戏的动作
        }

        /// <summary>
        /// Find a satisfactory plane on the board and move to one if that
        ///  plane's plane edges.
        /// 寻找棋盘上满意的平面，并移动到该平面的边缘
        /// A plane on the board is satisfactory if it meets the following
        ///  conditions:
        ///     1. Player has made exactly 4 moves on the plane.
        ///     2. Machine has made either 0 or one moves on the plane.
        ///  Such a plane is one that the player could likely use to form traps.
        /// This comment explains the conditions that define a satisfactory plane on the board.

        /// Original BASIC: 1830-2020
        /// This comment indicates the original version and its time frame.

        /// Line 1990 of the original basic calls 2370 (MovePlaneEdge). Only on
        ///  this call to MovePlaneEdge can line 2440 of that method be reached,
        ///  which surves to help this method iterate through the rows of a
        ///  plane.
        /// This comment explains the relationship between different lines of code and their purpose.

        /// </summary>
        /// <returns>
        /// Move if a move in a plane was found,
        /// None otherwise
        /// </returns>
        private MachineAction MoveByPlane()
        {
            // For each plane in the cube...
        /// This comment indicates the start of a loop that iterates through each plane in the cube.
            // 循环遍历飞机的18个位置
            for (int plane = 1; plane <= 18; plane++)
            {
                // 计算当前飞机位置的总和
                double planeSum = PlaneSum(plane);

                // 检查飞机总和是否满足条件
                const double P4 = PLAYER * 4;
                const double P4_M1 = (PLAYER * 4) + MACHINE;
                if (
                    (planeSum >= P4 && planeSum < P4 + 1) ||
                    (planeSum >= P4_M1 && planeSum < P4_M1 + 1)
                )
                {
                    // 尝试移动到每个飞机位置的边缘
                    // 首先，检查标记为POTENTIAL的飞机边缘
                    for (int row = (4 * plane) - 4; row < (4 * plane); row++)
                    {
                        // 移动飞机边缘，并检查移动结果
                        var moveResult = MovePlaneEdge(row, POTENTIAL);
                        if (moveResult != MachineAction.None)
                        {
                            return moveResult;
// 如果没有找到潜在的飞机边缘，就寻找一个空的边缘
for (int row = (4 * plane) - 4; row < (4 * plane); row++)
{
    // 移动飞机边缘，将其设置为空
    var moveResult = MovePlaneEdge(row, EMPTY);
    // 如果移动成功，则返回移动结果
    if (moveResult != MachineAction.None)
    {
        return moveResult;
    }
}
// 没有找到满意的带有开放飞机边缘的飞机
ClearPotentialMoves(); // 清除潜在的移动
return MachineAction.None; // 返回空操作
# Given a row, move to the first space in that row that:
#  1. is a plane edge, and
#  2. has the given value in Board
# Plane edges are any spaces on a plane with one face exposed. The AI
#  prefers to move to these spaces before others, presumably
#  because they are powerful moves: a plane edge is contained on 3-4
#  winning rows of the cube.
# Original BASIC: 2360-2490
# In the original BASIC, this code is pointed to from three different
#  locations by GOTOs:
#  - 1440/50, or MakePlayerTrap;
#  - 1160/70, or BlockMachineTrap; and
#  - 1990, or MoveByPlane.
# At line 2440, this code jumps back to line 2000, which is in
#  MoveByPlane. This makes it appear as though calling MakePlayerTrap
#  or BlockPlayerTrap in the BASIC could jump into the middle of the
        ///  MoveByPlane method; were this to happen, not all of MoveByPlane's
        ///  variables would be defined! However, the program logic prevents
        ///  this from ever occurring; see each method's description for why
        ///  this is the case.
        /// </summary>
        /// <param name="row">the row to try to move to</param>
        /// <param name="spaceValue">
        /// what value the space to move to should have in Board
        /// </param>
        /// <returns>
        /// Move if a plane edge piece in the row with the given spaceValue was
        /// found,
        /// None otherwise
        /// </returns>
        private MachineAction MovePlaneEdge(int row, double spaceValue)
        {
            // Given a row, we want to find the plane edge pieces in that row.
            // We know that each row is part of a plane, and that the first
            // and last rows of the plane are on the plane edge, while the
            // other two rows are in the middle. If we know whether a row is an
            // edge or not, we can determine if the spaceValue exists in that row
            // and if it does, we can move the plane edge piece to that space.
            // This method returns a MachineAction representing the move if a
            // plane edge piece with the given spaceValue was found, otherwise
            // it returns None.
            # 边缘或中间，我们可以确定该行中哪些空间是平面的边缘。
            #
            # 以下是立方体中一个平面的俯视图，行水平排列：
            #
            #   行 0: ( ) (1) (2) ( )
            #   行 1: (0) ( ) ( ) (3)
            #   行 2: (0) ( ) ( ) (3)
            #   行 3: ( ) (1) (2) ( )
            #
            # 平面边缘块有它们的行索引标记。上面的模式显示：
            #
            #  如果行 == 0 | 3，平面边缘空间 = [1, 2]
            #  如果行 == 1 | 2，平面边缘空间 = [0, 3]

            # 以下条件替换了以下BASIC代码（2370）：
            #
            #  I-(INT(I/4)*4)>1
// 这段代码是一个注释，用于解释在 C# 中如何实现与下面代码类似的功能
// 在上面的示例中，i 是 RowsByPlane 中的行的索引（从 1 开始）
// 这个条件根据给定的行是在平面的边缘还是中间选择不同的 a 值
int a = (row % 4) switch
{
    0 or 3 => 1,  // 行在平面的边缘
    1 or 2 => 2,  // 行在平面的中间
    _ => throw new Exception($"unreachable ({row % 4})"),  // 如果不满足以上条件，抛出异常
};

// 遍历行的平面边缘的部分
            // 遍历循环，根据不同的条件选择不同的空间范围
            // 如果 a = 1（行在边缘），则遍历 [0, 3]
            // 如果 a = 2（行在中间），则遍历 [1, 2]
            for (int space = a - 1; space <= 4 - a; space += 5 - (2 * a))
            {
                // 检查当前空间是否为指定的空值
                if (Board[RowsByPlane[row, space]] == spaceValue)
                {
                    // 找到一个可用的边缘位置！
                    // 将机器标记在该位置
                    Board[RowsByPlane[row, space]] = MACHINE;
                    // 打印机器移动的位置
                    Console.WriteLine($"MACHINE TAKES {IndexToCoord(RowsByPlane[row, space])}");
                    // 返回机器移动的动作
                    return MachineAction.Move;
                }
            }

            // 没有找到有效的边缘位置可用。
            // 返回无动作
            return MachineAction.None;
        }

        /// <summary>
        /// 在棋盘上找到第一个可用的角落或中心位置，并移动到那里。
        /// Original BASIC: 1200-1290
        /// 这是原始BASIC代码的范围
        ///
        /// This is the only place where the Z variable from the BASIC code is
        ///  used; here it is implied in the for loop.
        /// 这是BASIC代码中唯一使用Z变量的地方；在这里，它在for循环中隐含使用。
        /// </summary>
        /// <returns>
        /// Move if an open corner/center was found and moved to,
        /// None otherwise
        /// </returns>
        private MachineAction MoveCornerOrCenter()
        {
            foreach (int space in CornersAndCenters)
            {
                if (Board[space] == EMPTY)
                {
                    Board[space] = MACHINE;
                    Console.WriteLine($"MACHINE MOVES TO {IndexToCoord(space)}");
                    return MachineAction.Move;
                }
            }

            return MachineAction.None;
        }
```
这段代码是一个私有方法，用于在棋盘上找到第一个空位并移动到那里。如果找到并移动到了一个空位，就返回移动的动作；否则返回None。

```python
        /// <summary>
        /// Find the first open space in the board and move there.
        ///
        /// Original BASIC: 1720-1800
        /// </summary>
        /// <returns>
        /// Move if an open space was found and moved to,
        /// None otherwise
        /// </returns>
```
这是一个方法的注释，说明了这个方法的作用和原始的BASIC代码的范围。同时也说明了返回值的含义。

```python
        private MachineAction MoveAnyOpenSpace()
        {
            for (int row = 0; row < 64; row++)
            {
                if (Board[row] == EMPTY)
                {
```
这段代码是一个私有方法，用于在棋盘上找到第一个空位并移动到那里。它使用了一个for循环来遍历棋盘的每一行，如果找到了一个空位，就执行相应的操作。
                    Board[row] = MACHINE;  // 将机器在棋盘上的位置标记为MACHINE
                    Console.WriteLine($"MACHINE LIKES {IndexToCoord(row)}");  // 打印机器喜欢的位置
                    return MachineAction.Move;  // 返回机器的移动动作
                }
            }
            return MachineAction.None;  // 如果没有符合条件的情况，返回无动作
        }

        /// <summary>
        /// Draw the game in the event that there are no open spaces.
        ///
        /// Original BASIC: 1810-1820
        /// </summary>
        /// <returns>End</returns>
        private MachineAction DrawGame()
        {
            Console.WriteLine("THIS GAME IS A DRAW.");  // 打印游戏结束为平局
            return MachineAction.End;  // 返回游戏结束的动作
        }
        #endregion
        #endregion：结束当前代码块

        /***********************************************************************
        /* Helpers
        /**********************************************************************/
        #region Helpers
        #region Helpers：开始一个名为Helpers的代码块

        /// <summary>
        /// Attempt to transform a cube coordinate to an index into Board.
        ///
        /// A valid cube coordinate is a three-digit number, where each digit
        ///  of the number X satisfies 1 <= X <= 4.
        ///
        /// Examples:
        ///  111 -> 0
        ///  444 -> 63
        ///  232 -> 35
        ///
        /// If the coord provided is not valid, the transformation fails.
        ///
        /// <summary>：对函数进行注释说明
        # 这个方法用于将立方坐标转换为索引，本质上是一个 4 进制和 10 进制之间的转换
        # 原始的 BASIC 代码是 525-580
        # 这个方法修复了原始 BASIC 代码中的一个 bug (525-526)，原始代码只检查给定的坐标是否满足 111 <= coord <= 444。这允许了无效的坐标，比如 199 和 437，它们的个位数字超出了范围。
        # <param name="coord">立方坐标 (例如 "111", "342")</param>
        # <param name="index">转换后的输出</param>
        # <returns>
        # 如果转换成功则返回 true，否则返回 false
        # </returns>
        def TryCoordToIndex(coord, index):
            # 解析个位数字，减去 1 得到 4 进制数
            hundreds = (coord // 100) - 1
            tens = ((coord % 100) // 10) - 1
            // 计算个位数
            var ones = (coord % 10) - 1;

            // 对每个数字进行边界检查
            foreach (int digit in new int[] { hundreds, tens, ones })
            {
                if (digit < 0 || digit > 3)
                {
                    index = -1;
                    return false;
                }
            }

            // 将四进制转换为十进制
            index = (16 * hundreds) + (4 * tens) + ones;
            return true;
        }

        /// <summary>
        /// 将棋盘索引转换为有效的立方体坐标。
        ///
/// Examples:
///  0 -> 111
///  63 -> 444
///  35 -> 232
///
/// The conversion from index to coordinate is essentially a conversion
///  between base 10 and base 4.
///
/// Original BASIC: 1570-1610
/// </summary>
/// <param name="index">Board index</param>
/// <returns>the corresponding cube coordinate</returns>
private static int IndexToCoord(int index)
{
    // 检查索引是否有效
    if (index < 0 || index > 63)
    {
        // 运行时异常；此方法的所有用法都是由程序提供的索引，因此不应该出现失败
        throw new Exception($"index {index} is out of range");
    }
            }

            // 将索引转换为四进制，然后加1以获得立方坐标
            var hundreds = (index / 16) + 1;  // 计算百位
            var tens = ((index % 16) / 4) + 1;  // 计算十位
            var ones = (index % 4) + 1;  // 计算个位

            // 连接数字
            int coord = (hundreds * 100) + (tens * 10) + ones;  // 计算立方坐标
            return coord;  // 返回计算结果
        }

        /// <summary>
        /// 刷新RowSums中的值以反映任何更改。
        ///
        /// 原始BASIC代码：1640-1710
        /// </summary>
        private void RefreshRowSums()
        {
            for (var row = 0; row < 76; row++)  // 遍历76行
/// <summary>
/// Refresh a row in RowSums to reflect changes.
/// </summary>
/// <param name="row">row in RowSums to refresh</param>
/// <returns>row sum after refresh</returns>
private double RefreshRowSum(int row)
{
    // 初始化行总和为0
    double rowSum = 0;
    // 遍历该行的四个空格，将对应位置的值相加得到行总和
    for (int space = 0; space < 4; space++)
    {
        rowSum += Board[RowsByPlane[row, space]];
    }
    // 将计算得到的行总和更新到RowSums数组中对应的位置
    RowSums[row] = rowSum;
    // 返回刷新后的行总和
    return rowSum;
}
/// <summary>
/// 计算 RowSums 中一个立方体平面中空格的总和。
///
/// 原始BASIC代码：1840-1890
/// </summary>
/// <param name="plane">所需的平面</param>
/// <returns>平面中空格的总和</returns>
private double PlaneSum(int plane)
{
    double planeSum = 0; // 初始化平面空格总和为0
    for (int row = (4 * (plane - 1)); row < (4 * plane); row++) // 循环遍历所选平面的行
    {
        for (int space = 0; space < 4; space++) // 在每行中循环遍历空格
        {
            planeSum += Board[RowsByPlane[row, space]]; // 将每个空格的值加到平面空格总和中
        }
    }
    return planeSum; // 返回平面空格总和
}
        # 检查棋盘是否处于平局状态，即所有空格都已满，并且玩家和机器都没有获胜。
        # 原始的BASIC代码存在一个bug，即如果玩家先走，平局将无法被检测到。以下是一个导致这种平局的示例玩家输入序列（假设玩家先走）：
        # 114, 414, 144, 444, 122, 221, 112, 121,
        # 424, 332, 324, 421, 231, 232, 244, 311,
        # 333, 423, 331, 134, 241, 243, 143, 413,
        # 142, 212, 314, 341, 432, 412, 431, 442
        # 返回游戏是否为平局
        def CheckDraw():
            for i in range(64):
                if Board[i] != PLAYER and Board[i] != MACHINE:
                {
                    return false;  # 如果条件不满足，返回 false
                }
            }

            RefreshRowSums();  # 刷新行总和

            for (int row = 0; row < 76; row++)  # 遍历每一行
            {
                var rowSum = RowSums[row];  # 获取当前行的总和
                if (rowSum == PLAYER * 4 || rowSum == MACHINE * 4)  # 如果当前行的总和等于玩家或机器人的4倍
                {
                    return false;  # 返回 false
                }
            }


            return true;  # 如果以上条件都不满足，返回 true
        }
        # 重置棋盘上潜在移动的空格为EMPTY
        # 原始BASIC代码：2500-2540
        def ClearPotentialMoves():
            for i in range(64):
                if Board[i] == POTENTIAL:
                    Board[i] = EMPTY
        ```

        ```python
        # 重置棋盘上所有空格为EMPTY
        # 原始BASIC代码：400-420
        ```
        /// <summary>
        /// 清空棋盘上的所有格子
        /// </summary>
        private void ClearBoard()
        {
            for (var i = 0; i < 64; i++)
            {
                Board[i] = EMPTY;  // 将棋盘上的每个格子设置为空
            }
        }

        #endregion
    }
}
```