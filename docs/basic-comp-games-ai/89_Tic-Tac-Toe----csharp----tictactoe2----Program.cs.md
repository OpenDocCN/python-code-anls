# `basic-computer-games\89_Tic-Tac-Toe\csharp\tictactoe2\Program.cs`

```

// 创建一个长度为10的字符数组，用于表示游戏棋盘
char[] board = new char[10];
// 定义玩家和计算机的棋子
char human;
char computer;
// 初始化移动步数为0
int move = 0;
// 定义游戏结果
char result;

// 无限循环，表示游戏进行中
for(;;){
    // 在屏幕上打印文本，文本前面有30个空格
    Console.WriteLine("TIC TAC TOE".PadLeft(30));
    // 在屏幕上打印文本，文本前面有15个空格
    Console.WriteLine("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY".PadLeft(15));
    // 打印游戏规则说明
    Console.WriteLine("THE BOARD IS NUMBERED:");
    Console.WriteLine("1  2  3");
    Console.WriteLine("4  5  6");
    Console.WriteLine("7  8  9");
    Console.Write("\n\nDO YOU WANT 'X' OR 'O'");
    // 读取玩家输入的棋子选择
    var key = Console.ReadKey();
    Console.WriteLine();
    // 清空棋盘
    for(int i=0; i < 10; i++) {
        board[i]=' ';
    }
    // 如果玩家选择'X'，则玩家为'X'，计算机为'O'
    if (key.Key == ConsoleKey.X) {
        human = 'X';
        computer = 'O';
        // 读取玩家的移动
        move = readYourMove();
        // 在棋盘上标记玩家的移动
        board[move] = human;
        // 打印当前棋盘状态
        printBoard();
    } else {
        // 如果玩家选择'O'，则玩家为'O'，计算机为'X'
        human = 'O';
        computer = 'X';
    }
    // 无限循环，表示游戏进行中
    for(;;){
        // 提示计算机移动
        Console.WriteLine("THE COMPUTER MOVES TO...");
        // 计算机进行移动
        move = computerMove(move);
        // 在棋盘上标记计算机的移动
        board[move] = computer;
        // 打印当前棋盘状态
        result = printBoard();
        // 打印游戏结果
        printResult(result);
        // 读取玩家的移动
        move = readYourMove();
        // 在棋盘上标记玩家的移动
        board[move] = human;
        // 打印当前棋盘状态
        result = printBoard();
        // 打印游戏结果
        printResult(result);
    }
}

// 打印游戏结果
void printResult(int result) {
    if (result == '\0') {
        Console.WriteLine("IT'S A DRAW. THANK YOU.");
        Environment.Exit(0);
    } else if (result == computer) {
        Console.WriteLine("I WIN, TURKEY!!!");
        Environment.Exit(0);
    } else if (result == human) {
        Console.WriteLine("YOU BEAT ME!! GOOD GAME.");
        Environment.Exit(0);
    }
}

// 打印棋盘状态
char printBoard() {
    // 打印棋盘状态
    for (int i=1; i < 10; i++){
        Console.Write($" {board[i]} ");
        if (i % 3 == 0) {
            if (i < 9) {
                Console.Write("\n---+---+---\n");
            } else {
                Console.Write("\n");
            }
        } else {
            Console.Write("!");
        }
    }
    // 检查横向是否有赢家
    for (int i = 1; i <= 9; i += 3) {
        if (board[i] != ' ' && (board[i] == board[i+1]) && (board[i+1] == board[i+2])) {
            return board[i];
        }
    }
    // 检查纵向是否有赢家
    for (int i = 1; i <= 3; i++) {
        if (board[i] != ' ' && (board[i] == board[i+3]) && (board[i] == board[i+6])) {
            return board[i];
        }
    }
    // 检查对角线是否有赢家
    if (board[5] != ' ') {
        if ((board[1] == board[5] && board[9] == board[5]) || (board[3] == board[5] && board[7] == board[5])) {
            return board[5];
        }
    }
    // 检查是否为平局
    for (int i = 1; i <= 9; i++) {
        if (board[i] == ' ') {
            return ' ';
        }
    }
    return '\0';
}

// 读取玩家的移动
int readYourMove()  {
    int number = 0;
    for(;;) {
        Console.Write("\n\nWHERE DO YOU MOVE? ");
        var key = Console.ReadKey();
        Console.WriteLine();
        if (key.Key == ConsoleKey.D0) {
            Console.WriteLine("THANKS FOR THE GAME.");
            Environment.Exit(0);
        }
        if (key.Key >= ConsoleKey.D1 && key.Key <= ConsoleKey.D9) {
            number = key.Key - ConsoleKey.D0;
            if (number > 9 || board[number] != ' ') {
                Console.WriteLine("THAT SQUARE IS OCCUPIED.\n");
                continue;
            }
        }
        return number;
    }
}

// 获取索引
int getIndex(int number) {
    return ((number - 1) % 8) + 1; //number - 8 * (int)((number - 1) / 8);
}

// 计算机移动
int computerMove(int lastMove) {
    int[] boardMap = new int[] {0, 1, 2, 3, 6, 9, 8, 7, 4, 5};
    int index = Array.IndexOf(boardMap, lastMove);
    if (lastMove == 0 || board[5] == ' '){
        return 5;
    }
    // 其他情况下的计算机移动策略
    // ...
    return 0;
}

```