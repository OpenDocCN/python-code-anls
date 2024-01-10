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
// 无限循环，表示游戏进行
for(;;){
    // 在屏幕上打印文本，文本前面有30个空格
    Console.WriteLine("TIC TAC TOE".PadLeft(30));
    // 在屏幕上打印文本，文本前面有15个空格
    Console.WriteLine("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY".PadLeft(15));
    // 打印游戏规则提示
    Console.WriteLine("THE BOARD IS NUMBERED:");
    Console.WriteLine("1  2  3");
    Console.WriteLine("4  5  6");
    Console.WriteLine("7  8  9");
    Console.Write("\n\nDO YOU WANT 'X' OR 'O'");
    // 读取玩家选择的棋子
    var key = Console.ReadKey();
    Console.WriteLine();
    // 清空棋盘
    for(int i=0; i < 10; i++) {
        board[i]=' ';
    }
    // 玩家选择 'X' 先手
    if (key.Key == ConsoleKey.X) {
        human = 'X';
        computer = 'O';
        // 读取玩家的移动
        move = readYourMove();
        // 在棋盘上标记玩家的移动
        board[move] = human;
        // 打印棋盘
        printBoard();
    } else {
        // 玩家选择 'O' 先手
        human = 'O';
        computer = 'X';
    }
    // 无限循环，表示游戏进行
    for(;;){
        // 提示计算机移动
        Console.WriteLine("THE COMPUTER MOVES TO...");
        // 计算机进行移动
        move = computerMove(move);
        // 在棋盘上标记计算机的移动
        board[move] = computer;
        // 打印棋盘
        result = printBoard();
        // 打印游戏结果
        printResult(result);
        // 读取玩家的移动
        move = readYourMove();
        // 在棋盘上标记玩家的移动
        board[move] = human;
        // 打印棋盘
        result = printBoard();
        // 打印游戏结果
        printResult(result);
    }
}

// 打印游戏结果
void printResult(int result) {
    // 如果游戏结果为平局
    if (result == '\0') {
        Console.WriteLine("IT'S A DRAW. THANK YOU.");
        // 退出游戏
        Environment.Exit(0);
    } 
    // 如果游戏结果为计算机获胜
    else if (result == computer) {
        Console.WriteLine("I WIN, TURKEY!!!");
        // 退出游戏
        Environment.Exit(0);
    } 
    // 如果游戏结果为玩家获胜
    else if (result == human) {
        Console.WriteLine("YOU BEAT ME!! GOOD GAME.");
        // 退出游戏
        Environment.Exit(0);
    }
}
// 打印棋盘
char printBoard() {
    // 遍历棋盘的每个位置，输出对应的棋子或分隔符
    for (int i=1; i < 10; i++){
        Console.Write($" {board[i]} ");
        // 如果当前位置是每行的最后一个位置，输出分隔线
        if (i % 3 == 0) {
            // 如果不是最后一行，输出横向分隔线
            if (i < 9) {
                Console.Write("\n---+---+---\n");
            } else {
                Console.Write("\n");
            }
        } else {
            Console.Write("!");
        }
    }
    // 横向检查
    for (int i = 1; i <= 9; i += 3) {
        // 如果横向有相同的棋子，则返回该棋子
        if (board[i] != ' ' && (board[i] == board[i+1]) && (board[i+1] == board[i+2])) {
            return board[i];
        }
    }
    // 纵向检查
    for (int i = 1; i <= 3; i++) {
        // 如果纵向有相同的棋子，则返回该棋子
        if (board[i] != ' ' && (board[i] == board[i+3]) && (board[i] == board[i+6])) {
            return board[i];
        }
    }
    // 斜线检查
    if (board[5] != ' ') {
        // 如果斜线有相同的棋子，则返回该棋子
        if ((board[1] == board[5] && board[9] == board[5]) || (board[3] == board[5] && board[7] == board[5])) {
            return board[5];
        }
    }
    // 平局检查
    for (int i = 1; i <= 9; i++) {
        // 如果还有空位置，则返回空字符
        if (board[i] == ' ') {
            return ' ';
        }
    }
    // 如果没有胜者且没有空位置，则返回空字符
    return '\0';
// 读取玩家的移动输入
int readYourMove()  {
    int number = 0;
    // 无限循环，直到玩家输入有效移动
    for(;;) {
        // 提示玩家输入移动位置
        Console.Write("\n\nWHERE DO YOU MOVE? ");
        // 读取玩家输入的按键
        var key = Console.ReadKey();
        Console.WriteLine();
        // 如果玩家按下 0 键，结束游戏
        if (key.Key == ConsoleKey.D0) {
            Console.WriteLine("THANKS FOR THE GAME.");
            Environment.Exit(0);
        }
        // 如果玩家按下 1-9 键，记录移动位置
        if (key.Key >= ConsoleKey.D1 && key.Key <= ConsoleKey.D9) {
            number = key.Key - ConsoleKey.D0;
            // 如果移动位置超出范围或者已经被占据，提示玩家重新输入
            if (number > 9 || board[number] != ' ') {
                Console.WriteLine("THAT SQUARE IS OCCUPIED.\n");
                continue;
            }
        }
        // 返回玩家的移动位置
        return number;
    }
}

// 根据玩家的移动位置计算索引
int getIndex(int number) {
    return ((number - 1) % 8) + 1; // 根据移动位置计算对应的索引
}

// 计算计算机的移动位置
int computerMove(int lastMove) {
    int[] boardMap = new int[] {0, 1, 2, 3, 6, 9, 8, 7, 4, 5};
    // 获取上一次玩家的移动位置在 boardMap 中的索引
    int index = Array.IndexOf(boardMap, lastMove);
    // 如果上一次玩家移动到角落或者中心位置，计算机移动到中心位置
    if (lastMove == 0 || board[5] == ' '){
        return 5;
    }
    // 如果上一次玩家移动到中心位置，计算机移动到角落位置
    if (lastMove == 5) {
        return 1;
    }
    if (board[5] == human) {
        // 如果玩家在中间位置下棋
        // 检查可能的获胜情况
        if (board[1] == computer && board[2] == ' ' && board[3] == computer) {
            return 2;
        }
        if (board[7] == computer && board[8] == ' ' && board[9] == computer) {
            return 8;
        }
        if (board[1] == computer && board[4] == ' ' && board[7] == computer) {
            return 4;
        }
        if (board[3] == computer && board[6] == ' ' && board[7] == computer) {
            return 6;
        }
        // 检查对角线
        int crossIndex = boardMap[getIndex(index + 4)];
        if (board[crossIndex] == ' ') {
            return crossIndex;
        }
        int stepForward2 = boardMap[getIndex(index + 2)];
        if (board[stepForward2] == ' ') {
            return stepForward2;
        }
        int stepBackward2 = boardMap[getIndex(index + 6)];
        if (board[stepBackward2] == ' ') {
            return stepBackward2;
        }
        int stepForward1 = boardMap[getIndex(index + 1)];
        if (board[stepForward1] == ' ') {
            return stepForward1;
        }
        int stepBackward1 = boardMap[getIndex(index + 7)];
        if (board[stepBackward1] == ' ') {
            return stepBackward1;
        }
        int stepForward3 = boardMap[getIndex(index + 3)];
        if (board[stepForward3] == ' ') {
            return stepForward3;
        }
        int stepBackward3 = boardMap[getIndex(index + 5)];
        if (board[stepBackward3] == ' ') {
            return stepBackward3;
        }
    } else {
        // 检查可能的获胜情况
        if (board[1] == computer && board[9] == ' ') {
            return 9;
        }
        if (board[9] == computer && board[1] == ' ') {
            return 1;
        }
        if (board[3] == computer && board[7] == ' ') {
            return 7;
        }
        if (board[7] == computer && board[3] == ' ') {
            return 3;
        }
        // 如果是角落位置
        if (index % 2 == 1) {
            int stepForward2 = boardMap[getIndex(index + 2)];
            if (board[stepForward2] == ' ') {
                return stepForward2;
            }
            int stepBackward2 = boardMap[getIndex(index + 6)];
            if (board[stepBackward2] == ' ') {
                return stepBackward2;
            }
        } else {
            int stepForward1 = boardMap[getIndex(index + 1)];
            if (board[stepForward1] == ' ') {
                return stepForward1;
            }
            int stepBackward1 = boardMap[getIndex(index + 7)];
            if (board[stepBackward1] == ' ') {
                return stepBackward1;
            }
            int stepForward3 = boardMap[getIndex(index + 3)];
            if (board[stepForward3] == ' ') {
                return stepForward3;
            }
            int stepBackward3 = boardMap[getIndex(index + 5)];
            if (board[stepBackward3] == ' ') {
                return stepBackward3;
            }
            int crossIndex = boardMap[getIndex(index + 4)];
            if (board[crossIndex] == ' ') {
                return crossIndex;
            }
            int stepForward2 = boardMap[getIndex(index + 2)];
            if (board[stepForward2] == ' ') {
                return stepForward2;
            }
            int stepBackward2 = boardMap[getIndex(index + 6)];
            if (board[stepBackward2] == ' ') {
                return stepBackward2;
            }
        }
    }
    return 0;
# 闭合前面的函数定义
```