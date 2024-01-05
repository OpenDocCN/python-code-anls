# `89_Tic-Tac-Toe\csharp\tictactoe2\Program.cs`

```
// 创建一个长度为10的字符数组，用于表示游戏棋盘
char[] board = new char[10];
// 用于存储玩家选择的棋子类型
char human;
// 用于存储计算机选择的棋子类型
char computer;
// 用于记录当前进行的步数
int move = 0;
// 用于存储游戏结果
char result;

// 无限循环，表示游戏进行中
for(;;){
    // 在屏幕上打印文本，文本前面有30个空格
    Console.WriteLine("TIC TAC TOE".PadLeft(30));
    // 在屏幕上打印文本，文本前面有15个空格
    Console.WriteLine("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY".PadLeft(15));
    // 打印游戏规则提示
    Console.WriteLine("THIS PROGRAM PLAYS TIC TAC TOE");
    Console.WriteLine("THE BOARD IS NUMBERED:");
    Console.WriteLine("1  2  3");
    Console.WriteLine("4  5  6");
    Console.WriteLine("7  8  9");
    Console.Write("\n\nDO YOU WANT 'X' OR 'O'");
    // 读取玩家输入的选择
    var key = Console.ReadKey();
    Console.WriteLine();
    // 清空棋盘
}
	for(int i=0; i < 10; i++) {
		board[i]=' ';  // 初始化游戏棋盘，将每个位置都设置为空格
	}
	// X GOES FIRST
    if (key.Key == ConsoleKey.X) {  // 如果玩家选择 X 先走
		human = 'X';  // 玩家执 X
		computer = 'O';  // 电脑执 O
		move = readYourMove();  // 读取玩家的移动
		board[move] = human;  // 在棋盘上标记玩家的移动
		printBoard();  // 打印当前棋盘状态
    } else {  // 如果玩家选择 O 先走
		human = 'O';  // 玩家执 O
		computer = 'X';  // 电脑执 X
    }
	for(;;){  // 无限循环，直到游戏结束
		Console.WriteLine("THE COMPUTER MOVES TO...");  // 提示电脑正在移动
		move = computerMove(move);  // 电脑进行移动
		board[move] = computer;  // 在棋盘上标记电脑的移动
		result = printBoard();  // 打印当前棋盘状态
		printResult(result);  // 打印游戏结果
		move = readYourMove();  # 从用户输入中读取移动位置
		board[move] = human;  # 在游戏棋盘上标记用户的移动位置
		result = printBoard();  # 调用打印游戏棋盘的函数，并将结果保存在变量中
		printResult(result);  # 调用打印游戏结果的函数，并传入结果参数
	}
}

void printResult(int result) {  # 定义打印游戏结果的函数，接受一个结果参数
	if (result == '\0') {  # 如果结果为空
		Console.WriteLine("IT'S A DRAW. THANK YOU.");  # 打印平局信息
		Environment.Exit(0);  # 退出程序
	} else if (result == computer) {  # 如果结果为计算机获胜
		Console.WriteLine("I WIN, TURKEY!!!");  # 打印计算机获胜信息
		Environment.Exit(0);  # 退出程序
	} else if (result == human) {  # 如果结果为用户获胜
		Console.WriteLine("YOU BEAT ME!! GOOD GAME.");  # 打印用户获胜信息
		Environment.Exit(0);  # 退出程序
	}
}
char printBoard() {  # 定义打印游戏棋盘的函数
	# 遍历棋盘的每个位置，打印出棋盘的当前状态
	for (int i=1; i < 10; i++){
		Console.Write($" {board[i]} ");  # 打印出当前位置的棋子状态
		if (i % 3 == 0) {  # 如果当前位置是每行的最后一个位置
			if (i < 9) {  # 如果不是最后一行
				Console.Write("\n---+---+---\n");  # 打印分隔线
			} else {
				Console.Write("\n");  # 打印换行
			}
		} else {
			Console.Write("!");  # 如果不是每行的最后一个位置，打印感叹号
		}
	}
	# 水平检查
	for (int i = 1; i <= 9; i += 3) {  # 从第一列开始，每次增加3，进行水平检查
		if (board[i] != ' ' && (board[i] == board[i+1]) && (board[i+1] == board[i+2])) {  # 如果一行中三个位置的棋子状态相同且不为空
			return board[i];  # 返回该棋子状态
		}
	}
	# 垂直检查
	for (int i = 1; i <= 3; i++) {  # 从第一行开始，每次增加1，进行垂直检查
```
```python
		// vertical check
		for (int j = i; j <= i + 6; j += 3) {  # 从当前列开始，每次增加3，进行垂直检查
			if (board[j] != ' ' && (board[j] == board[j+3]) && (board[j+3] == board[j+6])) {  # 如果一列中三个位置的棋子状态相同且不为空
				return board[j];  # 返回该棋子状态
			}
		}
	}
	// 对角线检查
	if ((board[1] != ' ' && board[1] == board[5] && board[5] == board[9]) ||  # 如果左上到右下对角线上三个位置的棋子状态相同且不为空
		(board[3] != ' ' && board[3] == board[5] && board[5] == board[7])) {  # 或者右上到左下对角线上三个位置的棋子状态相同且不为空
		return board[5];  # 返回该棋子状态
	}
	// 如果没有获胜者，返回空字符
	return ' ';
}
		// 检查垂直方向是否有相同的标记
		if (board[i] != ' ' && (board[i] == board[i+3]) && (board[i] == board[i+6])) {
			return board[i]; // 如果有相同的标记，返回该标记
		}
	}
	// 检查对角线方向是否有相同的标记
	if (board[5] != ' ') {
		if ((board[1] == board[5] && board[9] == board[5]) || (board[3] == board[5] && board[7] == board[5])) {
			return board[5]; // 如果有相同的标记，返回该标记
		}
	}
	// 检查是否为平局
	for (int i = 1; i <= 9; i++) {
		if (board[i] == ' ') {
			return ' '; // 如果还有空格，表示游戏未结束，返回空格
		}
	}
	return '\0'; // 如果以上情况都不符合，返回空字符表示游戏继续
}

int readYourMove()  {
    # 初始化一个整数变量 number
    number = 0
    # 无限循环，直到遇到 break 或 return
    while True:
        # 提示用户输入移动位置
        Console.Write("\n\nWHERE DO YOU MOVE? ")
        # 读取用户输入的按键
        var key = Console.ReadKey()
        # 换行
        Console.WriteLine()
        # 如果用户按下 0 键，退出游戏
        if (key.Key == ConsoleKey.D0):
            Console.WriteLine("THANKS FOR THE GAME.")
            Environment.Exit(0)
        # 如果用户按下 1-9 键
        if (key.Key >= ConsoleKey.D1 && key.Key <= ConsoleKey.D9):
            # 将按键转换为数字
            number = key.Key - ConsoleKey.D0
            # 如果数字超出范围或者对应位置已经有棋子，则提示并继续循环
            if (number > 9 || board[number] != ' '):
                Console.WriteLine("THAT SQUARE IS OCCUPIED.\n")
                continue
        # 返回用户选择的位置
        return number
int getIndex(int number) {
	return ((number - 1) % 8) + 1; // 计算给定数字在数组中的索引位置
}
int computerMove(int lastMove) {
	int[] boardMap = new int[] {0, 1, 2, 3, 6, 9, 8, 7, 4, 5}; // 定义游戏棋盘的映射关系
	int index = Array.IndexOf(boardMap, lastMove); // 获取上一步移动的位置在棋盘映射数组中的索引
	if (lastMove == 0 || board[5] == ' '){ // 如果上一步移动是0或者棋盘中心位置为空
		return 5; // 返回棋盘中心位置
	}
	if (lastMove == 5) { // 如果上一步移动是棋盘中心位置
		return 1; // 返回棋盘左上角位置
	}
	if (board[5] == human) { // 如果棋盘中心位置被人类玩家占据
		// 检查可能的获胜位置
		if (board[1] == computer && board[2] == ' ' && board[3] == computer) { // 如果计算机在1和3位置，2位置为空
			return 2; // 返回2位置
		}
		if (board[7] == computer && board[8] == ' ' && board[9] == computer) { // 如果计算机在7和9位置，8位置为空
			return 8; // 返回8位置
		}
		// 如果第二行的第一格和第三格都是计算机的标记，而第二行的第二格是空的，返回第二行的第二格的索引
		if (board[1] == computer && board[4] == ' ' && board[7] == computer) {
			return 4;
		}
		// 如果第三行的第一格和第三格都是计算机的标记，而第三行的第二格是空的，返回第三行的第二格的索引
		if (board[3] == computer && board[6] == ' ' && board[7] == computer) {
			return 6;
		}
		// 检查对角线
		int crossIndex = boardMap[getIndex(index + 4)];
		// 如果对角线上的格子是空的，返回对角线上的格子的索引
		if (board[crossIndex] == ' ') {
			return crossIndex;
		}
		// 向前移动两步
		int stepForward2 = boardMap[getIndex(index + 2)];
		// 如果向前移动两步后的格子是空的，返回向前移动两步后的格子的索引
		if (board[stepForward2] == ' ') {
			return stepForward2;
		}
		// 向后移动两步
		int stepBackward2 = boardMap[getIndex(index + 6)];
		// 如果向后移动两步后的格子是空的，返回向后移动两步后的格子的索引
		if (board[stepBackward2] == ' ') {
			return stepBackward2;
		}
		// 向前移动一步
		int stepForward1 = boardMap[getIndex(index + 1)];
		# 如果当前位置的前进一步为空，则返回前进一步的位置
		if (board[stepForward1] == ' ') {
			return stepForward1;
		}
		# 获取当前位置的后退一步的位置
		int stepBackward1 = boardMap[getIndex(index + 7)];
		# 如果后退一步的位置为空，则返回后退一步的位置
		if (board[stepBackward1] == ' ') {
			return stepBackward1;
		}
		# 获取当前位置的前进三步的位置
		int stepForward3 = boardMap[getIndex(index + 3)];
		# 如果前进三步的位置为空，则返回前进三步的位置
		if (board[stepForward3] == ' ') {
			return stepForward3;
		}
		# 获取当前位置的后退三步的位置
		int stepBackward3 = boardMap[getIndex(index + 5)];
		# 如果后退三步的位置为空，则返回后退三步的位置
		if (board[stepBackward3] == ' ') {
			return stepBackward3;
		}
	} else {
		# 检查是否存在可能的获胜位置
		if (board[1] == computer && board[9] == ' ') {
			return 9;
		}
		// 如果计算机在位置9，且位置1为空，则返回1
		if (board[9] == computer && board[1] == ' ') {
			return 1;
		}
		// 如果计算机在位置3，且位置7为空，则返回7
		if (board[3] == computer && board[7] == ' ') {
			return 7;
		}
		// 如果计算机在位置7，且位置3为空，则返回3
		if (board[7] == computer && board[3] == ' ') {
			return 3;
		}
		// 如果是角落位置
		if (index % 2 == 1) {
			// 获取当前位置后两步的位置
			int stepForward2 = boardMap[getIndex(index + 2)];
			// 如果后两步的位置为空，则返回该位置
			if (board[stepForward2] == ' ') {
				return stepForward2;
			}
			// 获取当前位置前两步的位置
			int stepBackward2 = boardMap[getIndex(index + 6)];
			// 如果前两步的位置为空，则返回该位置
			if (board[stepBackward2] == ' ') {
				return stepBackward2;
			}
		} else {
			# 获取当前位置的下一个位置的索引
			int stepForward1 = boardMap[getIndex(index + 1)];
			# 如果下一个位置为空，则返回该位置的索引
			if (board[stepForward1] == ' ') {
				return stepForward1;
			}
			# 获取当前位置的上一个位置的索引
			int stepBackward1 = boardMap[getIndex(index + 7)];
			# 如果上一个位置为空，则返回该位置的索引
			if (board[stepBackward1] == ' ') {
				return stepBackward1;
			}
			# 获取当前位置的向右斜下方位置的索引
			int stepForward3 = boardMap[getIndex(index + 3)];
			# 如果向右斜下方位置为空，则返回该位置的索引
			if (board[stepForward3] == ' ') {
				return stepForward3;
			}
			# 获取当前位置的向左斜下方位置的索引
			int stepBackward3 = boardMap[getIndex(index + 5)];
			# 如果向左斜下方位置为空，则返回该位置的索引
			if (board[stepBackward3] == ' ') {
				return stepBackward3;
			}
			# 获取当前位置的对角位置的索引
			int crossIndex = boardMap[getIndex(index + 4)];
			# 如果对角位置为空，则返回该位置的索引
			if (board[crossIndex] == ' ') {
				return crossIndex;
			}
			# 获取当前位置向前移动两步后的索引
			int stepForward2 = boardMap[getIndex(index + 2)];
			# 如果向前移动两步后的位置为空，则返回该位置的索引
			if (board[stepForward2] == ' ') {
				return stepForward2;
			}
			# 获取当前位置向后移动两步后的索引
			int stepBackward2 = boardMap[getIndex(index + 6)];
			# 如果向后移动两步后的位置为空，则返回该位置的索引
			if (board[stepBackward2] == ' ') {
				return stepBackward2;
			}
		}
	}
	# 如果以上条件都不满足，则返回0
	return 0;
}
```