# BasicComputerGames源码解析 83

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Tic-Tac-Toe

The game of tic-tac-toe hardly needs any introduction. In this one, you play versus the computer. Moves are entered by number:
```
1   2   3

4   5   6

7   8   9
```

If you make any bad moves, the computer will win; if the computer makes a bad move, you can win; otherwise, the game ends in a tie.

A second version of the game is included which prints out the board after each move. This is ideally suited to a CRT terminal, particularly if you modify it to not print out a new board after each move, but rather use the cursor to make the move.

The first program was written by Tom Koos while a student researcher at the Oregon Museum of Science and Industry; it was extensively modified by Steve North of Creative Computing. The author of the second game is Curt Flick of Akron, Ohio.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=171)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=186)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

(please note any difficulties or challenges in porting here)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


# `89_Tic-Tac-Toe/csharp/tictactoe1/Program.cs`

这段代码是一个用于在控制台输出文本的 PowerShell 脚本。它通过在控制台前添加指定的字符数（如 30 个空格）并在文本前面输出指定的字符，来创建一个带有空格的字符串。

具体来说，这段代码的作用是：

1. 在屏幕上打印 "TIC TAC TOE" 字符串，带有 30 个空格。
2. 在屏幕上打印 "CREATIVE COMPUTING MORRISTOWN NEW JERSEY" 字符串，带有 15 个空格。
3. 在屏幕上打印三个空白行。
4. 在屏幕上打印 "THE GAME BOARD IS NUMBERED:\n1  2  3\n8  9  4\n7  6  5"。
5. 在控制台启动一个游戏：X 玩家和 O 玩家轮流在一个 4x4 的游戏板上玩井字棋。


```
﻿// See https://aka.ms/new-console-template for more information
// Print text on the screen with 30 spaces before text
Console.WriteLine("TIC TAC TOE".PadLeft(30));
// Print text on screen with 15 spaces before text
Console.WriteLine("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY".PadLeft(15));
// Print three blank lines on screen
Console.WriteLine("\n\n\n");
// THIS PROGRAM PLAYS TIC TAC TOE
// THE MACHINE GOES FIRST
Console.WriteLine("THE GAME BOARD IS NUMBERED:\n");
Console.WriteLine("1  2  3");
Console.WriteLine("8  9  4");
Console.WriteLine("7  6  5");

// Main program
```

这段代码是一个while循环，会一直无限循环下去直到被强制结束。在每次循环中，会定义4个整型变量a、b、c、d和e，以及4个整型变量p、q、r和s。其中a初始化为9，其他变量均未被初始化。

接下来，会调用一个名为computerMoves的函数，该函数会接收a、b、c、d和e这5个参数，并返回计算机玩家的下一步移动。然后，会分别调用move函数，接收移动步数p和玩家的下一步移动一步q，并尝试判断移动步数q是否等于玩家移动步数b+4，如果是，则执行以下操作：

1. 如果当前移动步数p是偶数，则执行以下操作：
  1. 调用computerMoves函数，并传递a、b、c、d和e这5个参数，以及移动步数p和玩家的下一步移动一步q。
  2. 如果第一步移动步数q是玩家的下一步移动步数，则执行以下操作：
   2. 读取玩家的下一步移动一步s，并尝试判断s是否等于移动步数q+4，如果是，则执行以下操作：
   3. 如果第一步移动步数s是玩家移动步数，则显示游戏胜利的信息。
   4. 如果第一步移动步数s不是玩家移动步数，则执行以下操作：
   5. 如果当前移动步数p是奇数，则执行以下操作：
   6. 如果第一步移动步数q是玩家的下一步移动步数，则执行以下操作：
   7. 如果第一步移动步数s不是玩家的下一步移动步数，则执行以下操作：
   8. 如果当前移动步数p是奇数，则执行以下操作：
   9. 如果第一步移动步数q是玩家的下一步移动步数，则执行以下操作：
   10. 如果第一步移动步数s不是玩家的下一步移动步数，则执行以下操作：
   11. 如果玩家胜利，则显示游戏胜利的信息，并结束游戏。

最后，在每次循环结束后，会输出一次结果，以供观察。


```
while(true) {
	int a, b, c, d, e;
	int p, q, r, s;
	a = 9;
	Console.WriteLine("\n\n");
	computerMoves(a);
	p = readYourMove();
	b = move(p + 1);
	computerMoves(b);
	q = readYourMove();
	if (q == move(b + 4)) {
		c = move(b + 2);
		computerMoves(c);
		r = readYourMove();
		if (r == move(c + 4)) {
			if (p % 2 != 0) {
				d = move(c + 3);
				computerMoves(d);
				s = readYourMove();
				if (s == move(d + 4)) {
					e = move(d + 6);
					computerMoves(e);
					Console.WriteLine("THE GAME IS A DRAW.");
				} else {
					e = move(d + 4);
					computerMoves(e);
					Console.WriteLine("AND WINS ********");
				}
			} else {
				d = move(c + 7);
				computerMoves(d);
				Console.WriteLine("AND WINS ********");
			}
		} else {
			d = move(c + 4);
			computerMoves(d);
			Console.WriteLine("AND WINS ********");
		}
	} else {
		c = move(b + 4);
		computerMoves(c);
		Console.WriteLine("AND WINS ********");
	}
}

```

这段代码定义了一个名为 computerMoves 的函数，其参数 move 是一个整数，表示棋子的移动方向。函数的作用是向计算机输出移动棋子的信息。

接着定义了一个名为 readYourMove 的函数，该函数使用一个 while 循环，会不断地向用户询问一个字符串类型的输入。该函数的作用是获取用户输入一个棋子的移动棋子，并返回输入的数字。

下面是计算棋子移动的函数 move，该函数接收一个整数参数 number，计算棋子从原来的位置移动到该位置的距离，并返回这个距离。根据 number 减去 8 乘以 (number - 1) 除以 8，将 number 减少 8 步，并计算出棋子移动后的位置。

最后，在 main 函数中，首先输出了计算机移动棋子的信息，然后读取用户输入一个棋子的移动信息。


```
void computerMoves(int move) {
		Console.WriteLine("COMPUTER MOVES " + move);
}
int readYourMove() {
	while(true) {
		Console.Write("YOUR MOVE?");
		string input = Console.ReadLine();
		if (int.TryParse(input, out int number)) {
			return number;
		}
	}
}

int move(int number) {
	return number - 8 * (int)((number - 1) / 8);
}

```

# `89_Tic-Tac-Toe/csharp/tictactoe2/Program.cs`

这段代码是一个用于实现 Tic Tac Toe 游戏的程序。在这个程序中，玩家需要通过点击屏幕来选择“X”或“O”来下注，程序会按照玩家的下注结果移动棋子，并最终显示游戏结果。

具体来说，程序首先创建了一个 10 行 10 列的棋盘，以及两个字符变量“human”和“computer”，分别代表“X”和“O”。接着，程序使用一个循环来不断执行以下操作：

1. 在屏幕上打印“TIC TAC TOE”这个文本，以及“THE BOARD IS NUMBERED:”这个文本，每个文本之间有 30 个空格。
2. 在屏幕上打印“TIC TAC TOE”这个文本，以及“DO YOU WANT 'X' OR 'O'”，每个文本之间有 15 个空格。
3. 如果玩家点击了“X”，则程序会将“human”变量赋值为“X”，并将“computer”变量赋值为“O”，然后将“move”变量赋值为下注的行数，并将“board”数组中下注为“X”的位置赋值为“human”。
4. 如果玩家点击了“O”，则程序会将“human”变量赋值为“O”，并将“computer”变量赋值为“X”。
5. 程序使用一个循环来不断执行以下操作：
a. 在屏幕上打印“THE COMPUTER MOVES TO...”，以及“X GOES FIRST”这个文本，每个文本之间有 7 个空格。
b. 如果当前棋盘状态中已经有“X”的位置，则程序会将“move”变量赋值为当前“computerMove”函数返回的行数，并将“board”数组中对应位置的字符赋值为“computer”。
c. 如果当前棋盘状态中没有“X”的位置，则程序会将“move”变量赋值为“computerMove”函数返回的行数，并将“board”数组中对应位置的


```
﻿// See https://aka.ms/new-console-template for more information
char[] board = new char[10];
char human;
char computer;
int move = 0;
char result;
for(;;){
    // Print text on the screen with 30 spaces before text
    Console.WriteLine("TIC TAC TOE".PadLeft(30));
    // Print text on screen with 15 spaces before text
    Console.WriteLine("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY".PadLeft(15));
    // THIS PROGRAM PLAYS TIC TAC TOE
    Console.WriteLine("THE BOARD IS NUMBERED:");
    Console.WriteLine("1  2  3");
    Console.WriteLine("4  5  6");
    Console.WriteLine("7  8  9");
    Console.Write("\n\nDO YOU WANT 'X' OR 'O'");
    var key = Console.ReadKey();
	Console.WriteLine();
	// cleanup the board
	for(int i=0; i < 10; i++) {
		board[i]=' ';
	}
	// X GOES FIRST
    if (key.Key == ConsoleKey.X) {
		human = 'X';
		computer = 'O';
		move = readYourMove();
		board[move] = human;
		printBoard();
    } else {
		human = 'O';
		computer = 'X';
    }
	for(;;){
		Console.WriteLine("THE COMPUTER MOVES TO...");
		move = computerMove(move);
		board[move] = computer;
		result = printBoard();
		printResult(result);
		move = readYourMove();
		board[move] = human;
		result = printBoard();
		printResult(result);
	}
}

```

This is a simple game where the player has to guess a randomly generated chessboard. The player wins if they correctly guess the position of the chessman, 'TURKEY', and human wins if they correctly guess the position of the chessman, 'HUMAN'.

The `printBoard()` function is used to print the chessboard.

The game logic starts with the initialization of the chessboard and the player's turn. In each iteration, the player is presented with a single piece (either 'TURKEY' or 'HUMAN') and has to choose the correct position for that piece on the chessboard.

If the player chooses the correct position, the game continues to the next iteration. If not, the player loses a life.

The game ends when either the player has 'TURKEY' or 'HUMAN' or the game has reached the end of the chessboard.


```
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
char printBoard() {
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
	// horizontal check
	for (int i = 1; i <= 9; i += 3) {
		if (board[i] != ' ' && (board[i] == board[i+1]) && (board[i+1] == board[i+2])) {
			return board[i];
		}
	}
	// vertical check
	for (int i = 1; i <= 3; i++) {
		if (board[i] != ' ' && (board[i] == board[i+3]) && (board[i] == board[i+6])) {
			return board[i];
		}
	}
	// cross
	if (board[5] != ' ') {
		if ((board[1] == board[5] && board[9] == board[5]) || (board[3] == board[5] && board[7] == board[5])) {
			return board[5];
		}
	}
	// draw check
	for (int i = 1; i <= 9; i++) {
		if (board[i] == ' ') {
			return ' ';
		}
	}
	return '\0';
}

```

这段代码是一个简单的程序，用于读取玩家的下一步移动。程序的基本逻辑如下：

1. 初始化一个整数类型的变量 number，并将其值设置为 0。

2. 进入一个无限循环，直到程序被手动中断。

3. 在循环中，程序会先输出一段文本，并提示玩家下一步在哪里移动。

4. 如果玩家输入的是 ConsoleKey.D0，也就是向左移动，程序会输出 "THANKS FOR THE GAME。" 并退出程序。

5. 如果玩家输入的是 ConsoleKey.D1 到 ConsoleKey.D9 中的任意一个数字，程序会将 number 设为当前数字，并与该数字在 board 数组中查找是否有对应的空位置。

6. 如果找到了空位置，程序会输出 "THAT SQUARE IS OCCUPIED。" 并将 number 重新设置为 0，以便继续尝试其他可能的空位置。

7. 循环结束后，返回 number，即玩家下一步的移动将被存储在 number 中。


```
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

```

This appears to be a Java method that plays the board game 'Minesweeper'. The method takes an array of 'boardMap' which is the game board, and an array of 'stepForward2' and 'stepBackward2' which are the current and future一步可以 move the user to, respectively. It also takes an array of 'boardMap' which is the game board and an array of 'stepForward1' and 'stepBackward1' which are the current and future一步 can move the user to, respectively.

It starts by initializing all the coordinates's value to 0, and then it checks for each position on the board if it is ' ' (empty). If it is not ' ', it then checks the value of the position by getting the index and then it checks the value of the position by comparing it with ' ' (null). If the value is not ' ' it returns the stepForward2 and stepBackward2 of the position.

It also appears to be using some sort of logic that if the value is ' ' it returns the stepForward2 and stepBackward2 of the position. It also appears to be using the crossIndex which is the position of the ' ' (null) value. It then returns the crossIndex if it is not ' ' and the stepForward2 or stepBackward2 if it is ' '.

It also appears to be checking for the ' ' (null) value at the position (index+4) and if it is not ' ' it returns the stepForward2 and stepBackward2 of the position.

It also appears to be using some sort of logic that if the value is ' ' it returns the stepForward2 and stepBackward2 of the position. It also appears to be using the crossIndex which is the position of the ' ' (null) value. It then returns the stepForward2 or stepBackward2 if it is ' '.


```
int getIndex(int number) {
	return ((number - 1) % 8) + 1; //number - 8 * (int)((number - 1) / 8);
}
int computerMove(int lastMove) {
	int[] boardMap = new int[] {0, 1, 2, 3, 6, 9, 8, 7, 4, 5};
	int index = Array.IndexOf(boardMap, lastMove);
	if (lastMove == 0 || board[5] == ' '){
		return 5;
	}
	if (lastMove == 5) {
		return 1;
	}
	if (board[5] == human) {
		// check possible win
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
		// check cross
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
		// check possible win
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
		// if corner
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
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `89_Tic-Tac-Toe/java/src/Board.java`

This is a Java class that represents a chessboard. It has 9 elements in each row, representing the 9 cells of a chessboard. It also has 9 elements in each column, representing the 9 cells of a chessboard. It has a constructor that initializes the elements of the board to an array of strings, a constructor that initializes the board to a default chessboard, and a method `checkWin()` that checks if there is a winner in the game.

The class also has a method `checkDraw()` that returns true if there is a draw in the game (i.e. if the game is over without any player having won) and a method `clear()` that resets the board to its default state.

Note: This class assumes that the chessboard has 9 elements in each row and column. This can be changed by modifying the code accordingly.



```
/**
 * @author Ollie Hensman-Crook
 */
public class Board {
    private char arr[];

    public Board() {
        this.arr = new char[9];
        for (int x = 1; x <= 9; x++) {
            this.arr[x - 1] = ' ';
        }
    }


    /**
     * Place 'X' or 'O' on the board position passed
     * @param position
     * @param player
     */
    public void setArr(int position, char player) {
        if (player == 'X') {
            this.arr[position-1] = 'X';
        } else {
            this.arr[position -1] = 'O';
        }
    }

    public void printBoard() {
        System.out.format("%-3c ! %-3c ! %-3c\n----+----+----\n%-3c ! %-3c ! %-3c\n----+----+----\n%-3c ! %-3c ! %-3c\n",
        this.arr[0], this.arr[1], this.arr[2], this.arr[3], this.arr[4], this.arr[5], this.arr[6], this.arr[7], this.arr[8]
        );
    }


    /**
     * @param x
     * @return the value of the char at a given position
     */
    public char getBoardValue(int x) {
        return arr[x-1];
    }


    /**
     * Go through the board and check for win (horizontal, diagonal, vertical)
     * @param player
     * @return whether a win has occured
     */
    public boolean checkWin(char player) {
        if(this.arr[0] == player && this.arr[1] == player && this.arr[2] == player)
            return true;


        if(this.arr[3] == player && this.arr[4] == player && this.arr[5] == player)
            return true;


        if(this.arr[6] == player && this.arr[7] == player && this.arr[8] == player)
            return true;

        if(this.arr[0] == player && this.arr[4] == player && this.arr[8] == player)
            return true;

        if(this.arr[2] == player && this.arr[4] == player && this.arr[6] == player)
            return true;

        if(this.arr[0] == player && this.arr[3] == player && this.arr[6] == player)
            return true;

        if(this.arr[1] == player && this.arr[4] == player && this.arr[7] == player)
            return true;

        if(this.arr[2] == player && this.arr[5] == player && this.arr[8] == player)
            return true;

        return false;
    }
    public boolean checkDraw() {
        if(this.checkWin('X') == false && this.checkWin('O') == false) {
            if(this.getBoardValue(1) != ' ' && this.getBoardValue(2) != ' ' && this.getBoardValue(3) != ' ' && this.getBoardValue(4) != ' ' && this.getBoardValue(5) != ' ' && this.getBoardValue(6) != ' ' && this.getBoardValue(7) != ' ' && this.getBoardValue(8) != ' ' && this.getBoardValue(9) != ' ' ) {
                return true;
            }
        }

        return false;
    }
    /**
     * Reset the board
     */
    public void clear() {
        for (int x = 1; x <= 9; x++) {
            this.arr[x - 1] = ' ';
        }
    }

}

```

# `89_Tic-Tac-Toe/java/src/TicTacToe2.java`

This is a Java program that simulates a game of Connect-the-Dots. The game board is represented by a class called `GameBoard`. The `GameBoard` class has methods for checking if the game is over, checking if the player has won or the player has tied, and checking if the game is a draw.

The `main` method is used to start the game. In the `main` method, the game board is initialized and then game loop is used to simulate the game. The game loop uses a while loop and in each iteration, the game board is updated with the current state of the game.

In the game loop, the player is prompted to choose whether they want to play again or exit the game. If the player chooses to play again, the game board is cleared and the game is repeated. If the player chooses to exit the game, the game board is saved and the game is over.

If the game is over, the game board is printed and the game is considered over. If the game is a draw, the game board is printed and the game is considered a draw.

Overall, this program simulates a game of Connect-the-Dots and provides a simple way for the player to decide if they want to play again or exit the game.


```
import java.util.Scanner;
import java.util.Random;

/**
 * @author Ollie Hensman-Crook
 */
public class TicTacToe2 {
    public static void main(String[] args) {
        Board gameBoard = new Board();
        Random compChoice = new Random();
        char yourChar;
        char compChar;
        Scanner in = new Scanner(System.in);

        System.out.println("              TIC-TAC-TOE");
        System.out.println("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println("\nTHE BOARD IS NUMBERED: ");
        System.out.println(" 1  2  3\n 4  5  6\n 7  8  9\n");

        while (true) {
            // ask if the player wants to be X or O and if their input is valid set their
            // play piece as such
            System.out.println("DO YOU WANT 'X' OR 'O'");
            while (true) {
                try {
                    char input;
                    input = in.next().charAt(0);

                    if (input == 'X' || input == 'x') {
                        yourChar = 'X';
                        compChar = 'O';
                        break;
                    } else if (input == 'O' || input == 'o') {
                        yourChar = 'O';
                        compChar = 'X';
                        break;
                    } else {
                        System.out.println("THATS NOT 'X' OR 'O', TRY AGAIN");
                        in.nextLine();
                    }

                } catch (Exception e) {
                    System.out.println("THATS NOT 'X' OR 'O', TRY AGAIN");
                    in.nextLine();
                }
            }

            while (true) {
                System.out.println("WHERE DO YOU MOVE");

                // check the user can move where they want to and if so move them there
                while (true) {
                    int input;
                    try {
                        input = in.nextInt();
                        if (gameBoard.getBoardValue(input) == ' ') {
                            gameBoard.setArr(input, yourChar);
                            break;
                        } else {
                            System.out.println("INVALID INPUT, TRY AGAIN");
                        }
                        in.nextLine();
                    } catch (Exception e) {
                        System.out.println("INVALID INPUT, TRY AGAIN");
                        in.nextLine();
                    }
                }

                gameBoard.printBoard();
                System.out.println("THE COMPUTER MOVES TO");

                while (true) {
                    int position = 1 + compChoice.nextInt(9);
                    if (gameBoard.getBoardValue(position) == ' ') {
                        gameBoard.setArr(position, compChar);
                        break;
                    }
                }

                gameBoard.printBoard();

                // if there is a win print if player won or the computer won and ask if they
                // want to play again
                if (gameBoard.checkWin(yourChar)) {
                    System.out.println("YOU WIN, PLAY AGAIN? (Y/N)");
                    gameBoard.clear();
                    while (true) {
                        try {
                            char input;
                            input = in.next().charAt(0);

                            if (input == 'Y' || input == 'y') {
                                break;
                            } else if (input == 'N' || input == 'n') {
                                System.exit(0);
                            } else {
                                System.out.println("THATS NOT 'Y' OR 'N', TRY AGAIN");
                                in.nextLine();
                            }

                        } catch (Exception e) {
                            System.out.println("THATS NOT 'Y' OR 'N', TRY AGAIN");
                            in.nextLine();
                        }
                    }
                    break;
                } else if (gameBoard.checkWin(compChar)) {
                    System.out.println("YOU LOSE, PLAY AGAIN? (Y/N)");
                    gameBoard.clear();
                    while (true) {
                        try {
                            char input;
                            input = in.next().charAt(0);

                            if (input == 'Y' || input == 'y') {
                                break;
                            } else if (input == 'N' || input == 'n') {
                                System.exit(0);
                            } else {
                                System.out.println("THATS NOT 'Y' OR 'N', TRY AGAIN");
                                in.nextLine();
                            }

                        } catch (Exception e) {
                            System.out.println("THATS NOT 'Y' OR 'N', TRY AGAIN");
                            in.nextLine();
                        }
                    }
                    break;
                } else if (gameBoard.checkDraw()) {
                    System.out.println("DRAW, PLAY AGAIN? (Y/N)");
                    gameBoard.clear();
                    while (true) {
                        try {
                            char input;
                            input = in.next().charAt(0);

                            if (input == 'Y' || input == 'y') {
                                break;
                            } else if (input == 'N' || input == 'n') {
                                System.exit(0);
                            } else {
                                System.out.println("THATS NOT 'Y' OR 'N', TRY AGAIN");
                                in.nextLine();
                            }

                        } catch (Exception e) {
                            System.out.println("THATS NOT 'Y' OR 'N', TRY AGAIN");
                            in.nextLine();
                        }
                    }
                    break;
                }

            }
        }
    }
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Shells)


# `89_Tic-Tac-Toe/javascript/tictactoe1.js`

这是一个基于JavaScript的Tic-Tac-Toe游戏的实现代码。该代码将游戏的规则转换为JavaScript代码，并使用document.getElementById("output")获取一个元素，将游戏玩家输入的游戏结果输出到该元素中。

print函数的作用是将一个字符串打印到网页上。在这个例子中，当调用该函数时，将在网页上打印出"XOXO"。

input函数的作用是获取用户输入的玩家编号，并将该编号存储在变量input_str中。然后，将输入字符串打印到网页上，并在网页上清除之前的结果，以便在每次游戏重新开始时正确显示结果。当调用input函数时，将弹出一个包含"XOXO"字样的输入框，要求用户输入他们的编号。用户输入数字后，input函数将提取该数字并将其存储在input_str变量中。然后，将数字打印到网页上，并将其显示为当前游戏中的胜者。


```
// TIC TAC TOE 1
//
// Converted from BASIC to Javascript by Oscar Toledo G. (nanochess)
//

function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

function input()
{
    var input_element;
    var input_str;

    return new Promise(function (resolve) {
                       input_element = document.createElement("INPUT");

                       print("? ");
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       document.getElementById("output").appendChild(input_element);
                       input_element.focus();
                       input_str = undefined;
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      input_str = input_element.value;
                                                      document.getElementById("output").removeChild(input_element);
                                                      print(input_str);
                                                      print("\n");
                                                      resolve(input_str);
                                                      }
                                                      });
                       });
}

```

这是一个 JavaScript 函数，名为 "computer_moves"，具有以下功能：

1. 打印 "COMPUTER MOVES" 消息，并在消息后跟着计算机移动的步数。
2. 使用 Math.floor() 函数对传入的 x 值进行向下取整，然后将 x 与 8 取余数。
3. 使用字符串拼接 " "，并在拼接时使用 space-- 变量（-- space 指针）来获取字符串中当前已经有的字符数。
4. 使用 while 循环语句，在 space-- 变量还等于 0 时重复执行以下操作：
a. 拼接当前字符串中的所有空格，并将空格数量存储在拼接的字符串中。
b. 将 space-- 变量递减 1。
c. 在循环中调用 ComputerMoves() 函数，将移动步数打印在消息中。

总体而言，该函数的主要目的是在屏幕上显示计算机的移动步数。在调用 ComputerMoves() 函数时，会根据传入的 x 值执行一系列计算，并将结果打印出来。


```
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

function mf(x)
{
    return x - 8 * Math.floor((x - 1) / 8);
}

function computer_moves()
{
    print("COMPUTER MOVES " + m + "\n");
}

```

This is a program that simulates a game of Connect-the-Dot. The Connect-the-Dot is a board game where players take turns marking a dot on a grid, and the first player to connect three dots in a row wins the game.

The program starts by printing the rules of the game, then it sets the initial state of the game to the default state, where all dots are marked as "."

The program then enters a loop where it displays the player's move, waits for the player's input to make the computer's move, and then updates the game state by either connecting three dots or not connecting.

The program uses the `print()` function to display the messages that the player sees, and the `await` function to wait for the player's input before making a move.

The program also uses a function called `mf()` that returns the value of the mark on the dot that corresponds to the position passed to it. This function is not defined in the program, so it is not clear what it should do.

Overall, the program is designed to simulate the game of Connect-the-Dot, but it is not clear what the `mf()` function does.


```
var m;

// Main control section
async function main()
{
    print(tab(30) + "TIC TAC TOE\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    //
    // This program plays Tic Tac Toe
    // The machine goes first
    print("THE GAME BOARD IS NUMBERED:\n");
    print("\n");
    print("1  2  3\n");
    print("8  9  4\n");
    print("7  6  5\n");
    print("\n");
    //
    // Main program
    while (1) {
        print("\n");
        print("\n");
        a = 9;
        m = a;

        computer_moves();
        print("YOUR MOVE");
        m = parseInt(await input());

        p = m;
        b = mf(p + 1);
        m = b;

        computer_moves();
        print("YOUR MOVE");
        m = parseInt(await input());

        q = m;
        if (q != mf(b + 4)) {
            c = mf(b + 4);
            m = c;
            computer_moves();
            print("AND WINS ********\n");
            continue;
        }

        c = mf(b + 2);
        m = c;

        computer_moves();
        print("YOUR MOVE");
        m = parseInt(await input());

        r = m;
        if (r != mf(c + 4)) {
            d = mf(c + 4);
            m = d;
            computer_moves();
            print("AND WINS ********\n");
            continue;
        }

        if (p % 2 == 0) {
            d = mf(c + 7);
            m = d;
            computer_moves();
            print("AND WINS ********\n");
            continue;
        }

        d = mf(c + 3);
        m = d;

        computer_moves();
        print("YOUR MOVE");
        m = parseInt(await input());

        s = m;
        if (s != mf(d + 4)) {
            e = mf(d + 4);
            m = e;
            computer_moves();
        }
        e = mf(d + 6);
        m = e;
        computer_moves();
        print("THE GAME IS A DRAW.\n");
    }
}

```

这是经典的 "Hello, World!" 程序，用于在 Unix 和类 Unix 的操作系统中打印出 "Hello, World!" 这个字符串。

在 C 语言中，`main()` 函数是程序的入口点，也是程序的起点。当程序运行时，首先会进入 `main()` 函数，然后开始执行程序体。

上面的代码只是一个简单的程序，没有任何其他的功能或资源，它的主要目的是输出 "Hello, World!" 这个字符串。


```
main();

```

# `89_Tic-Tac-Toe/javascript/tictactoe2.js`

这段代码定义了两个函数，分别是 `print` 和 `input`。

`print` 函数的作用是在文档中创建一个 `<textarea>` 元素，然后将其内容设置为传入的 `str` 参数。最后将 `<textarea>` 元素添加到文档中的 `<output>` 元素内部，以便在页面中显示。

`input` 函数的作用是从用户那里获取一个字符串，然后将其存储在变量 `input_str` 中。该函数通过创建一个 `<input>` 元素，并将其 `type` 属性设置为 `text`，`length` 属性设置为 `50`，将该元素添加到文档中的 `<output>` 元素内部。然后，该函数监听 `keydown` 事件，当事件处理程序收到按 `<Tab>` 键时，函数会将 `input_str` 的值添加到 `print` 函数中，并在屏幕上输出 `input_str` 的值。


```
// TIC TAC TOE 2
//
// Converted from BASIC to Javascript by Oscar Toledo G. (nanochess)
//

function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

function input()
{
    var input_element;
    var input_str;

    return new Promise(function (resolve) {
                       input_element = document.createElement("INPUT");

                       print("? ");
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       document.getElementById("output").appendChild(input_element);
                       input_element.focus();
                       input_str = undefined;
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      input_str = input_element.value;
                                                      document.getElementById("output").removeChild(input_element);
                                                      print(input_str);
                                                      print("\n");
                                                      resolve(input_str);
                                                      }
                                                      });
                       });
}

```

这两段代码定义了一个名为 `tab` 的函数和一个名为 `who_win` 的函数。

`tab` 函数的作用是打印一个指定长度的空格字符串，它接受一个参数 `space`，表示要打印的空格数量。函数创建了一个字符串变量 `str`，并使用 while 循环来逐个添加空格字符，直到 `space` 变量减至 0。循环结束后，函数返回生成的字符串。

`who_win` 函数的作用是判断游戏中的胜者。它接受一个参数 `piece`，表示玩家的棋子类型。函数通过 if-else 语句检查 `piece` 的值，如果 `piece` 为 -1，则输出 "I WIN, TURKEY!!!"，如果 `piece` 为 1，则输出 "YOU BEAT ME!! GOOD GAME。函数通过 `print` 函数来输出胜者信息。


```
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

var s = [];

function who_win(piece)
{
    if (piece == -1) {
        print("I WIN, TURKEY!!!\n");
    } else if (piece == 1) {
        print("YOU BEAT ME!! GOOD GAME.\n");
    }
}

```



这是一个 Python 语言的函数，它的作用是输出一个 9x9 的棋盘，并在棋盘上根据胜者(即玩家的位置)显示字符。

具体来说，代码执行以下操作：

1. 输出一个空格，然后开始迭代。

2. 对于每个玩家(即位置)，先输出该位置的空格，然后输出该位置的字母，用 letter q 表示当前位置是皇后(QB)，用 letter p 表示当前位置是平分(PB)，用 letter s 表示当前位置是世袭(HS)，用 letter q 表示当前位置是输掉(S)，用 letter w 表示当前位置是赢了(W)，用 letter e 表示当前位置是平局(E)，用 letter n 表示当前位置是未指定(N)。

3. 如果当前位置是赢了(W)，则输出一条竖线，然后输出 "---+---+--" 表示获胜方的行列标记，最后输出玩家的名字。

4. 如果当前位置是世袭(HS)，则输出一条竖线，然后输出 "   "。

5. 如果当前位置是平分(PB)，则输出一条竖线，然后输出 "   "。

6. 如果当前位置是输掉(S)，则输出一条竖线，然后输出 "!"。

7. 如果当前位置是赢了(W)，则输出一条竖线，然后输出 "---+---+--"。

8. 如果当前位置是世袭(HS)，则输出一条竖线，然后输出 "   "。

9. 如果当前位置是平局(E)，则输出一条竖线，然后输出 "   "。

10. 如果当前位置是未指定(N)，则输出一条竖线，然后输出 "   "。

11. 如果当前位置是赢了(W)，则输出一条竖线，然后输出 "!"，表示获胜方的行列标记。

12. 如果当前位置是世袭(HS)，则输出一条竖线，然后输出 "   "。

13. 如果当前位置是平分(PB)，则输出一条竖线，然后输出 "   "。

14. 如果当前位置是输掉(S)，则输出一条竖线，然后输出 "!"。


```
function show_board()
{
    print("\n");
    for (i = 1; i <= 9; i++) {
        print(" ");
        if (s[i] == -1) {
            print(qs + " ");
        } else if (s[i] == 0) {
            print("  ");
        } else {
            print(ps + " ");
        }
        if (i == 3 || i == 6) {
            print("\n");
            print("---+---+---\n");
        } else if (i != 9) {
            print("!");
        }
    }
    print("\n");
    print("\n");
    print("\n");
    for (i = 1; i <= 7; i += 3) {
        if (s[i] && s[i] == s[i + 1] && s[i] == s[i + 2]) {
            who_win(s[i]);
            return true;
        }
    }
    for (i = 1; i <= 3; i++) {
        if (s[i] && s[i] == s[i + 3] && s[i] == s[i + 6]) {
            who_win(s[i]);
            return true;
        }
    }
    if (s[1] && s[1] == s[5] && s[1] == s[9]) {
        who_win(s[1]);
        return true;
    }
    if (s[3] && s[3] == s[5] && s[3] == s[7]) {
        who_win(s[3]);
        return true;
    }
    for (i = 1; i <= 9; i++) {
        if (s[i] == 0)
            break;
    }
    if (i > 9) {
        print("IT'S A DRAW. THANK YOU.\n");
        return true;
    }
    return false;
}

```

In this board game, the player must move their piece to an empty square on the board. The computer will respond with the square the piece can move to. The computer will also move their piece one square towards the player's piece if it is on a square that the player's piece is on. The game ends when the player's piece reaches the opposite side of the board or the computer has reached the九年级制。


```
// Main control section
async function main()
{
    print(tab(30) + "TIC-TAC-TOE\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    for (i = 1; i <= 9; i++)
        s[i] = 0;
    print("THE BOARD IS NUMBERED:\n");
    print(" 1  2  3\n");
    print(" 4  5  6\n");
    print(" 7  8  9\n");
    print("\n");
    print("\n");
    print("\n");
    print("DO YOU WANT 'X' OR 'O'");
    str = await input();
    if (str == "X") {
        ps = "X";
        qs = "O";
        first_time = true;
    } else {
        ps = "O";
        qs = "X";
        first_time = false;
    }
    while (1) {
        if (!first_time) {
            g = -1;
            h = 1;
            if (s[5] == 0) {
                s[5] = -1;
            } else if (s[5] == 1 && s[1] == 0) {
                s[1] = -1;
            } else if (s[5] != 1 && s[2] == 1 && s[1] == 0 || s[5] != 1 && s[4] == 1 && s[1] == 0) {
                s[1] = -1;
            } else if (s[5] != 1 && s[6] == 1 && s[9] == 0 || s[5] != 1 && s[8] == 1 && s[9] == 0) {
                s[9] = -1;
            } else {
                while (1) {
                    played = false;
                    if (g == 1) {
                        j = 3 * Math.floor((m - 1) / 3) + 1;
                        if (3 * Math.floor((m - 1) / 3) + 1 == m)
                            k = 1;
                        if (3 * Math.floor((m - 1) / 3) + 2 == m)
                            k = 2;
                        if (3 * Math.floor((m - 1) / 3) + 3 == m)
                            k = 3;
                    } else {
                        j = 1;
                        k = 1;
                    }
                    while (1) {
                        if (s[j] == g) {
                            if (s[j + 2] == g) {
                                if (s[j + 1] == 0) {
                                    s[j + 1] = -1;
                                    played = true;
                                    break;
                                }
                            } else {
                                if (s[j + 2] == 0 && s[j + 1] == g) {
                                    s[j + 2] = -1;
                                    played = true;
                                    break;
                                }
                            }
                        } else {
                            if (s[j] != h && s[j + 2] == g && s[j + 1] == g) {
                                s[j] = -1;
                                played = true;
                                break;
                            }
                        }
                        if (s[k] == g) {
                            if (s[k + 6] == g) {
                                if (s[k + 3] == 0) {
                                    s[k + 3] = -1;
                                    played = true;
                                    break;
                                }
                            } else {
                                if (s[k + 6] == 0 && s[k + 3] == g) {
                                    s[k + 6] = -1;
                                    played = true;
                                    break;
                                }
                            }
                        } else {
                            if (s[k] != h && s[k + 6] == g && s[k + 3] == g) {
                                s[k] = -1;
                                played = true;
                                break;
                            }
                        }
                        if (g == 1)
                            break;
                        if (j == 7 && k == 3)
                            break;
                        k++;
                        if (k > 3) {
                            k = 1;
                            j += 3;
                            if (j > 7)
                                break;
                        }
                    }
                    if (!played) {
                        if (s[5] == g) {
                            if (s[3] == g && s[7] == 0) {
                                s[7] = -1;
                                played = true;
                            } else if (s[9] == g && s[1] == 0) {
                                s[1] = -1;
                                played = true;
                            } else if (s[7] == g && s[3] == 0) {
                                s[3] = -1;
                                played = true;
                            } else if (s[9] == 0 && s[1] == g) {
                                s[9] = -1;
                                played = true;
                            }
                        }
                        if (!played) {
                            if (g == -1) {
                                g = 1;
                                h = -1;
                            }
                        }
                    }
                    if (played)
                        break;
                }
                if (!played) {
                    if (s[9] == 1 && s[3] == 0 && s[1] != 1) {
                        s[3] = -1;
                    } else {
                        for (i = 2; i <= 9; i++) {
                            if (s[i] == 0) {
                                s[i] = -1;
                                break;
                            }
                        }
                        if (i > 9) {
                            s[1] = -1;
                        }
                    }
                }
            }
            print("\n");
            print("THE COMPUTER MOVES TO...");
            if (show_board())
                break;
        }
        first_time = false;
        while (1) {
            print("\n");
            print("WHERE DO YOU MOVE");
            m = parseInt(await input());
            if (m == 0) {
                print("THANKS FOR THE GAME.\n");
                break;
            }
            if (m >= 1 && m <= 9 && s[m] == 0)
                break;
            print("THAT SQUARE IS OCCUPIED.\n");
            print("\n");
            print("\n");
        }
        g = 1;
        s[m] = 1;
        if (show_board())
            break;
    }
}

```

这道题的代码是 `main()`，这是一个程序的入口函数。在大多数程序中，`main()`函数是负责启动程序并执行程序的主要函数。

`main()`函数的函数体通常包含程序的主要操作，这些操作可能会读取用户输入、执行文件操作、网络请求等。

由于您没有提供代码，我无法具体解释 `main()`函数的作用。建议您提供更多上下文信息，以便我为您提供详细的解释。


```
main();

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Kotlin](https://kotlinlang.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Lua](https://www.lua.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Perl](https://www.perl.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


# `89_Tic-Tac-Toe/python/tictactoe2.py`

这段代码定义了两个枚举类型：OccupiedBy和Winner，分别表示电脑和玩家被占领的情况。

OccupiedBy枚举类型有三种取值，分别为COMPUTER、EMPTY和PLAYER，分别表示电脑、空闲的和玩家被占领的情况。

Winner枚举类型也有三种取值，分别为NONE、COMPUTER和PLAYER，表示赢得游戏的情况。

这两个枚举类型都使用了Python的枚举类型API，用于定义枚举变量。在程序中，可以使用这些枚举变量来表示游戏中的电脑和玩家的状态，以及胜利的情况。例如，可以使用OccupiedBy.COMPUTER来表示电脑被占领的情况，使用OccupiedBy.PLAYER来表示玩家被占领的情况，使用Winner.COMPUTER来表示电脑获胜的情况，使用Winner.PLAYER来表示玩家获胜的情况，或者使用Winner.DRAW来表示平局的情况。


```
#!/usr/bin/env python3
from enum import Enum


class OccupiedBy(Enum):
    COMPUTER = -1
    EMPTY = 0
    PLAYER = 1


class Winner(Enum):
    NONE = 0
    COMPUTER = 1
    PLAYER = 2
    DRAW = 3


```

这段代码定义了一个名为Space的枚举类型，它表示棋盘上的位置。这个枚举类型定义了九个不同的位置，分别是TOP_LEFT、TOP_CENTER、TOP_RIGHT、MID_LEFT、MID_CENTER、MID_RIGHT、BOT_LEFT、BOT_CENTER和BOT_RIGHT。

然后，定义了一个名为line_170的函数，它接受一个棋盘对象board、游戏状态game、当前行数j和当前列数k，以及四个参数g、h、j和k。

这个函数的作用是判断当前行是否在指定位置，如果当前行在指定的中间列位置，则执行相应的操作并返回。如果当前行不在指定位置，则递归执行line_118函数。line_118函数的作用是在当前棋盘上查找与指定g位置相邻的位置，并返回该位置。如果当前位置与指定g位置相邻，则递归继续查找与指定h位置相邻的位置，并返回该位置。如果当前位置既不在指定列中，也不与指定g位置相邻，则返回当前行数。


```
class Space(Enum):
    TOP_LEFT = 0
    TOP_CENTER = 1
    TOP_RIGHT = 2
    MID_LEFT = 3
    MID_CENTER = 4
    MID_RIGHT = 5
    BOT_LEFT = 6
    BOT_CENTER = 7
    BOT_RIGHT = 8


def line_170(board, g, h, j, k):
    if g == OccupiedBy.Player and board[Space.MID_CENTER] == g:
        if (
            board[Space.TOP_RIGHT] == g and board[Space.BOTTOM_LEFT] is OccupiedBy.EMPTY
        ):  # Line 171
            return Space.BOTTOM_LEFT  # Line 187
        elif (
            board[Space.BOTTOM_RIGHT] == g and board[Space.TOP_LEFT] is OccupiedBy.EMPTY
        ):  # Line 172
            return Space.TOP_LEFT  # Line 181
        elif (
            board[Space.BOTTOM_LEFT] == g and board[Space.TOP_RIGHT] is OccupiedBy.EMPTY
        ) or (
            board[Space.BOTTOM_RIGHT] is OccupiedBy.PLAYER
            and board[Space.TOP_RIGHT] is OccupiedBy.EMPTY
        ):  # Line 173 and 174
            return Space.TOP_RIGHT  # Line 183 and Line 189
        elif g is OccupiedBy.COMPUTER:
            g = OccupiedBy.PLAYER
            h = OccupiedBy.COMPUTER
            return line_118(board, g, h, j, k)


```

这段代码是一个名为`line_150`的函数，它接受一个`board`二维列表、一个`g`整数、一个`h`整数和一个`j`整数作为参数。函数的作用是检查当前细胞是否在目标区域内(`board[k] == g`或`board[k + 6] != g`或`board[k + 3] != g`)，如果是，就返回当前位置向右移动的步数(即`k + 6`)，否则，返回当前位置。

具体地，代码可以拆分为以下几行：

```python
if board[k] == g: 
   if board[k + 6] != g: 
       if (board[k + 6] == h or board[k + 6] != 0 or board[k + 6] != board[k]): 
           return -1 
       else: 
           return k + 6 
   else: 
       return -1 

if board[k + 6] != g: 
   if board[k + 6] != 0 or board[k + 3] != g: 
       return -1 
   else: 
       return k + 6 

if board[k + 3] != g: 
   return -1
```

总而言之，该函数的作用是检查给定的细胞是否在目标区域内，如果是，就返回当前位置向右移动的步数，否则，返回当前位置。


```
def line_150(board, g, h, j, k):
    if board[k] != g:  # line 150
        if (
            board[k] == h  # line 160
            or board[k + 6] != g  # line 161
            or board[k + 3] != g
        ):  # line 162
            return -1  # Goto 170
        else:
            return k + 3  # Line 163
    elif board[k + 6] != g:  # line 152
        if board[k + 6] != 0 or board[k + 3] != g:  # line 165
            return -1  # Goto 170
    elif board[k + 3]:  # line 156
        return -1

    return k + 6


```

这两段代码都定义了名为 `line_120` 的函数，但它们的作用是不同的。

第一段代码的作用是检查棋盘中的方块是否与给定的垂直和水平方向上的方块不匹配。具体而言，如果当前位置与给定的垂直或水平方向上的方块不匹配，函数将检查该位置是否有与方块 `g` 相同的方块。如果是，且当前位置、当前列的相邻位置和当前行的相邻位置均与方块 `g` 不同，则函数将跳过以下语句：

```
if board[k] != g and (board[k + 6] != g or board[k + 3] != g):
```

否则，如果 `k` 列中的方块与 `g` 不同，或者 `k` 列中的方块是空格，则函数跳过以下语句：

```
elif board[j + 2] is not g:
```

否则，如果 `j + 1` 行中的方块不是空格，则函数将返回 `line_120` 函数。

第二段代码的作用是检查给定的棋盘是否可以放置 `g` 方块。具体而言，它将调用 `line_120` 函数，并检查返回的结果是否为真。如果是，则说明可以放置 `g` 方块，否则无法放置。


```
def line_120(board, g, h, j, k):
    if board[j] != g:
        if board[j] == h or board[j + 2] != g or board[j + 1] != g:
            if board[k] != g:
                if board[k + 6] != g and (board[k + 6] != 0 or board[k + 3] != g):
                    # 450 IF G=1 THEN 465
                    pass
            elif board[j + 2] is not g:  # Line 122
                pass
            elif board[j + 1] is not OccupiedBy.EMPTY:
                pass


def line_118(board, g, h):
    for j in range(7):
        for k in range(3):
            return line_120(board, g, h, j, k)


```

这段代码是一个名为`think`的函数，它接受一个4x4的游戏棋盘`board`，一个表示当前移动次数`moves`，以及一个空字典`h`，表示已经想到的位置。

函数的作用是判断当前棋盘上哪个位置是否为空，如果为空则返回该位置。否则，根据移动次数尝试更新位置，使得当前位置为指定的移动目标。

对于移动次数为1、2、3的情况，分别尝试更新board[Space.TOP_CENTER]、board[Space.MID_LEFT]、board[Space.BOT_RIGHT]的位置，以便继续执行之前的移动操作。如果仍然有位置为指定的移动目标，则返回相应的空位置。


```
def think(board, g, h, moves):

    if board[Space.MID_CENTER] is OccupiedBy.EMPTY:
        return Space.MID_CENTER

    if board[Space.MID_CENTER] is OccupiedBy.PLAYER:
        if (
            board[Space.TOP_CENTER] is OccupiedBy.PLAYER
            and board[Space.TOP_LEFT] is OccupiedBy.EMPTY
            or board[Space.MID_LEFT] is OccupiedBy.PLAYER
            and board[Space.TOP_LEFT] is OccupiedBy.EMPTY
        ):
            return Space.BOT_LEFT
        elif (
            board[Space.MID_RIGHT] is OccupiedBy.PLAYER
            and board[Space.BOT_RIGHT] is OccupiedBy.EMPTY
            or board[Space.BOT_CENTER] is OccupiedBy.PLAYER
            and board[Space.BOT_RIGHT] is OccupiedBy.EMPTY
        ):
            return Space.BOT_RIGHT

    if g == OccupiedBy.PLAYER:
        j = 3 * int((moves - 1) / 3)
        if move == j + 1:  # noqa: This definitely is a bug!
            k = 1
        if move == j + 2:  # noqa: This definitely is a bug!
            k = 2
        if move == j + 3:  # noqa: This definitely is a bug!
            k = 3
        return subthink(g, h, j, k)  # noqa: This definitely is a bug!


```

This appears to be a function definition for a board game. It appears to be checkin


```
def render_board(board, space_mapping):
    vertical_divider = "!"
    horizontal_divider = "---+---+---"
    lines = []
    lines.append(vertical_divider.join(space_mapping[space] for space in board[0:3]))
    lines.append(horizontal_divider)
    lines.append(vertical_divider.join(space_mapping[space] for space in board[3:6]))
    lines.append(horizontal_divider)
    lines.append(vertical_divider.join(space_mapping[space] for space in board[6:9]))
    return "\n".join(lines)


def determine_winner(board, g):
    # Check for matching horizontal lines
    for i in range(Space.TOP_LEFT.value, Space.BOT_LEFT.value + 1, 3):  # Line 1095
        if board[i] != board[i + 1] or board[i] != board[i + 2]:  # Lines 1100 and 1105
            continue  # First third of Line 1115
        elif board[i] == OccupiedBy.COMPUTER:  #
            return Winner.COMPUTER
        elif board[i] == OccupiedBy.PLAYER:
            return Winner.PLAYER

    # Check for matching vertical lines
    for i in range(
        Space.TOP_LEFT.value, Space.TOP_RIGHT.value + 1, 1
    ):  # Second third of Line 1115
        if (
            board[i] != board[i + 3] or board[i] != board[i + 6]
        ):  # Last third of Line 1115
            continue  # First third of 1150
        elif board[i] == OccupiedBy.COMPUTER:  # Line 1135
            return Winner.COMPUTER
        elif board[i] == OccupiedBy.PLAYER:  # Line 1137
            return Winner.PLAYER

    # Check diagonals
    if any(space is OccupiedBy.EMPTY for space in board):
        if board[Space.MID_CENTER.value] != g:
            return Winner.NONE
        elif (
            board[Space.TOP_LEFT.value] == g and board[Space.BOT_RIGHT.value] == g
        ) or (board[Space.BOT_LEFT.value] == g and board[Space.TOP_RIGHT.value] == g):
            return Winner.COMPUTER if g is OccupiedBy.COMPUTER else Winner.PLAYER
        else:
            return Winner.NONE

    return Winner.DRAW


```



这两段代码一起组成了一个简单的棋盘游戏，让两个玩家轮流移动棋子，直到其中一个玩家胜利。

具体来说，第一段代码定义了一个名为 `computer_think` 的函数，它接收一个棋盘游戏板(board)作为参数，并返回棋盘上没有被占领的空白位置的索引。这个函数的作用是遍历棋盘上的所有位置，并返回没有被占领的空白位置的索引。

第二段代码定义了一个名为 `prompt_player` 的函数，它接收一个棋盘游戏板和一个玩家当前的位置作为参数，并返回玩家当前可以移动的位置。这个函数的作用是提示玩家移动棋子，并等待玩家输入一个移动位置。如果移动位置不是 0，则会判断该位置是否被占领，如果不是，则继续执行移动操作。

第三段代码是一个简单的辅助函数，用于在棋盘游戏板上打印提示信息。


```
def computer_think(board):
    empty_indices = [
        index for index, space in enumerate(board) if space is OccupiedBy.EMPTY
    ]

    return empty_indices[0]


def prompt_player(board):
    while True:
        move = int(input("\nWHERE DO YOU MOVE? "))

        if move == 0:
            return 0

        if move > 9 or board[move - 1] is not OccupiedBy.EMPTY:
            print("THAT SQUARE IS OCCUPIED.\n\n")
            continue

        return move


```

I'm sorry, but as an AI language model, I am not capable of determining the winner of the game since it depends on the player's choice of which piece to move. I can only provide the board and the space mapping, and the function `determine_winner`, which you can use in the game. If you have any questions or if there is anything else I can assist you with, please feel free to ask.


```
def main() -> None:
    print(" " * 30 + "TIC-TAC-TOE")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
    print("\n\n")

    print("THE BOARD IS NUMBERED:")
    print(" 1  2  3")
    print(" 4  5  6")
    print(" 7  8  9")
    print("\n\n")

    # Default state
    board = [OccupiedBy.EMPTY] * 9
    current_player = OccupiedBy.PLAYER
    space_mapping = {
        OccupiedBy.EMPTY: "   ",
        OccupiedBy.PLAYER: " X ",
        OccupiedBy.COMPUTER: " O ",
    }

    symbol = input("DO YOU WANT 'X' OR 'O'? ").upper()

    # If the player doesn't choose X, then assume you want O
    # and the computer goes first.
    if symbol != "X":
        space_mapping[OccupiedBy.PLAYER] = " O "
        space_mapping[OccupiedBy.COMPUTER] = " X "
        current_player = OccupiedBy.COMPUTER

    while True:
        if current_player is OccupiedBy.PLAYER:
            move = prompt_player(board)
            if move == 0:
                print("THANKS FOR THE GAME.")
                break
            board[move - 1] = current_player

        elif current_player is OccupiedBy.COMPUTER:
            print("\nTHE COMPUTER MOVES TO...")
            board[computer_think(board)] = current_player

        print(render_board(board, space_mapping))

        winner = determine_winner(board, current_player)

        if winner is not Winner.NONE:
            print(winner)
            break

        if current_player is OccupiedBy.COMPUTER:
            current_player = OccupiedBy.PLAYER
        elif current_player is OccupiedBy.PLAYER:
            current_player = OccupiedBy.COMPUTER


```

这段代码是一个if语句，它会判断当前脚本是否作为主程序运行。如果脚本作为主程序运行，则会执行if语句内部的代码。

在这个例子中，if语句内部的代码是“main()”，它将调用一个名为“main”的函数，这个函数通常包含程序的入口点。在这个特定的环境中，main函数可能会做一些设置或初始化工作，然后就可以开始执行程序了。

if语句还会检查当前脚本是否作为模块(module)运行。如果是模块，则会执行if语句内部的代码，否则就会跳过if语句并继续执行后续代码。


```
if __name__ == "__main__":
    main()

```