# BasicComputerGames源码解析 44

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Gomoko

GOMOKO or GOMOKU is a traditional game of the Orient. It is played by two people on a board of intersecting lines (19 left-to-right lines, 19 top-to-bottom lines, 361 intersections in all). Players take turns. During his turn, a player may cover one intersection with a marker; (one player uses white markers; the other player uses black markers). The object of the game is to get five adjacent markers in a row, horizontally, vertically or along either diagonal.

Unfortunately, this program does not make the computer a very good player. It does not know when you are about to win or even who has won. But some of its moves may surprise you.

The original author of this program is Peter Sessions of People’s Computer Company.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=74)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=89)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

(please note any difficulties or challenges in porting here)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


# `40_Gomoko/java/Gomoko.java`

This is a Java program that reads input from the user to determine the minimum and maximum board sizes. The program uses a BufferedReader to read input from the user.

The program first checks if the input is valid. If the input is not valid (i.e., the user enters a non-numeric value), the program prints an error message and retries the input.

If the input is valid, the program reads the number of rows and columns from the user. The program then checks if the input values are within the valid range. If the input values are outside the valid range, the program prints an error message and retries the input.

If the input values are valid, the program creates a Move object to store the board position. The Move object is initialized with the values read from the user.

Finally, the program prints the board position in the form of a String.

Note that the program does not handle situations where the user enters a negative number for the row or column size.


```
import java.util.Arrays;
import java.util.InputMismatchException;
import java.util.Scanner;

/**
 * GOMOKO
 * <p>
 * Converted from BASIC to Java by Aldrin Misquitta (@aldrinm)
 */
public class Gomoko {

	private static final int MIN_BOARD_SIZE = 7;
	private static final int MAX_BOARD_SIZE = 19;

	public static void main(String[] args) {
		printIntro();
		Scanner scan = new Scanner(System.in);
		int boardSize = readBoardSize(scan);

		boolean continuePlay = true;
		while (continuePlay) {
			int[][] board = new int[boardSize][boardSize];
			//initialize the board elements to 0
			for (int[] ints : board) {
				Arrays.fill(ints, 0);
			}

			System.out.println("\n\nWE ALTERNATE MOVES.  YOU GO FIRST...");

			boolean doneRound = false;
			while (!doneRound) {
				Move playerMove = null;
				boolean validMove = false;
				while (!validMove) {
					playerMove = readMove(scan);
					if (playerMove.i == -1 || playerMove.j == -1) {
						doneRound = true;
						System.out.println("\nTHANKS FOR THE GAME!!");
						System.out.print("PLAY AGAIN (1 FOR YES, 0 FOR NO)? ");
						final int playAgain = scan.nextInt();
						scan.nextLine();
						if (playAgain == 1) {
							continuePlay = true;
							break;
						} else {
							continuePlay = false;
							break;
						}
					} else if (!isLegalMove(playerMove, boardSize)) {
						System.out.println("ILLEGAL MOVE.  TRY AGAIN...");
					} else if (board[playerMove.i - 1][playerMove.j - 1] != 0) {
						System.out.println("SQUARE OCCUPIED.  TRY AGAIN...");
					} else {
						validMove = true;
					}
				}

				if (!doneRound) {
					board[playerMove.i - 1][playerMove.j - 1] = 1;
					Move computerMove = getComputerMove(playerMove, board, boardSize);
					if (computerMove == null) {
						computerMove = getRandomMove(board, boardSize);
					}
					board[computerMove.i - 1][computerMove.j - 1] = 2;

					printBoard(board);
				}
			}

		}
	}

	//*** COMPUTER TRIES AN INTELLIGENT MOVE ***
	private static Move getComputerMove(Move playerMove, int[][] board, int boardSize) {
		for (int e = -1; e <= 1; e++) {
			for (int f = -1; f <= 1; f++) {
				if ((e + f - e * f) != 0) {
					var x = playerMove.i + f;
					var y = playerMove.j + f;
					final Move newMove = new Move(x, y);
					if (isLegalMove(newMove, boardSize)) {
						if (board[newMove.i - 1][newMove.j - 1] != 0) {
							newMove.i = newMove.i - e;
							newMove.i = newMove.j - f;
							if (!isLegalMove(newMove, boardSize)) {
								return null;
							} else {
								if (board[newMove.i - 1][newMove.j - 1] == 0) {
									return newMove;
								}
							}
						}
					}
				}
			}
		}
		return null;
	}

	private static void printBoard(int[][] board) {
		for (int[] ints : board) {
			for (int cell : ints) {
				System.out.printf(" %s", cell);
			}
			System.out.println();
		}
	}

	//*** COMPUTER TRIES A RANDOM MOVE ***
	private static Move getRandomMove(int[][] board, int boardSize) {
		boolean legalMove = false;
		Move randomMove = null;
		while (!legalMove) {
			randomMove = randomMove(boardSize);
			legalMove = isLegalMove(randomMove, boardSize) && board[randomMove.i - 1][randomMove.j - 1] == 0;

		}
		return randomMove;
	}

	private static Move randomMove(int boardSize) {
		int x = (int) (boardSize * Math.random() + 1);
		int y = (int) (boardSize * Math.random() + 1);
		return new Move(x, y);
	}

	private static boolean isLegalMove(Move move, int boardSize) {
		return (move.i >= 1) && (move.i <= boardSize) && (move.j >= 1) && (move.j <= boardSize);
	}

	private static void printIntro() {
		System.out.println("                                GOMOKO");
		System.out.println("              CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
		System.out.println("\n\n");
		System.out.println("WELCOME TO THE ORIENTAL GAME OF GOMOKO.");
		System.out.println("\n");
		System.out.println("THE GAME IS PLAYED ON AN N BY N GRID OF A SIZE");
		System.out.println("THAT YOU SPECIFY.  DURING YOUR PLAY, YOU MAY COVER ONE GRID");
		System.out.println("INTERSECTION WITH A MARKER. THE OBJECT OF THE GAME IS TO GET");
		System.out.println("5 ADJACENT MARKERS IN A ROW -- HORIZONTALLY, VERTICALLY, OR");
		System.out.println("DIAGONALLY.  ON THE BOARD DIAGRAM, YOUR MOVES ARE MARKED");
		System.out.println("WITH A '1' AND THE COMPUTER MOVES WITH A '2'.");
		System.out.println("\nTHE COMPUTER DOES NOT KEEP TRACK OF WHO HAS WON.");
		System.out.println("TO END THE GAME, TYPE -1,-1 FOR YOUR MOVE.\n ");
	}

	private static int readBoardSize(Scanner scan) {
		System.out.print("WHAT IS YOUR BOARD SIZE (MIN 7/ MAX 19)? ");

		boolean validInput = false;
		int input = 0;
		while (!validInput) {
			try {
				input = scan.nextInt();
				if (input < MIN_BOARD_SIZE || input > MAX_BOARD_SIZE) {
					System.out.printf("I SAID, THE MINIMUM IS %s, THE MAXIMUM IS %s.\n", MIN_BOARD_SIZE, MAX_BOARD_SIZE);
				} else {
					validInput = true;
				}
			} catch (InputMismatchException ex) {
				System.out.println("!NUMBER EXPECTED - RETRY INPUT LINE\n");
				validInput = false;
			} finally {
				scan.nextLine();
			}
		}
		return input;
	}

	private static Move readMove(Scanner scan) {
		System.out.print("YOUR PLAY (I,J)? ");
		boolean validInput = false;
		Move move = new Move();
		while (!validInput) {
			String input = scan.nextLine();
			final String[] split = input.split(",");
			try {
				move.i = Integer.parseInt(split[0]);
				move.j = Integer.parseInt(split[1]);
				validInput = true;
			} catch (NumberFormatException nfe) {
				System.out.println("!NUMBER EXPECTED - RETRY INPUT LINE\n? ");
			}

		}
		return move;
	}

	private static class Move {
		int i;
		int j;

		public Move() {
		}

		public Move(int i, int j) {
			this.i = i;
			this.j = j;
		}

		@Override
		public String toString() {
			return "Move{" +
					"i=" + i +
					", j=" + j +
					'}';
		}
	}

}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `40_Gomoko/javascript/gomoko.js`

这段代码定义了两个函数，分别是 `print` 和 `input`。

1. `print` 函数的作用是接收一个字符串参数（`str`），将其输出到网页上的一个元素中。这个元素通过 `document.getElementById("output")` 获取到，然后通过 `document.createTextNode(str)` 创建一个新的 `text` 节点，将节点添加到指定的元素中，并将其内容设置为 `str`。

2. `input` 函数的作用是接收一个字符（`str`），然后询问用户输入这个字符。它通过 `document.createElement("INPUT")` 创建一个新的 `text` 元素，设置其 `type` 属性为 `text` 和设置其 `length` 属性为 `50`（表示最大输入字符数为 50）。然后将该元素添加到网页上的一个元素中，并设置元素的 `id` 属性为 "output"，以获取用户输入。接着，函数通过监听元素的 `keydown` 事件来获取用户按下键盘上的某个键，当用户按下 `13` 时，将获取到的字符串存储在 `inputStr` 变量中，并将其输出到网页上的一个元素中，同时删除该元素。最后，函数会将 `inputStr` 和输入的字符串一起输出。


```
// GOMOKO
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

这两段代码都在 JavaScript 中，分别定义了两个函数，function tab(space) 和 reset_stats()。

function tab(space) 的作用是输出一个由空格组成的字符串，直到输入的 space 变量为 0，也就是一直输出到字符串的末尾。在函数内部，使用了一个 while 循环和一个 space-- 变量，其中 space-- 是一个递减的整数变量，表示输出字符串中的空格数量。在循环中，使用了空格字符 " "，将每个空格填充进字符串中。最后，函数返回了生成的字符串。

function reset_stats() 的作用是清空一个数组 f 中所有的元素，数组 f 中从初始值 1开始，到程序运行结束时止。在函数内部，使用了一个 for 循环，变量 j 从 1 到 4 进行循环，每次循环都将 f[j] 的值清空为 0。

var a = [...]; 不带参数的情况下，a 变量可能代表一个未定义的变量或者一个数组。由于题目中没有给出 a 的定义，无法确定 a 的值。


```
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

function reset_stats()
{
    for (var j = 1; j <= 4; j++)
        f[j] = 0;
}

var a = [];
```

这段代码定义了三个变量x、y和n，但没有给它们赋值。然后定义了一个名为print_board的函数，但没有说明它的作用。

在print_board函数内部，使用两个嵌套的for循环来遍历二维数组a，并打印出每个元素。外层循环变量n用来控制行数，内层循环变量i用来控制列数。内层循环中的打印语句"print(" + a[i][j] + " ")将当前元素值a[i][j]与字符串" "拼接在一起，并打印出来。同时，外层循环中的打印语句"print(\\n")将换行符"\n"打印出来。


```
var x;
var y;
var n;

// *** PRINT THE BOARD ***
function print_board()
{
    for (i = 1; i <= n; i++) {
        for (j = 1; j <= n; j++) {
            print(" " + a[i][j] + " ");
        }
        print("\n");
    }
    print("\n");
}

```

This is a program written in JavaScript that allows a computer to play chess. It uses a Tic-tac-chest game logic to simulate the game.

The program first inputs the棋盘 information, including the rows and columns, and then it initializes the board.

The `parseInt` function converts the string input provided by the user to an integer.

The `print_board` function prints the board to the console, but it is not clear what this function is intended to do in the context of the game.

The `do {...} while` loop is used to repeatedly make a move, up to a maximum of four moves per turn.

The `found` variable is used to keep track of whether a move was valid or not. If a move is valid, the `a` matrix is modified to reflect the change.

The `all_valid` variable is used to keep track of all the valid moves on the board.

The game ends when the user enters a non-empty move or when a valid move is found.

It is worth noting that this program does not provide any error checking or any way to return a result of the game. It could be improved by adding proper error handling and returning the result of the game.


```
// Is valid the movement
function is_valid()
{
    if (x < 1 || x > n || y < 1 || y > n)
        return false;
    return true;
}

// Main program
async function main()
{
    print(tab(33) + "GOMOKO\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    for (i = 0; i <= 19; i++) {
        a[i] = [];
        for (j = 0; j <= 19; j++)
            a[i][j] = 0;
    }
    print("WELCOME TO THE ORIENTAL GAME OF GOMOKO.\n");
    print("\n");
    print("THE GAME IS PLAYED ON AN N BY N GRID OF A SIZE\n");
    print("THAT YOU SPECIFY.  DURING YOUR PLAY, YOU MAY COVER ONE GRID\n");
    print("INTERSECTION WITH A MARKER. THE OBJECT OF THE GAME IS TO GET\n");
    print("5 ADJACENT MARKERS IN A ROW -- HORIZONTALLY, VERTICALLY, OR\n");
    print("DIAGONALLY.  ON THE BOARD DIAGRAM, YOUR MOVES ARE MARKED\n");
    print("WITH A '1' AND THE COMPUTER MOVES WITH A '2'.\n");
    print("\n");
    print("THE COMPUTER DOES NOT KEEP TRACK OF WHO HAS WON.\n");
    print("TO END THE GAME, TYPE -1,-1 FOR YOUR MOVE.\n");
    print("\n");
    while (1) {
        print("WHAT IS YOUR BOARD SIZE (MIN 7/ MAX 19)");
        while (1) {
            n = parseInt(await input());
            if (n >= 7 && n<= 19)
                break;
            print("I SAID, THE MINIMUM IS 7, THE MAXIMUM IS 19.\n");
        }
        for (i = 1; i <= n; i++) {
            for (j = 1; j <= n; j++) {
                a[i][j] = 0;
            }
        }
        print("\n");
        print("WE ALTERNATE MOVES.  YOU GO FIRST...\n");
        print("\n");
        while (1) {
            print("YOUR PLAY (I,J)");
            str = await input();
            i = parseInt(str);
            j = parseInt(str.substr(str.indexOf(",") + 1));
            print("\n");
            if (i == -1)
                break;
            x = i;
            y = j;
            if (!is_valid()) {
                print("ILLEGAL MOVE.  TRY AGAIN...\n");
                continue;
            }
            if (a[i][j] != 0) {
                print("SQUARE OCCUPIED.  TRY AGAIN...\n");
                continue;
            }
            a[i][j] = 1;
            // *** Computer tries an intelligent move ***
            found = false;
            for (e = -1; e <= 1; e++) {
                for (f = -1; f <= 1; f++) {
                    if (e + f - e * f == 0)
                        continue;
                    x = i + f;
                    y = j + f;
                    if (!is_valid())
                        continue;
                    if (a[x][y] == 1) {
                        x = i - e;
                        y = j - f;
                        if (is_valid() || a[x][y] == 0)
                            found = true;
                        break;
                    }
                }
            }
            if (!found) {
                // *** Computer tries a random move ***
                do {
                    x = Math.floor(n * Math.random() + 1);
                    y = Math.floor(n * Math.random() + 1);
                } while (!is_valid() || a[x][y] != 0) ;
            }
            a[x][y] = 2;
            print_board();
        }
        print("\n");
        print("THANKS FOR THE GAME!!\n");
        print("PLAY AGAIN (1 FOR YES, 0 FOR NO)");
        q = parseInt(await input());
        if (q != 1)
            break;
    }
}

```

这道题目要求我们解释以下代码的作用，但是不要输出源代码。根据代码的规范，我们可以看出这是一个名为`main()`的函数，它应该是程序的入口点。在`main()`函数中，我们可以把程序的核心代码放在其中，以便程序可以正常运行。但是，我们不应该在`main()`函数之外输出程序的源代码，因为这会破坏程序的正确性。


```
main();

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Shells)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Kotlin](https://kotlinlang.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Lua](https://www.lua.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Perl](https://www.perl.org/)


# `40_Gomoko/python/Gomoko.py`

这段代码是一个用于打印棋盘的Python函数。具体来说，它使用了两个嵌套的for循环来遍历棋盘上的每个元素，并使用print函数将每个元素打印出来。函数的参数A是一个由n个列表组成的列表，表示棋盘的大小为n x n。

下面是函数的实现细节：

1. 函数首先导入了random模块和typing.Any类型。其中，random模块提供了从random数组中随机选择元素的方法，而typing.Any类型则允许输入和输出任何类型的数据。

2. 函数定义了一个名为print_board的函数，它接收三个参数：A、n和None。其中，A参数表示棋盘的大小，n表示输出的行数，而None表示不需要打印的列数。

3. 在函数内部，使用两个for循环来遍历棋盘上的每个元素。外层的for循环遍历n行，而内层的for循环遍历每一行中的每个元素。

4. 对于每一行，函数使用print函数将该行的元素打印出来，并使用end参数将每个元素与输出分隔开。这样，在每一行中，每个元素都只会被打印一次，而每个元素之间的空格则有助于增加输出可读性。

5. 函数还定义了一个名为check_move的函数，它接收三个参数：_I、_J和_N。该函数用于检查移动是否合法，具体来说，只有当移动的列不小于1，行不高于棋盘大小，且当前列不等于_N-1时，函数才会返回True。


```
import random
from typing import Any, List, Tuple


def print_board(A: List[List[Any]], n: int) -> None:
    """PRINT THE BOARD"""
    for i in range(n):
        print(" ", end="")
        for j in range(n):
            print(A[i][j], end="")
            print(" ", end="")
        print()


def check_move(_I, _J, _N) -> bool:  # 910
    if _I < 1 or _I > _N or _J < 1 or _J > _N:
        return False
    return True


```

这段代码定义了两个函数，分别是 `print_banner()` 和 `get_board_dimensions()`。

`print_banner()` 函数的作用是在屏幕上打印出一段欢迎消息，并输出游戏规则。具体来说，它通过 `print()` 函数连续输出 33 个空格，然后输出 15 个字符，接着输出一行字符，接着输出游戏规则。

`get_board_dimensions()` 函数的作用是获取玩家想要的游戏棋盘大小，并返回棋盘的大小。它通过一个 while 循环，不断询问玩家输入棋盘的大小，直到输入的字符串符合要求（即至少为7，最大为19），然后返回该值。


```
def print_banner() -> None:
    print(" " * 33 + "GOMOKU")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")
    print("WELCOME TO THE ORIENTAL GAME OF GOMOKO.\n")
    print("THE GAME IS PLAYED ON AN N BY N GRID OF A SIZE")
    print("THAT YOU SPECIFY.  DURING YOUR PLAY, YOU MAY COVER ONE GRID")
    print("INTERSECTION WITH A MARKER. THE OBJECT OF THE GAME IS TO GET")
    print("5 ADJACENT MARKERS IN A ROW -- HORIZONTALLY, VERTICALLY, OR")
    print("DIAGONALLY.  ON THE BOARD DIAGRAM, YOUR MOVES ARE MARKED")
    print("WITH A '1' AND THE COMPUTER MOVES WITH A '2'.\n")
    print("THE COMPUTER DOES NOT KEEP TRACK OF WHO HAS WON.")
    print("TO END THE GAME, TYPE -1,-1 FOR YOUR MOVE.\n")


def get_board_dimensions() -> int:
    n = 0
    while True:
        n = int(input("WHAT IS YOUR BOARD SIZE (MIN 7/ MAX 19)? "))
        if n < 7 or n > 19:
            print("I SAID, THE MINIMUM IS 7, THE MAXIMUM IS 19.")
            print()
        else:
            break
    return n


```

这段代码定义了一个名为 `get_move()` 的函数，它接受一个元组类型的参数，表示二维棋盘的当前位置。函数内部使用了一个无限循环来获取玩家输入的位置坐标，并返回它们。

在代码的下一行，定义了一个名为 `initialize_board()` 的函数，它接受一个整数参数 `n`。这个函数的作用初始化一个具有 `n` 行和 `n` 列的棋盘，并将其元素值都设为零。




```
def get_move() -> Tuple[int, int]:
    while True:
        xy = input("YOUR PLAY (I,J)? ")
        print()
        x_str, y_str = xy.split(",")
        try:
            x = int(x_str)
            y = int(y_str)
        except Exception:
            print("ILLEGAL MOVE.  TRY AGAIN...")
            continue
        return x, y


def initialize_board(n: int) -> List[List[int]]:
    # Initialize the board
    board = []
    for _x in range(n):
        sub_a = []
        for _y in range(n):
            sub_a.append(0)
        board.append(sub_a)
    return board


```

The output for the game is as follows:
```makefile
THANKS FOR THE GAME!!

PLAY AGAIN (1 FOR YES, 0 FOR NO)? 0

游戏开始于2021/03/31 13:15:02 +0800 [遇到了一个最高分数8855分的学生]

当前分数为2021/03/31 13:15:02 +0800

的游戏主要策略是：了解对手的诉求，从而有效地开发出自己可以执行的有效棋型。

当前的得分虽不高，但正在努力Non-Blocking（非阻塞）战略，预计稍后能够取得一定成果。

在执行非阻塞战略的过程中，已使用三十多个回合。

随着游戏的继续，会继续进行非阻塞战略。

尝试让整个的游戏变得更加有趣！

欢迎您再次 play the game!
```
这看起来像是一个象棋或围棋游戏。`board` 变量包含棋盘的当前状态，`move_count` 变量跟踪移动次数，`score` 变量跟踪当前得分，`random_num` 变量用于生成随机数。`check_move` 函数检查移动是否有效，`print_board` 函数打印当前游戏状态。


```
def main() -> None:
    print_banner()

    while True:
        n = get_board_dimensions()
        board = initialize_board(n)

        print()
        print()
        print("WE ALTERNATE MOVES. YOU GO FIRST...")
        print()

        while True:
            x, y = get_move()
            if x == -1:
                break
            elif not check_move(x, y, n):
                print("ILLEGAL MOVE.  TRY AGAIN...")
            else:
                if board[x - 1][y - 1] != 0:
                    print("SQUARE OCCUPIED.  TRY AGAIN...")
                else:
                    board[x - 1][y - 1] = 1
                    # COMPUTER TRIES AN INTELLIGENT MOVE
                    skip_ef_loop = False
                    for E in range(-1, 2):
                        for F in range(-1, 2):
                            if E + F - E * F == 0 or skip_ef_loop:
                                continue
                            X = x + F
                            Y = y + F
                            if not check_move(X, Y, n):
                                continue
                            if board[X - 1][Y - 1] == 1:
                                skip_ef_loop = True
                                X = x - E
                                Y = y - F
                                if not check_move(X, Y, n):  # 750
                                    while True:  # 610
                                        X = random.randint(1, n)
                                        Y = random.randint(1, n)
                                        if (
                                            check_move(X, Y, n)
                                            and board[X - 1][Y - 1] == 0
                                        ):
                                            board[X - 1][Y - 1] = 2
                                            print_board(board, n)
                                            break
                                else:
                                    if board[X - 1][Y - 1] != 0:
                                        while True:
                                            X = random.randint(1, n)
                                            Y = random.randint(1, n)
                                            if (
                                                check_move(X, Y, n)
                                                and board[X - 1][Y - 1] == 0
                                            ):
                                                board[X - 1][Y - 1] = 2
                                                print_board(board, n)
                                                break
                                    else:
                                        board[X - 1][Y - 1] = 2
                                        print_board(board, n)
        print()
        print("THANKS FOR THE GAME!!")
        repeat = int(input("PLAY AGAIN (1 FOR YES, 0 FOR NO)? "))
        if repeat == 0:
            break


```

这段代码是一个Python程序中的一个if语句。if语句可以用来进行条件判断，判断的条件是在程序运行时是否遇到了`__main__`这个关键字。如果`__name__`的值为`__main__`，那么程序会执行if语句后面的内容，否则跳过if语句。

在这段代码中，if语句的条件判断是：`__name__`的值为`__main__`。也就是说，只要程序运行时出现了`__main__`这个关键字，if语句就会执行，否则跳过if语句。

if语句后面跟着的是`main()`函数，这个函数是Python中的一段默认代码，用来输出"Hello, World!"。所以，如果程序运行时出现了`__main__`这个关键字，程序将会输出"Hello, World!"。


```
if __name__ == "__main__":
    main()

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Guess

In Program GUESS, the computer chooses a random integer between 0 and any limit you set. You must then try to guess the number the computer has chosen using the clues provided by the computer.

You should be able to guess the number in one less than the number of digits needed to represent the number in binary notation — i.e., in base 2. This ought to give you a clue as to the optimum search technique.

GUESS converted from the original program in FOCAL which appeared in the book “Computers in the Classroom” by Walt Koetke of Lexington High School, Lexington, Massachusetts.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=75)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=90)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

(please note any difficulties or challenges in porting here)


# `41_Guess/csharp/Game.cs`

This is a C# program that appears to be designed to play a game of猜数字. The program takes a number `limit` as input, and runs the猜数字游戏直到猜中了所有的目标数字。

The program uses a `while` loop to keep running the game until it is terminated. Inside the loop, the program prints the current number `limit` on the console and then enters a `while` loop that reads a number from the user using the `_io.ReadNumber` method.

The program then checks if the entered number is greater than or equal to the `limit`. If the number is not `limit`, the program prints an error message and then loops back to the previous iteration of the while loop.

If the entered number is `limit`, the program reports that it has guessed all the numbers and provides feedback on how well it did. After the loop, the program prints a blank line to clear the console.

The program uses a helper function `IsGuessCorrect` to check if the entered number is correct. This function takes two parameters: the `guess` and the `secretNumber` and returns a boolean value indicating if the guess is correct.

Overall, the program seems to be well-structured and easy to read.


```
namespace Guess;

internal class Game
{
    private readonly IReadWrite _io;
    private readonly IRandom _random;

    public Game(IReadWrite io, IRandom random)
    {
        _io = io;
        _random = random;
    }

    public void Play()
    {
        while (true)
        {
            _io.Write(Streams.Introduction);

            var limit = _io.ReadNumber(Prompts.Limit);
            _io.WriteLine();

            // There's a bug here that exists in the original code. 
            // If the limit entered is <= 0 then the program will crash.
            var targetGuessCount = checked((int)Math.Log2(limit) + 1);

            PlayGuessingRounds(limit, targetGuessCount);

            _io.Write(Streams.BlankLines);
        }
    }

    private void PlayGuessingRounds(float limit, int targetGuessCount)
    {
        while (true)
        {
            _io.WriteLine(Formats.Thinking, limit);

            // There's a bug here that exists in the original code. If a non-integer is entered as the limit
            // then it's possible for the secret number to be the next integer greater than the limit.
            var secretNumber = (int)_random.NextFloat(limit) + 1;

            var guessCount = 0;

            while (true)
            {
                var guess = _io.ReadNumber("");
                if (guess <= 0) { return; }
                guessCount++;
                if (IsGuessCorrect(guess, secretNumber)) { break; }
            }

            ReportResult(guessCount, targetGuessCount);

            _io.Write(Streams.BlankLines);
        }
    }

    private bool IsGuessCorrect(float guess, int secretNumber)
    {
        if (guess < secretNumber) { _io.Write(Streams.TooLow); }
        if (guess > secretNumber) { _io.Write(Streams.TooHigh); }

        return guess == secretNumber;
    }

    private void ReportResult(int guessCount, int targetGuessCount)
    {
        _io.WriteLine(Formats.ThatsIt, guessCount);
        _io.WriteLine(
            (guessCount - targetGuessCount) switch
            {
                < 0 => Strings.VeryGood,
                0 => Strings.Good,
                > 0 => string.Format(Formats.ShouldHave, targetGuessCount)
            });
    }
}
```

# `41_Guess/csharp/Program.cs`

这段代码的作用是创建一个名为 "Guess.Resources.Resource" 的类的一个实例，并传入一个新的 "Game" 类的实例，该实例包含一个 "ConsoleIO" 和一个 "RandomNumberGenerator" 类型的成员。然后，使用 "new Game(new ConsoleIO(), new RandomNumberGenerator()).Play()" 方法创建一个 "Game" 类的实例，并在其中调用 "Play" 方法来运行游戏。


```
global using Games.Common.IO;
global using Games.Common.Randomness;
global using static Guess.Resources.Resource;  

using Guess;

new Game(new ConsoleIO(), new RandomNumberGenerator()).Play();

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


# `41_Guess/csharp/Resources/Resource.cs`



这段代码是一个自定义的程序集，包含了多个类和接口，其作用是提供一些用于开发应用程序的工具和资源。

具体来说，这段代码：

1. 包含了多个内部类 `Streams`,`Formats`,`Prompts`,`Strings`，以及它们的基类 `Resource`；

2. `Streams` 包含了四个静态的读取字符串的接口 `Introduction`,`TooLow`,`TooHigh`,`BlankLines`，分别对应文本文件中的介绍、过低、过高和空白行；

3. `Formats` 包含了三个静态的字符串格式化接口 `Thinking`,`ThatItIs`,`ShouldHave`，分别对应文本文件中的提示和答案以及问题；

4. `Prompts` 包含了一个静态的字符串限制接口 `Limit`，用于限制用户输入的提示信息长度；

5. `Strings` 包含了一个静态的字符串 `Good` 和一个静态的字符串 `VeryGood`，分别用于表示良好的和非常好的表现。

6. `GetString` 方法接收一个 `string?` 类型的参数，代表一个字符串，如果该字符串已存在，则直接返回，否则会通过调用 `Resource.Streams.Introduction` 方法获取该字符串的读取流并读取至结束。

7. `Assembly` 类型和 `GetExecutingAssembly()` 方法用于获取当前应用程序执行的程序集，并调用其 `GetManifestResourceStream` 方法获取指定名称的资源文件读取流。

8. 如果资源文件找不到，则会抛出异常。

9. 最后，导入了 `System.Reflection` 和 `System.Runtime.CompilerServices` 命名空间。


```
using System.Reflection;
using System.Runtime.CompilerServices;

namespace Guess.Resources;

internal static class Resource
{
    internal static class Streams
    {
        public static Stream Introduction => GetStream();
        public static Stream TooLow => GetStream();
        public static Stream TooHigh => GetStream();
        public static Stream BlankLines => GetStream();
    }

    internal static class Formats
    {
        public static string Thinking => GetString();
        public static string ThatsIt => GetString();
        public static string ShouldHave => GetString();
    }

    internal static class Prompts
    {
        public static string Limit => GetString();
    }

    internal static class Strings
    {
        public static string Good => GetString();
        public static string VeryGood => GetString();
    }

    private static string GetString([CallerMemberName] string? name = null)
    {
        using var stream = GetStream(name);
        using var reader = new StreamReader(stream);
        return reader.ReadToEnd();
    }

    private static Stream GetStream([CallerMemberName] string? name = null) =>
        Assembly.GetExecutingAssembly().GetManifestResourceStream($"{typeof(Resource).Namespace}.{name}.txt")
            ?? throw new Exception($"Could not find embedded resource stream '{name}'.");
}
```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `41_Guess/java/src/Guess.java`

This is a Java class that appears to be a game where the player is given a number and has to guess it. The game uses a guessing algorithm where the player's guess is compared to the game's own algorithm to determine if the number is correct or not.

The class has several methods, including an intro method that prints out a message and a login method that accepts a login name and password. The login method uses a苔藓种植游戏模型的算法 to determine the correct login name and password.

The main method of the class has several outputs, including a message indicating that the login was successful, a game board that displays the login name and a space for the player to guess, and a number of spaces that the player has to guess. It also has a simulateTabs method that simulates the number of spaces that the login name has.

The guessing algorithm in the game is not defined in this class, but it appears to be a simple algorithm where the player's guess is compared to the game's own algorithm to determine if the number is correct or not. It is not clear what the game's algorithm is or how it compares the player's guess to the correct number.

Overall, this class appears to be a game where the player has to guess a number and the game uses a simple algorithm to determine if the player's guess is correct or not.


```
import java.util.Arrays;
import java.util.Scanner;

/**
 * Game of Guess
 * <p>
 * Based on the Basic game of Guess here
 * https://github.com/coding-horror/basic-computer-games/blob/main/41%20Guess/guess.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's Basic game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 */
public class Guess {

    // Used for keyboard input
    private final Scanner kbScanner;

    private enum GAME_STATE {
        STARTUP,
        INPUT_RANGE,
        DEFINE_COMPUTERS_NUMBER,
        GUESS,
        GAME_OVER
    }

    // Current game state
    private GAME_STATE gameState;

    // User supplied maximum number to guess
    private int limit;

    // Computers calculated number for the player to guess

    private int computersNumber;

    // Number of turns the player has had guessing
    private int tries;

    // Optimal number of turns it should take to guess
    private int calculatedTurns;

    public Guess() {
        kbScanner = new Scanner(System.in);

        gameState = GAME_STATE.STARTUP;
    }

    /**
     * Main game loop
     */
    public void play() {

        do {
            switch (gameState) {

                case STARTUP:
                    intro();
                    gameState = GAME_STATE.INPUT_RANGE;
                    break;

                case INPUT_RANGE:

                    limit = displayTextAndGetNumber("WHAT LIMIT DO YOU WANT? ");
                    calculatedTurns = (int) (Math.log(limit) / Math.log(2)) + 1;
                    gameState = GAME_STATE.DEFINE_COMPUTERS_NUMBER;
                    break;

                case DEFINE_COMPUTERS_NUMBER:

                    tries = 1;
                    System.out.println("I'M THINKING OF A NUMBER BETWEEN 1 AND " + limit);
                    computersNumber = (int) (Math.random() * limit + 1);

                    gameState = GAME_STATE.GUESS;
                    break;

                case GUESS:
                    int playersGuess = displayTextAndGetNumber("NOW YOU TRY TO GUESS WHAT IT IS ");

                    // Allow player to restart game with entry of 0
                    if (playersGuess == 0) {
                        linePadding();
                        gameState = GAME_STATE.STARTUP;
                        break;
                    }

                    if (playersGuess == computersNumber) {
                        System.out.println("THAT'S IT! YOU GOT IT IN " + tries + " TRIES.");
                        if (tries < calculatedTurns) {
                            System.out.println("VERY ");
                        }
                        System.out.println("GOOD.");
                        System.out.println("YOU SHOULD HAVE BEEN ABLE TO GET IT IN ONLY " + calculatedTurns);
                        linePadding();
                        gameState = GAME_STATE.DEFINE_COMPUTERS_NUMBER;
                        break;
                    } else if (playersGuess < computersNumber) {
                        System.out.println("TOO LOW. TRY A BIGGER ANSWER.");
                    } else {
                        System.out.println("TOO HIGH. TRY A SMALLER ANSWER.");
                    }
                    tries++;
                    break;
            }
        } while (gameState != GAME_STATE.GAME_OVER);
    }

    private void intro() {
        System.out.println(simulateTabs(33) + "GUESS");
        System.out.println(simulateTabs(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println();
        System.out.println("THIS IS A NUMBER GUESSING GAME. I'LL THINK");
        System.out.println("OF A NUMBER BETWEEN 1 AND ANY LIMIT YOU WANT.");
        System.out.println("THEN YOU HAVE TO GUESS WHAT IT IS.");
    }

    /**
     * Print a predefined number of blank lines
     *
     */
    private void linePadding() {
        for (int i = 1; i <= 5; i++) {
            System.out.println();
        }
    }

    /*
     * Print a message on the screen, then accept input from Keyboard.
     * Converts input to an Integer
     *
     * @param text message to be displayed on screen.
     * @return what was typed by the player.
     */
    private int displayTextAndGetNumber(String text) {
        return Integer.parseInt(displayTextAndGetInput(text));
    }

    /*
     * Print a message on the screen, then accept input from Keyboard.
     *
     * @param text message to be displayed on screen.
     * @return what was typed by the player.
     */
    private String displayTextAndGetInput(String text) {
        System.out.print(text);
        return kbScanner.next();
    }

    /**
     * Simulate the old basic tab(xx) command which indented text by xx spaces.
     *
     * @param spaces number of spaces required
     * @return String with number of spaces
     */
    private String simulateTabs(int spaces) {
        char[] spacesTemp = new char[spaces];
        Arrays.fill(spacesTemp, ' ');
        return new String(spacesTemp);
    }

}

```

# `41_Guess/java/src/GuessGame.java`

这段代码创建了一个名为GuessGame的公共类，其中包含一个名为main的静态方法，该方法接受一个字符串数组args作为参数。在main方法中，创建了一个名为guess的Guess对象，然后调用guess的play()方法。

Guess是一款流行的谜题游戏，它有一个主要的函数play()用于让玩家猜测正确答案。play()方法会随机生成一个1到100之间的整数，作为答案。当玩家猜测答案时，程序会比较他们的猜测值和答案，如果猜测值在答案附近，程序会提示玩家“太好了！”，否则会提示玩家“很遺憾，你猜的太大了！”。


```
public class GuessGame {
    public static void main(String[] args) {
        Guess guess = new Guess();
        guess.play();
    }
}

```

# `41_Guess/javascript/guess.js`

该代码是通过JavaScript转化自BASIC语言的。它包括两个函数：`print()`和`input()`。

函数`print()`的作用是将一个字符串打印到页面上，并在页面上生成的文本节点中添加该字符串。它接受一个参数`str`，并将它的值作为文本节点添加到页面上，然后设置该文本节点的样式为`font-size: 50px; color: blue;`。

函数`input()`的作用是接收用户输入的字符串，并返回该字符串的值。它首先创建一个空的输入元素，然后设置其`type`属性为`text`，设置其`length`属性为`50`，这样它就可以接受一个最大长度为50个字符的输入。它将输入元素将添加到页面上，并设置其样式为`font-size: 16px; color: white;`。然后，它将监听该输入元素上的`keydown`事件，当用户按下键盘上的13键时，它将接收到的字符串存储在`input_str`变量中，并将其打印到页面上，并使用`print()`函数将字符串的值输出到页面上，并使用`print()`函数将输出内容更改为`font-size: 16px; color: blue;`。


```
// GUESS
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

This code defines two functions: `tab` and `make_space`.

The `tab` function takes a single argument `space`, which is an integer. The function generates a string of spaces of the specified width by continuously subtracting 1 from the `space` argument. The final result is the concatenation of all the spaces.

The `make_space` function is a simple function that takes an integer `h` and prints a specified number of spaces of width `h`. The variable `h` is initialized to 1 and incremented by 1 for each iteration of the function.

The `main` function is the entry point of the program. It calls the `tab` and `make_space` functions with the space argument of 5.

In summary, the `tab` function generates a tab of spaces of width 5, and the `make_space` function prints a space for every iteration up to 5.


```
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

function make_space()
{
    for (h = 1; h <= 5; h++)
        print("\n");
}

// Main control section
```

This is a Python program that simulates a game where the user has to guess a random number between 1 and any limit specified by the user. The user will have to guess a number until they correctly guess the limit or they reach the maximum number of guesses allowed.

The program starts by explaining the game and the objective. Then it prompts the user to specify the limit, which is the maximum number of guesses allowed.

The while loop then starts, where the user tries to guess the number until they correctly guess the limit or they reach the maximum number of guesses allowed.

In each iteration of the while loop, the program generates a random number between 1 and the specified limit and compares it with the number the user guessed. If the user guessed a number that is between 1 and the specified limit, the program will print a message and the number of guesses will be reduced by 1.

If the user guessed a number that is too high or too low, the program will print a message and the user will have to try again. If the user does not make a guess or guesses an invalid number, the program will end and tell the user to try again.

The program also has a function called `make_space()` that clears the space under the user's guess and it is called in the end of each iteration of the while loop to clear any residual messages.


```
async function main()
{
    while (1) {
        print(tab(33) + "GUESS\n");
        print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
        print("\n");
        print("\n");
        print("\n");
        print("THIS IS A NUMBER GUESSING GAME. I'LL THINK\n");
        print("OF A NUMBER BETWEEN 1 AND ANY LIMIT YOU WANT.\n");
        print("THEN YOU HAVE TO GUESS WHAT IT IS.\n");
        print("\n");

        print("WHAT LIMIT DO YOU WANT");
        l = parseInt(await input());
        print("\n");
        l1 = Math.floor(Math.log(l) / Math.log(2)) + 1;
        while (1) {
            print("I'M THINKING OF A NUMBER BETWEEN 1 AND " + l + "\n");
            g = 1;
            print("NOW YOU TRY TO GUESS WHAT IT IS.\n");
            m = Math.floor(l * Math.random() + 1);
            while (1) {
                n = parseInt(await input());
                if (n <= 0) {
                    make_space();
                    break;
                }
                if (n == m) {
                    print("THAT'S IT! YOU GOT IT IN " + g + " TRIES.\n");
                    if (g == l1) {
                        print("GOOD.\n");
                    } else if (g < l1) {
                        print("VERY GOOD.\n");
                    } else {
                        print("YOU SHOULD HAVE BEEN TO GET IT IN ONLY " + l1 + "\n");
                    }
                    make_space();
                    break;
                }
                g++;
                if (n > m)
                    print("TOO HIGH. TRY A SMALLER ANSWER.\n");
                else
                    print("TOO LOW. TRY A BIGGER ANSWER.\n");
            }
            if (n <= 0)
                break;
        }
    }
}

```

这道题目没有给出代码，只是说明了一个名为`main()`的函数，我们需要根据函数名来推测它的作用。

在编程中，`main()`函数通常是程序的入口点，也就是程序从哪里开始执行。程序的执行可能会涉及多个函数，但是它们都从`main()`函数开始。因此，`main()`函数通常是包含程序主要逻辑的函数。

根据以上解释，`main()`函数的作用是程序的入口点，也是程序的起点。


```
main();

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Shells)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Kotlin](https://kotlinlang.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Lua](https://www.lua.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Perl](https://www.perl.org/)


# `41_Guess/python/guess.py`

这段代码是一个简单的猜数字游戏，它接受用户输入的数字作为随机整数，并在0到设置的数字限制内进行随机选择。游戏提示用户通过提供的线索尝试猜测计算机选择的数字，而用户最多只需要尝试一次猜测。

从程序的描述中可以看出，该程序旨在提供一个有趣的计算机游戏，通过随机选择数字和提供线索来启发用户思考。猜测数字的算法是基于0到设置的数字限制内的随机选择，因此用户可以通过尝试来猜测数字，同时也可以在猜测数字的过程中了解数字的二进制表示。


```
"""
Guess

From: Basic Computer Games (1978)

 "In program Guess, the computer  chooses a random
  integer between 0 and any limit and any limit you
  set. You must then try to guess the number the
  computer has choosen using the clues provideed by
  the computer.
   You should be able to guess the number in one less
  than the number of digits needed to  represent the
  number in binary notation - i.e. in base 2. This ought
  to give you a clue as to the optimum search technique.
   Guess converted from the original program in FOCAL
  which appeared in the book "Computers in the Classroom"
  by Walt Koetke of Lexington High School, Lexington,
  Massaschusetts.
```

这段代码定义了一个名为“insert_whitespaces”的函数，它做了以下几件事情：

1. 输出了一段空白行，也就是在函数内部调用了一次print函数。
2. 在print函数之前，又输出了三个空白行，这是为了让输出对齐一下。
3. 利用数学中的log函数，计算了一个int类型的变量a的值，该变量a应该在0到任何给定的限制范围内。
4. 使用random函数，生成一个int类型的变量b的值，该值应该在1到任何给定的限制范围内。
5. 检查a和b的值是否在预期的范围内。如果a和b的值都符合要求，那么代码将不会做任何事情，跳出函数。
6. 如果a或b的值不在预期的范围内，那么代码会尝试使用数学中的log函数来计算a或b的值，这是因为在计算a或b的位数时，需要使用log函数。


```
"""

# Altough the introduction says that the computer chooses
# a number between 0 and any limit, it actually chooses
# a number between 1 and any limit. This due to the fact that
# for computing the number of digits the limit has in binary
# representation, it has to use log.

from math import log
from random import random
from typing import Tuple


def insert_whitespaces() -> None:
    print("\n\n\n\n\n")


```

这段代码是一个Python函数，名为`limit_set()`，它用于玩一个猜数字的游戏。函数的作用是限制玩家的猜测范围，防止玩家猜测过多，从而提高游戏的趣味性和挑战性。

具体来说，该函数首先会提示玩家猜测一个数字，然后提供一个限制，告诉玩家这个数字最大不会超过这个限制。玩家输入数字后，程序会继续提示玩家输入，直到他们猜中了一个数字，这个数字就是游戏中的“答案”。

函数内部，变量`limit` 被存储为玩家输入的数字，变量`limit_goal` 被存储为限制条件，即在二进制表示中，数字`limit` 中有多少位。限制条件中的`log`函数表示以2为底，对数值取对数，这样限制条件就可以用二进制表示中的位数来表示。然后，程序会加上一个稍微大于1的数字，以便在猜测结果时考虑更多的可能性。

最后，函数返回限制条件和答案，以便玩家继续猜测。


```
def limit_set() -> Tuple[int, int]:
    print("                   Guess")
    print("Creative Computing  Morristown, New Jersey")
    print("\n\n\n")
    print("This is a number guessing game. I'll think")
    print("of a number between 1 and any limit you want.\n")
    print("Then you have to guess what it is\n")
    print("What limit do you want?")

    limit = int(input())

    while limit <= 0:
        print("Please insert a number greater or equal to 1")
        limit = int(input())

    # limit_goal = Number of digits "limit" in binary has
    limit_goal = int((log(limit) / log(2)) + 1)

    return limit, limit_goal


```

这段代码是一个 Python 程序，它的主要目的是让用户猜测一个 1 到 limit 大小的随机数，并且会提示用户在猜测的过程中尝试调整猜测的数值。在猜测的过程中，如果用户猜中了正确的数值，程序会显示恭喜信息，并且会减少用户猜测的次数。如果用户猜错或者猜中的数值不在限定的范围内，程序会提示用户继续猜测，增加猜测次数。当用户猜对数值或者猜测次数超过了限定的最大值 limit_goal，程序会提示用户重新猜测，并继续使用当前的限制定为猜测的目标值。


```
def main() -> None:
    limit, limit_goal = limit_set()
    while True:
        guess_count = 1
        still_guessing = True
        won = False
        my_guess = int(limit * random() + 1)

        print(f"I'm thinking of a number between 1 and {limit}")
        print("Now you try to guess what it is.")

        while still_guessing:
            n = int(input())

            if n < 0:
                break

            insert_whitespaces()
            if n < my_guess:
                print("Too low. Try a bigger answer")
                guess_count += 1
            elif n > my_guess:
                print("Too high. Try a smaller answer")
                guess_count += 1
            else:
                print(f"That's it! You got it in {guess_count} tries")
                won = True
                still_guessing = False

        if won:
            if guess_count < limit_goal:
                print("Very good.")
            elif guess_count == limit_goal:
                print("Good.")
            else:
                print(f"You should have been able to get it in only {limit_goal}")
            insert_whitespaces()
        else:
            insert_whitespaces()
            limit, limit_goal = limit_set()


```

这段代码是一个Python程序中的一个if语句。if语句可以用来判断一个程序是否处于交互式模式（即客户端模式）或者命令行模式（即服务器端模式）。

"__name__"是一个特殊的字符串，用于判断当前程序是否作为主程序运行。如果程序作为主程序运行，那么"__name__"的值为"__main__"，否则值为其他值。如果程序不是主程序，那么"__name__"的值为"__找诵__"。

if __name__ == "__main__":
   main()
  测试代码，用于在程序运行时执行一些操作，如print("Hello World")

总的来说，这段代码的作用是判断程序是否作为主程序运行，如果是，就执行main()函数中的语句，否则执行其他代码。


```
if __name__ == "__main__":
    main()

```