# BasicComputerGames源码解析 12

# `04_Awari/csharp/Program.cs`

这段代码是一个AWARI游戏的人工智能，其目的是让玩家和计算机在玩游戏的过程中交替进行移动和决策。在游戏开始时，游戏将被重置并输出"AWARI"。

在游戏进行过程中，游戏将不断显示当前的游戏状态(包括玩家和计算机的移动)。在每次游戏状态更改时，游戏将调用一个名为`DisplayGame`的函数，该函数将输出当前游戏状态。

在游戏循环中，游戏将在每个迭代中调用`Reset`函数来重置游戏状态。然后，游戏将调用一个名为`PlayerMove`的函数，让玩家决定何时移动。如果玩家在迭代中决定移动，游戏将调用`ComputerTurn`函数来让计算机进行移动。

在每次移动或决策后，游戏将调用`DisplayGame`函数来显示当前游戏状态。然后，游戏将调用`GetOutcome`函数来获取游戏的结果，并输出胜利者或结果。胜利者或结果将根据玩家或计算机的游戏状态来确定。如果游戏没有完成，则会抛出`InvalidOperationException`异常。

最后，游戏将循环等待游戏完成，并输出"AWARI"。


```
﻿using Awari;

Console.WriteLine(Tab(34) + "AWARI");
Console.WriteLine(Tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");

Game game = new();

while (true)
{
    game.Reset();
    DisplayGame();

    while (game.State != GameState.Done)
    {
        switch (game.State)
        {
            case GameState.PlayerMove:
                PlayerMove(second: false);
                break;
            case GameState.PlayerSecondMove:
                PlayerMove(second: true);
                break;
            case GameState.ComputerMove:
                ComputerTurn();
                break;
        }

        DisplayGame();
    }

    var outcome = game.GetOutcome();

    string outcomeLabel =
        outcome.Winner switch
        {
            GameWinner.Computer => $"I WIN BY {outcome.Difference} POINTS",
            GameWinner.Draw => "DRAWN GAME",
            GameWinner.Player => $"YOU WIN BY {outcome.Difference} POINTS",
            _ => throw new InvalidOperationException($"Unexpected winner {outcome.Winner}."),
        };
    Console.WriteLine(outcomeLabel);
    Console.WriteLine();
}

```

这段代码是一个方法 `DisplayGame()`，它用于在控制台显示游戏中的 pit 信息。

具体来说，这个方法包含以下操作：

1. 在屏幕上打印计算机的 pit 信息。为此，它遍历游戏中的 `ComputerPits` 数组，并使用 `Console.Write` 方法将每个 pit 输出到控制台上。由于 `ComputerPits` 数组是按 Reverse() 方法反序排列的，因此遍历结果是计算机 pit 信息的相反顺序。
2. 在屏幕上打印 both homes 信息。为此，它使用 `Console.WriteLine` 方法将“both homes”字符串输出到控制台，并使用 `Tab(19)` 方法在字符串中插入 19 个空格，以便在控制台上的显示效果更像一个游戏中的 both homes 界面。
3. 在屏幕上打印玩家的 pit 信息。为此，它遍历游戏中的 `PlayerPits` 数组，并使用 `Console.Write` 方法将每个 pit 输出到控制台上。
4. 在屏幕上打印一个空行，以使控制台上的输出更易于阅读。

最终，这段代码会输出类似这样的游戏 pit 信息，其中：
```
  Comput
   Home    
  Player
   Hope
```



```
void DisplayGame()
{
    // display the computer's pits
    Console.Write("   ");
    foreach (var pit in game.ComputerPits.Reverse())
        Console.Write($"{pit,2} ");
    Console.WriteLine();

    // display both homes
    Console.WriteLine($"{game.ComputerHome,2}{Tab(19)}{game.PlayerHome,2}");

    // display the player's pits
    Console.Write("   ");
    foreach (var pit in game.PlayerPits)
        Console.Write($"{pit,2} ");
    Console.WriteLine();

    Console.WriteLine();
}

```

这段代码定义了一个名为 PlayerMove 的函数，用于让玩家移动棋子。函数有两个参数，第二个参数为布尔类型，表示是否为人工智能选择棋局。函数中首先定义了一个名为 move 的整数变量，用于存储玩家的棋子移动数值。接着定义了一个名为 GetMove 的函数，用于在玩家确认后获取他们想要移动的棋子数。在函数中使用了一个 while 循环来不断询问玩家想要移动多少棋子，并检查输入是否合法。如果输入合法(即是一个数字)，函数会尝试将输入转换为一个整数，并判断该棋子是否在合法范围内。如果输入不合法或者棋子超出了范围，函数会输出一个错误信息并继续等待输入。最后，函数会调用 game.PlayerMove 函数来更新游戏状态。


```
void PlayerMove(bool second = false)
{
    int move = GetMove(second);
    game.PlayerMove(move);
}

int GetMove(bool second)
{
    string prompt = second ? "AGAIN? " : "YOUR MOVE? ";

    while (true)
    {
        Console.Write(prompt);

        string input = Console.ReadLine() ?? "";

        // input must be a number between 1 and 6, and the pit must have > 0 beans
        if (int.TryParse(input, out int move)
         && game.IsLegalPlayerMove(move))
            return move;

        Console.WriteLine("ILLEGAL MOVE");
    }
}

```

这段代码定义了一个名为 "ComputerTurn" 的函数，其作用是执行游戏中的电脑的移动操作。函数内部使用了一个名为 "game" 的引用，该引用可能是一个自定义的游戏类，负责生成计算机的移动操作。

函数的内部定义了一个名为 "moves" 的变量，它可能用于存储游戏中的计算机的移动列表。变量内部使用 "ComputerTurn" 函数，生成了一个字符串类型的变量 "movesString"，将所有计算机的移动操作的字符串连接成一个字符串，用","字符串连接。

然后，函数的内部使用 "Console.WriteLine" 函数，输出字符串类型的变量 "movesString"。

接着，定义了一个名为 "Tab" 的函数，它的作用是打印出指定数目的制表符。函数内部使用 "int" 类型变量 "n"，并返回一个字符串类型的变量 "Tab"，用制表符将 "n" 打印出来，其中 "n" 是参数。


```
void ComputerTurn()
{
    var moves = game.ComputerTurn();
    string movesString = string.Join(",", moves);

    Console.WriteLine($"MY MOVE IS {movesString}");
}

string Tab(int n) => new(' ', n);

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


# `04_Awari/java/Awari.java`

This is a Java program that simulates a game of Connect-the-Dot using a basic paper-based game-board and a computer algorithm to determine the player to win. The computer algorithm uses the strategy of making the first move and wins half the points.

The program starts by initializing the board and the computer player as player 1 and setting the computer player to player 2. Then it runs a loop that plays the game until one of the players wins or the game is a draw.

The `distribute()` method is used to distribute the points among the players according to their position on the board. The `calculateOpposite()` method is used to calculate the opposite team's player for each player's home team.

The `isGameOver()` method checks if either player has won or if the game is a draw. If either player has won, it prints a message and if the game is a draw it prints "DRAW".

The `printBoard()` method is used to print the game board.

Overall, this program provides a basic framework for simulating a game of Connect-the-Dot using a paper-based game-board and a computer algorithm.


```
import java.util.Scanner;
import java.util.Random;

public class Awari{
	int []board;
	private final int playerPits;
	private final int computerPits;
	private final int playerHome;
	private final int computerHome;
	Scanner input;
	int sumPlayer;
	int sumComputer;
	Awari(){
		input = new Scanner(System.in);
		playerPits = 0;
		computerPits = 7;
		playerHome = 6;
		computerHome = 13;
		sumPlayer = 18;
		sumComputer = 18;
		board = new int [14];
		for (int i=0;i<6;i++){
			board[playerPits+i]=3;
			board[computerPits+i]=3;
		}
		System.out.println("		 AWARI");
		System.out.println("CREATIVE COMPUTING MORRISTOWN, NEW JERSEY");
		printBoard();
		playerMove(true);
	}

	private void printBoard(){
		System.out.print("\n    ");
		for (int i=0;i<6;i++){
			System.out.print(String.format("%2d",board[12-i]));
			System.out.print("  ");
		}
		System.out.println("");
		System.out.print(String.format("%2d",board[computerHome]));
		System.out.print("                          ");
		System.out.println(String.format("%2d",board[playerHome]));
		System.out.print("    ");
		for(int i=0;i<6;i++){
			System.out.print(String.format("%2d",board[playerPits+i]));
                        System.out.print("  ");
		}
		System.out.println("");
	}

	private void playerMove(boolean val){
		System.out.println("\nComputerSum PlayerSum"+sumComputer+" "+sumPlayer);
		if(val == true)
			System.out.print("YOUR MOVE? ");
		else
			System.out.print("AGAIN? ");
		int move =  input.nextInt();
		while(move<1||move>6||board[move-1]==0){
			System.out.print("INVALID MOVE!!! TRY AGAIN  ");
			move = input.nextInt();
		}
		int seeds = board[move-1];
		board[move-1] = 0;
		sumPlayer -= seeds;
		int last_pos = distribute(seeds,move);
		if(last_pos == playerHome){
			printBoard();
			if(isGameOver(true)){
				System.exit(0);
			}
			playerMove(false);
		}
		else if(board[last_pos] == 1&&last_pos != computerHome){
			int opp = calculateOpposite(last_pos);
			if(last_pos<6){
				sumPlayer+=board[opp];
				sumComputer-=board[opp];
			}
			else{
				sumComputer+=board[opp];
				sumPlayer-=board[opp];
			}
			board[last_pos]+=board[opp];
			board[opp] = 0;
			printBoard();
			if(isGameOver(false)){
				System.exit(0);
			}
			computerMove(true);
		}
		else{
			printBoard();
			if(isGameOver(false)){
				System.exit(0);
			}
			computerMove(true);
		}
	}

	private void computerMove(boolean value){
		int val=-1;
		System.out.println("\nComputerSum PlayerSum"+sumComputer+" "+sumPlayer);
		for(int i=0;i<6;i++){
			if(6-i == board[computerPits+i])
				val = i;
		}
		int move ;
		if(val == -1)
		{
			Random random = new Random();
			move = random.nextInt(6)+computerPits;
			while(board[move] == 0){
				move = random.nextInt(6)+computerPits;
			}
			if(value == true)
				System.out.println(String.format("MY MOVE IS %d ",move-computerPits+1));
			else
				System.out.println(String.format(",%d",move-computerPits+1));
			int seeds = board[move];
			board[move] = 0;
			sumComputer-=seeds;
			int last_pos = distribute(seeds,move+1);
			if(board[last_pos] == 1&&last_pos != playerHome){
                	        int opp = calculateOpposite(last_pos);
				 if(last_pos<6){
	                                sumPlayer+=board[opp];
        	                        sumComputer-=board[opp];
                	        }
                        	else{
	                                sumComputer+=board[opp];
        	                        sumPlayer-=board[opp];
                	        }
        	                board[last_pos]+=board[opp];
	                        board[opp] = 0;
                        	printBoard();
                	        if(isGameOver(false)){
        	                        System.exit(0);
	                        }
                	}
			else{
				printBoard();
	                        if(isGameOver(false)){
        	                        System.exit(0);
                	        }
			}
			playerMove(true);
		}
		else {
			move = val+computerPits;
			if(value == true)
				System.out.print(String.format("MY MOVE IS %d",move-computerPits+1));
			else
				System.out.print(String.format(",%d",move-computerPits+1));
			int seeds = board[move];
                        board[move] = 0;
                        sumComputer-=seeds;
                        int last_pos = distribute(seeds,move+1);
			if(last_pos == computerHome){
				if(isGameOver(true) ){
					System.exit(0);
				}
				computerMove(false);
			}
		}
	}


	private int distribute(int seeds, int pos){
		while(seeds!=0){
			if(pos==14)
				pos=0;
			if(pos<6)
				sumPlayer++;
			else if(pos>6&&pos<13)
				sumComputer++;
			board[pos]++;
			pos++;
			seeds--;
		}
		return pos-1;
	}

	private int calculateOpposite(int pos){
		return 12-pos;
	}

	private boolean isGameOver(boolean show){
		if(sumPlayer == 0 || sumComputer == 0){
			if(show)
				printBoard();
			System.out.println("GAME OVER");
			if(board[playerHome]>board[computerHome]){
				System.out.println(String.format("YOU WIN BY %d POINTS",board[playerHome]-board[computerHome]));
			}
			else if(board[playerHome]<board[computerHome]){
				System.out.println(String.format("YOU LOSE BY %d POINTS",board[computerHome]-board[playerHome]));
			}
			else{
				System.out.println("DRAW");
			}
			return true;
		}
		return false;
	}


}

```

# `04_Awari/java/AwariGame.java`

这段代码定义了一个名为AwariGame的公共类，该类具有一个名为main的静态方法，其参数是一个字符串数组args，表示程序接收的命令行参数。

在main方法中，使用new关键字创建了一个Awari对象awari，并将该对象赋值给一个名为awari的变量。Awari是一个类，该类可能是一个游戏或应用程序的模拟。

由于在代码中没有提供Awari类的定义，因此无法提供其具体的实现。


```
public class AwariGame {
    public static void main(String[] args) {
        Awari awari = new Awari();
    }
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `04_Awari/javascript/awari.js`

这段代码的作用是向网页上的一个 output 元素中输入用户输入的字符串，并将输入的字符串输出到该元素中。

具体来说，这个代码包含两个函数：input()和print()。

input()函数的作用是获取用户输入的字符串，将其转换成小写，然后在网页上创建一个 input 元素，设置其 type 为 text，长度为 50。这个 input 元素会被添加到网页上的一个 output 元素中，并且该 output 元素的 focus 属性会被设置为 input 元素。当用户点击 input 元素时，input()函数会监听 keydown 事件，在事件处理程序中获取用户输入的值，并将其存储在 input_str 变量中。最后，input()函数会调用 print() 函数将 input_str 字符串输出到网页上的 output 元素中，并输出一个换行符。

print()函数的作用是接收输入的字符串，将其存储在变量中，并将其输出到网页上的 output 元素中。


```
// AWARI
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

这段代码定义了一个名为 `tab` 的函数，它接受一个参数 `space`，它的作用是打印一个指定长度的空格字符串。

具体来说，代码中首先创建了一个名为 `str` 的字符串变量，并使用一个循环来遍历 `space` 次减1的结果，每次循环都将一个空格添加到 `str` 字符串的末尾。这样，当 `space` 次减1的结果为 0 时，循环将停止，此时 `str` 字符串将包含一个由空格组成的字符串，它的长度为 `space`。

接着，代码使用 `print` 函数来打印调用 `tab` 函数的结果。在 `tab` 函数中，我们将调用 `print` 函数两次，每次传递一个不同的参数，第一个参数是 `tab(34)`，第二个参数是 `tab(15)`。这些参数将被传递给 `tab` 函数中的循环，用于计算字符串中的空格数量。

在循环内部，我们创建了两个名为 `b` 和 `g` 的二维数组，用于存储每个字符串中的字符数量。初始时，`b` 和 `g` 数组都包含 0，表示每个字符串中都没有字符。

在循环中，我们使用 `str` 变量中的字符来更新 `b` 和 `g` 数组中的字符数量。每次循环结束后，我们将 `b` 和 `g` 数组中的字符数量都加倍，以使它们与 `str` 中的字符数量保持一致。

最后，由于没有手动关闭循环或清除变量，程序将继续运行，直到被强制终止。


```
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

print(tab(34) + "AWARI\n");
print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");

n = 0;

b = [0,0,0,0,0,0,0,0,0,0,0,0,0,0];
g = [0,0,0,0,0,0,0,0,0,0,0,0,0,0];
```

这段代码定义了一个名为 `f` 的列表，其中包含 50 个元素，每个元素都为 0。

接下来，定义了一个名为 `show_number` 的函数，用于打印数字。

然后，定义了一个名为 `show_board` 的函数，用于打印矩阵。

`show_number` 函数的实现是判断传入的数字是否小于 10，如果是，则输出该数字，否则输出 "0"。

`show_board` 函数的实现是打印一个 7x7 的矩阵，其中的数字由 `show_number` 函数打印。对于每个元素，根据其行和列索引打印字符 " " 或 " " 加数字 "0"。该函数还打印了一个垂直于其行和列索引的 "|" 字符，以分隔不同的行和列。


```
f = [];
for (i = 0; i <= 50; i++) {
    f[i] = 0;
}

function show_number(number)
{
    if (number < 10)
        print("  " + number + " ");
    else
        print(" " + number + " ");
}

function show_board()
{
    var i;

    print("\n");
    print("   ");
    for (i = 12; i >= 7; i--)
        show_number(b[i]);
    print("\n");
    i = 13;
    show_number(b[i]);
    print("                       " + b[6] + "\n");
    print("   ");
    for (i = 0; i <= 5; i++)
        show_number(b[i]);
    print("\n");
    print("\n");
}

```

这是一个 JavaScript 函数，名为 "do_move"，其作用是移动棋盘上的棋子，使其遵循 8x8 的棋盘约束，并在移动过程中计算棋子的周围得分。

具体来说，函数实现以下步骤：

1. 将变量 k、e 和 c 的值设置为 0、0 和 0，分别代表棋子移动后不会超出棋盘边界，以及将棋子移动后不会改变其位置。

2. 如果 k 的值大于 6，则将 k 值减去 7，以保证 k 的值不会超过 6。

3. 增加变量 f 中棋子的值，其值为 f[n] * 6 + k，其中 f[n] 代表棋子 "n" 周围的六个棋子的值，k 代表棋子 "n" 移动后的值。

4. 遍历变量 b 中的每个棋子，如果该棋子周围的六个棋子中有任何一个不为 0，则计算变量 e 的值，如果 e 的值为 1，则立即返回，否则继续遍历。

5. 如果遍历完所有的棋子后，变量 e 的值仍然为 0，则说明该棋子已经移动到棋盘的角落处，需要将其周围的六个棋子的值都设置为 0。

6. 在每次移动棋子后，函数会计算出移动后棋子的位置，并更新棋子周围的得分，包括使用动态规划计算得分和修改棋子位置。


```
function do_move()
{
    k = m;
    adjust_board();
    e = 0;
    if (k > 6)
        k -= 7;
    c++;
    if (c < 9)
        f[n] = f[n] * 6 + k
        for (i = 0; i <= 5; i++) {
            if (b[i] != 0) {
                for (i = 7; i <= 12; i++) {
                    if (b[i] != 0) {
                        e = 1;
                        return;
                    }
                }
            }
        }
}

```



这段代码定义了一个名为adjust_board的函数，其作用是调整棋盘上的元素，使得每行、每列、每个对角线上的元素和都等于2。

具体实现过程如下：

1. 从第二行开始，将该行所有元素都设为0，即b[m] = 0。

2. 从第一行开始，遍历p所指向的元素，将其下标m的值加1，即b[m] = 1。

3. 在遍历过程中，如果p所指向的元素下标m大于13，则执行以下操作：将m下移14，即m = m - 14，并将p下移1，即p = p - 1。

4. 如果p所指向的元素下标m等于6或13，则执行以下操作：将该行所有元素都设为0，即b[m-1] = b[m-1] = 0，并将该列所有元素都设为0，即b[m-1+i] = b[m-1+i] = 0，其中i从0到9。

5. 如果p所指向的元素下标m为12-6或12-13，则执行以下操作：将该行所有元素都设为0，即b[m+12-i] = b[m+12-i] = 0，其中i从0到9。

6. 如果b[m-1]为非0元素，则执行以下操作：将该元素和为13并与当前元素和相加，即new_sum = new_sum + b[m-1] + 1，并将该元素设为0，即b[m-1] = 0。

7. 如果b[m-12]为非0元素，则执行以下操作：将该元素和为13并与当前元素和相加，即new_sum = new_sum + b[m-12] + 1，并将该元素设为0，即b[m-12] = 0。

8. 如果当前元素和等于13，则输出“OK”。

9. 如果所有元素都为0，则输出“ADJUSTED”。


```
function adjust_board()
{
    p = b[m];
    b[m] = 0;
    while (p >= 1) {
        m++;
        if (m > 13)
            m -= 14;
        b[m]++;
        p--;
    }
    if (b[m] == 1) {
        if (m != 6 && m != 13) {
            if (b[12 - m] != 0) {
                b[h] += b[12 - m] + 1;
                b[m] = 0;
                b[12 - m] = 0;
            }
        }
    }
}

```

这段代码定义了一个名为 computer_move 的函数，它实现了计算机的走棋游戏。以下是该函数的功能解释：

1. 初始化游戏板：定义了游戏板的行列数、初始值以及棋子的初始位置。
2. 生成初始游戏板：通过一个循环来遍历所有的行列，将当前行列的棋子值初始化为默认值 0。
3. 处理移动棋子：定义了一个 for 循环，接收一个棋子（行棋子，列棋子，或者 dz），计算移动后的位置，并更新棋子的位置。其中，行棋子移动跨越了整个棋盘，列棋子需要减去 7，而 dz 则代表对当前棋子的跨越。
4. 处理紧接的行：在 for 循环外，定义了一个 if 语句，判断当前行是否紧接在 b 中的 0 位置，如果是，则表示当前行可以移动的步数等于 b 中 0 位置与紧接的 0 位置之间的步数。
5. 处理紧接的列：与处理紧接的行类似，判断当前列是否紧接在 b 中的 0 位置，如果是，则表示当前列可以移动的步数等于 b 中 0 位置与紧接的 0 位置之间的步数。
6. 处理跨越棋盘的移动：定义了一个 do_move 函数，在当前棋子的行列跨越整个棋盘时，需要计算不同方向上的移动步数，最终更新棋子的位置。
7. 处理计算分数：定义了一个 f 函数，计算当前棋子与目标棋子之间的分数。通过 f 函数计算出来的移动步数，会自动影响到棋子的分数计算。
8. 恢复游戏板：在 for 循环外，定义了一个 do_move 函数，用于恢复游戏板。这个函数会将当前棋子的值更新为从棋盘中间位置恢复过来，并在最后调整一下棋子的值。
9. 判断游戏是否结束：在 for 循环外，定义了一个 should_quit 函数，用于判断游戏是否结束。如果已经判断出游戏结束，那么这个函数的返回值将为 true，使得游戏结束。

总之，这个函数实现了计算机的走棋游戏，玩家可以在游戏中通过移动棋子来走动，并与程序进行交互。


```
function computer_move()
{
    d = -99;
    h = 13;
    for (i = 0; i<= 13; i++)	// Backup board
        g[i] = b[i];
    for (j = 7; j <= 12; j++) {
        if (b[j] == 0)
            continue;
        q = 0;
        m = j;
        adjust_board();
        for (i = 0; i <= 5; i++) {
            if (b[i] == 0)
                continue;
            l = b[i] + i;
            r = 0;
            while (l > 13) {
                l -= 14;
                r = 1;
            }
            if (b[l] == 0) {
                if (l != 6 && l != 13)
                    r = b[12 - l] + r;
            }
            if (r > q)
                q = r;
        }
        q = b[13] - b[6] - q;
        if (c < 8) {
            k = j;
            if (k > 6)
                k -= 7;
            for (i = 0; i <= n - 1; i++) {
                if (f[n] * 6 + k == Math.floor(f[i] / Math.pow(7 - c, 6) + 0.1))
                    q -= 2;
            }
        }
        for (i = 0; i <= 13; i++)	// Restore board
            b[i] = g[i];
        if (q >= d) {
            a = j;
            d = q;
        }
    }
    m = a;
    print(m - 6);
    do_move();
}

```

This is a board game where two players take turns moving pieces on a 21-point grid. The first player to capture all of their opponents' pieces is declared the winner. If there is a draw, the game is considered a draw. The game also has a computer player that players can choose to play against.

The code looks like it has a good starting point, but there are a few issues that could be improved upon. For example, the code does not check for the case where the number of pieces remaining is negative, which could happen if one player has more points than the other. Additionally, the code does not handle the case where the computer player chooses not to play.

Additionally, the code does not provide any explanation for why the computer player might choose not to play. This could be added to improve the readability of the code.

Overall, the code has the potential to be a good starting point for a board game simulator, but there are some areas that could be improved upon to make the program more robust and user-friendly.



```
// Main program
async function main()
{
    while (1) {
        print("\n");
        print("\n");
        e = 0;
        for (i = 0; i <= 12; i++)
            b[i] = 3;

        c = 0;
        f[n] = 0;
        b[13] = 0;
        b[6] = 0;

        while (1) {
            show_board();
            print("YOUR MOVE");
            while (1) {
                m = parseInt(await input());
                if (m < 7) {
                    if (m > 0) {
                        m--;
                        if (b[m] != 0)
                            break;
                    }
                }
                print("ILLEGAL MOVE\n");
                print("AGAIN");
            }
            h = 6;
            do_move();
            show_board();
            if (e == 0)
                break;
            if (m == h) {
                print("AGAIN");
                while (1) {
                    m = parseInt(await input());
                    if (m < 7) {
                        if (m > 0) {
                            m--;
                            if (b[m] != 0)
                                break;
                        }
                    }
                    print("ILLEGAL MOVE\n");
                    print("AGAIN");
                }
                h = 6;
                do_move();
                show_board();
            }
            if (e == 0)
                break;
            print("MY MOVE IS ");
            computer_move();
            if (e == 0)
                break;
            if (m == h) {
                print(",");
                computer_move();
            }
            if (e == 0)
                break;
        }
        print("\n");
        print("GAME OVER\n");
        d = b[6] - b[13];
        if (d < 0)
            print("I WIN BY " + -d + " POINTS\n");
        else if (d == 0) {
            n++;
            print("DRAWN GAME\n");
        } else {
            n++;
            print("YOU WIN BY " + d + " POINTS\n");
        }
    }
}

```

这是经典的 "Hello, World!" 程序，用于在 C 语言环境中启动一个新程序并输出 "Hello, World!" 消息。该程序将以下代码执行：

```
#include <stdio.h>

int main() {
  printf("Hello, World!\n");
  return 0;
}
```

程序首先引入了 `stdio.h` 头文件，该文件包含了在程序中使用 printf() 函数所必需的函数。然后，程序定义了一个名为 `main` 的函数，它是程序的入口点。

在 `main()` 函数中，程序使用 `printf()` 函数输出了一条消息 "Hello, World!" 到屏幕。`\n` 是一个转义序列，表示一个 Unicode 字符并换行，这意味着在 `printf()` 函数中使用的字符将覆盖屏幕上的任何其他内容，并且将使用 UTF-8 编码。

最后，程序使用 `return 0;` 语句告诉操作系统程序成功执行，并将 0 作为程序的返回值。在操作系统中，0 通常表示成功，而不是像其他情况下的错误状态。


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


# `04_Awari/python/awari.py`

这段代码是一个名为"Ancient African Game"的游戏，也称为Kalah或Mancala游戏。这个游戏最初是由Dave LeCompte创作的。在这段注释中，作者提到这个游戏最初是由70行BASIC代码创建的，并且他已经将这个游戏从原始代码中移植过来了。作者发现这个游戏的代码非常有效率（密集 packed），并且在 densely packed（密集地堆积）的情况下表现出色。作者还指出，原始代码的变量名相当神秘（as was forced by BASIC's limitation on long (2+ character) variable names）。作者在这段注释中已经尽自己最大的努力来解释这个游戏的工作原理。


```
"""
AWARI

An ancient African game (see also Kalah, Mancala).

Ported by Dave LeCompte
"""

# PORTING NOTES
#
# This game started out as 70 lines of BASIC, and I have ported it
# before. I find it somewhat amazing how efficient (densely packed) the
# original code is. Of course, the original code has fairly cryptic
# variable names (as was forced by BASIC's limitation on long (2+
# character) variable names). I have done my best here to interpret what
```

这段代码是一个文本，解释了每个变量的用途，并为每个变量提供了合理的名称。

这段代码的作用是评估每个游戏的游戏树，并记录游戏历史。当人类玩家赢得游戏或Draw时，计算机会增加游戏编号，从而记录这个失败的游戏编号。当计算机评估移动时，它检查当前游戏状态与这些失败的游戏记录是否匹配。如果是，则移动会被判处两个分数的惩罚。这段代码与MENACE（一种机械设备）有关，该设备据说具有评估游戏树的功能。


```
# each variable is doing in context, and rename them appropriately.
#
# I have endeavored to leave the logic of the code in place, as it's
# interesting to see a 2-ply game tree evaluation written in BASIC,
# along with what a reader in 2021 would call "machine learning".
#
# As each game is played, the move history is stored as base-6
# digits stored losing_book[game_number]. If the human player wins or
# draws, the computer increments game_number, effectively "recording"
# that loss to be referred to later. As the computer evaluates moves, it
# checks the potential game state against these losing game records, and
# if the potential move matches with the losing game (up to the current
# number of moves), that move is evaluated at a two point penalty.
#
# Compare this, for example with MENACE, a mechanical device for
```

这段代码是一个简单的 Tic-Tac-Toe 游戏的实现，使用了 learning 模式。这个模式通过在每次游戏结束时减少洋洋得意和失败的结果，来鼓励玩家继续游戏。在这个模式下，玩家需要通过窗口（Windows 和 Linux）来输入游戏历史。

具体来说，这段代码实现了以下功能：

1. 在游戏开始时，初始化屏幕为空，用 draw 函数绘制出所有的格子。
2. 玩家输入游戏历史，使用接收输入的文件描述符来读取文件中的内容并将其转换为字符串。
3. 在循环内，判断玩家输入是否为结束（End Game）。如果是，就清除屏幕并结束游戏。否则，清空游戏历史并更新屏幕。
4. 在每次循环结束后，将游戏历史中的胜利和失败的字符串打印到屏幕上，以便在游戏结束时进行比较。
5. 使用 base-6 编码将游戏历史更高效地表示。

总之，这段代码主要实现了 Tic-Tac-Toe 游戏的 base-6 编码表示，以及通过文件输入来读取游戏历史并将其显示在屏幕上的功能。


```
# "learning" tic-tac-toe:
# https://en.wikipedia.org/wiki/Matchbox_Educable_Noughts_and_Crosses_Engine
#
# The base-6 representation allows game history to be VERY efficiently
# represented. I considered whether to rewrite this representation to be
# easier to read, but I elected to TRY to document it, instead.
#
# Another place where I have made a difficult decision between accuracy
# and correctness is inside the "wrapping" code where it considers
# "while human_move_end > 13". The original BASIC code reads:
#
# 830 IF L>13 THEN L=L-14:R=1:GOTO 830
#
# I suspect that the intention is not to assign 1 to R, but to increment
# R. I discuss this more in a porting note comment next to the
```

这段代码是一个文本游戏，源自一本书中提到的井字棋(棋盘游戏)。该游戏的主要目的是让读者更好地理解书中的内容，并提供一种更准确地玩这款游戏的途径。

该代码的作用是提供一个更准确的井字棋游戏，使得读者可以通过运行该代码来更好地理解书中的游戏玩法和策略。这个游戏程序并不是让读者与其他人进行实时的对战，而是让读者更好地理解书中的游戏玩法和策略，以及更好地理解井字棋的规则和技巧。

该代码的具体实现包括两个主要部分：

1. 生成一个棋盘，并在棋盘上初始化了一些起始位置的石头计数器。

2. 允许用户手动输入胜利的计数器值，以及是否使用“赢得比赛”或“更准确的模拟”模式。

3. 处理用户输入并更新游戏状态，使用“赢得比赛”模式则直接显示胜利计数器值，使用“更准确的模拟”模式则根据输入的计数器值更新计数器值。

4. 检查游戏是否结束，如果是，则显示胜利的计数器值并结束游戏。如果不是，则允许用户继续输入计数器值以继续游戏。

该代码的主要目的是提供一个更准确的井字棋游戏，帮助读者更好地理解井字棋的规则和技巧，以及更好地评估书中提到的机器学习方法的有效性。


```
# translated code. If you wish to play a more accurate version of the
# game as written in the book, you can convert the increment back to an
# assignment.
#
# I continue to be impressed with this jewel of a game; as soon as I had
# the AI playing against me, it was beating me. I've been able to score
# a few wins against the computer, but even at its 2-ply lookahead, it
# beats me nearly always. I would like to become better at this game to
# explore the effectiveness of the "losing book" machine learning.
#
#
# EXERCISES FOR THE READER
# One could go many directions with this game:
# - change the initial number of stones in each pit
# - change the number of pits
```

这段代码是一个关于围棋（也称为头脑围棋）AI的实现。它的主要目的是让AI更智能，更具策略性。通过修改现有的游戏规则，使其更具挑战性，从而促使AI在游戏过程中更加谨慎地行动。以下是这段代码的功能解释：

1. 限制只允许玩家在游戏结束时进行捕捉操作，这有助于确保游戏的公平性。
2. 禁止在任何情况下进行捕捉操作，这有助于避免不必要的压力。
3. 不允许玩家在每次行动后将其石子掉入敌人的“家”（指棋盘的中央位置），这有助于确保玩家在每次行动后都有选择。
4. 允许玩家沿顺时针或逆时针方向移动，而不是每次只能向一个方向移动。
5. 通过允许玩家在达到“家”后进行自由移动，使得游戏更具策略性。
6. 增加了AI的审视距离，使其更容易发现玩家可能的行动计划。
7. 通过允许AI使用“分而治之”策略（在棋盘中央放置棋子，然后环绕四周进行观察），提高了其策略性。
8. 将游戏历史存储到文件中，以便AI在后续游戏中学习和适应。随着游戏的进行，AI会逐渐调整其策略，以取得更好的成绩。

总之，这段代码通过修改现有的游戏规则，使得AI在游戏过程中更加智能和策略性。通过限制玩家的行动，以及增加游戏的策略性，使得AI更具挑战性。


```
# - only allow capturing if you end on your side of the board
# - don't allow capturing at all
# - don't drop a stone into the enemy "home"
# - go clockwise, instead
# - allow the player to choose to go clockwise or counterclockwise
# - instead of a maximum of two moves, allow each move that ends on the
#   "home" to be followed by a free move.
# - increase the AI lookahead
# - make the scoring heuristic a little more nuanced
# - store history to a file on disk (or in the cloud!) to allow the AI
#   to learn over more than a single session

from typing import Dict, List, Tuple

game_number: int = 0
```

这段代码定义了一个名为move_count的整数变量，其初始值为0，另外还定义了一个名为losing_book的列表变量，其初始值为空列表([])，以及一个名为n的整数变量，其初始值为0。

接着，定义了一个名为MAX_HISTORY的整数变量，其值为9，以及一个名为LOSING_BOOK_SIZE的整数变量，其值为50。

接下来定义了一个名为draw_pit的函数，该函数接收一个名为line的字符串参数，一个名为board的字符串参数和一个名为 pit_index的整数参数。函数的作用是在board的第pit_index个位置上取得一个整数，并返回该整数对应的字符串。函数中首先取得整数val，然后使用if语句判断val是否小于10，如果是，就执行if语句中的代码，否则直接执行else后面的代码。在if语句中的代码中，首先使用str(val)将val转换成字符串并将其加入line中，然后使用line + str(val)将整数val转换成字符串并将其加入line中，最后使用return将整个line返回。

最后，在程序中没有做任何其他事情，直接进入主程序。


```
move_count: int = 0
losing_book: List[int] = []
n = 0

MAX_HISTORY = 9
LOSING_BOOK_SIZE = 50


def draw_pit(line: str, board, pit_index) -> str:
    val = board[pit_index]
    line = line + " "
    if val < 10:
        line = line + " "
    line = line + str(val) + " "
    return line


```

这段代码定义了一个名为 `draw_board` 的函数，用于在控制台或游戏板上绘制一个 12 行 6 列的迷宫。

函数中包含三个循环，分别用于绘制电脑、侧面和底部的 pit。在每次绘制 pit 时，函数都会输出当前 pit 的位置，并在周围输出一些垂直的线来使 pit 看起来更真实。

电脑 pit 的位置从 12 开始，每隔 1 行递减一次，而侧面 pit 的位置从 13 开始，每隔 2 行递减一次。底部的 pit 则从 0 开始，每隔 1 行递减一次。

通过调用 `draw_board` 函数，可以在控制台或游戏板上绘制出上述的迷宫。


```
def draw_board(board) -> None:
    print()

    # Draw the top (computer) pits
    line = "   "
    for i in range(12, 6, -1):
        line = draw_pit(line, board, i)
    print(line)

    # Draw the side (home) pits
    line = draw_pit("", board, 13)
    line += " " * 24
    line = draw_pit(line, board, 6)
    print(line)

    # Draw the bottom (player) pits
    line = "   "
    for i in range(0, 6):
        line = draw_pit(line, board, i)
    print(line)
    print()
    print()


```

这段代码定义了一个名为 `play_game` 的函数，用于在给定的棋盘上玩井字棋游戏。以下是该函数的功能和流程：

1. 初始化棋盘并放置开始时的小球。
2. 放置玩家的起始位置。
3. 初始化游戏记录，以便在每次游戏结束后统计胜者。
4. 开始游戏流程。
5. 每次游戏开始时，检查棋盘状态，包括检查是否有玩家剩余、开始位置和小球。
6. 如果棋盘状态正常，则进行玩家和电脑的下一步操作。
7. 如果棋盘状态有变化，则输出相应信息并继续游戏流程。
8. 如果游戏结束，则输出游戏结果并结束游戏流程。

该函数中还有一些 global 变量，分别是 `move_count`、`losing_book` 和 `game_number`，分别用于跟踪游戏中的步数、失败者信息和游戏编号。这些 global 变量在该函数中都是初始化的，用于记录整个游戏过程。


```
def play_game(board: List[int]) -> None:
    # Place the beginning stones
    for i in range(0, 13):
        board[i] = 3

    # Empty the home pits
    board[6] = 0
    board[13] = 0

    global move_count
    move_count = 0

    # clear the history record for this game
    losing_book[game_number] = 0

    while True:
        draw_board(board)

        print("YOUR MOVE")
        landing_spot, is_still_going, home = player_move(board)
        if not is_still_going:
            break
        if landing_spot == home:
            landing_spot, is_still_going, home = player_move_again(board)
        if not is_still_going:
            break

        print("MY MOVE")
        landing_spot, is_still_going, home, msg = computer_move("", board)

        if not is_still_going:
            print(msg)
            break
        if landing_spot == home:
            landing_spot, is_still_going, home, msg = computer_move(msg + " , ", board)
        if not is_still_going:
            print(msg)
            break
        print(msg)

    game_over(board)


```

This is a Python implementation of a board game where two players can play against each other. The game is simulated using a artificial neural network that uses a different neural network for the computer player.

The `Board` class represents the game board, which is a 14x14 matrix. It contains the values of each square on the board, as well as some additional information such as the history of the game, the current score, and the quality of the player's move.

The `History` class represents the history of the game, with information about the previous moves made by each player. It contains a list of all the moves made by the player, as well as the time taken to make each move.

The `Player` class represents each player, with information about their current score, the quality of their last move, and whether they have lost the game.

The `Game` class is the main class that controls the game loop. It contains information about the current state of the game, the history of the game, and the quality of the player's moves. It also handles the logic of the game, such as updating the game state, checking for胜者， and printing the game board.

The `Display` function is a simple function that prints the game board to the console.

The `执行func`是一个辅助函数，用于打印出最新的玩家的操作。


```
def computer_move(msg: str, board) -> Tuple[int, bool, int, str]:
    # This function does a two-ply lookahead evaluation; one computer
    # move plus one human move.
    #
    # To do this, it makes a copy (temp_board) of the board, plays
    # each possible computer move and then uses math to work out what
    # the scoring heuristic is for each possible human move.
    #
    # Additionally, if it detects that a potential move puts it on a
    # series of moves that it has recorded in its "losing book", it
    # penalizes that move by two stones.

    best_quality = -99

    # Make a copy of the board, so that we can experiment. We'll put
    # everything back, later.
    temp_board = board[:]

    # For each legal computer move 7-12
    for computer_move in range(7, 13):
        if board[computer_move] == 0:
            continue
        do_move(computer_move, 13, board)  # try the move (1 move lookahead)

        best_player_move_quality = 0
        # for all legal human moves 0-5 (responses to computer move computer_move)
        for human_move_start in range(0, 6):
            if board[human_move_start] == 0:
                continue

            human_move_end = board[human_move_start] + human_move_start
            this_player_move_quality = 0

            # If this move goes around the board, wrap backwards.
            #
            # PORTING NOTE: The careful reader will note that I am
            # incrementing this_player_move_quality for each wrap,
            # while the original code only set it equal to 1.
            #
            # I expect this was a typo or oversight, but I also
            # recognize that you'd have to go around the board more
            # than once for this to be a difference, and even so, it
            # would be a very small difference; there are only 36
            # stones in the game, and going around the board twice
            # requires 24 stones.

            while human_move_end > 13:
                human_move_end = human_move_end - 14
                this_player_move_quality += 1

            if (
                (board[human_move_end] == 0)
                and (human_move_end != 6)
                and (human_move_end != 13)
            ):
                # score the capture
                this_player_move_quality += board[12 - human_move_end]

            if this_player_move_quality > best_player_move_quality:
                best_player_move_quality = this_player_move_quality

        # This is a zero sum game, so the better the human player's
        # move is, the worse it is for the computer player.
        computer_move_quality = board[13] - board[6] - best_player_move_quality

        if move_count < MAX_HISTORY:
            move_digit = computer_move
            if move_digit > 6:
                move_digit = move_digit - 7

            # Calculate the base-6 history representation of the game
            # with this move. If that history is in our "losing book",
            # penalize that move.
            for prev_game_number in range(game_number):
                if losing_book[game_number] * 6 + move_digit == int(
                    losing_book[prev_game_number] / 6 ^ (7 - move_count) + 0.1  # type: ignore
                ):
                    computer_move_quality -= 2

        # Copy back from temporary board
        for i in range(14):
            board[i] = temp_board[i]

        if computer_move_quality >= best_quality:
            best_move = computer_move
            best_quality = computer_move_quality

    selected_move = best_move

    move_str = chr(42 + selected_move)
    if msg:
        msg += ", " + move_str
    else:
        msg = move_str

    move_number, is_still_going, home = execute_move(selected_move, 13, board)

    return move_number, is_still_going, home, msg


```

这段代码是一个函数，名为 `game_over`，其作用是在一个棋盘游戏的基础上，对游戏进行结束处理。

函数体内部先打印输出字符，然后输出“GAME OVER”字符，表明游戏已经结束。

接着，函数体内部计算当前最左下方的方块与最右上方的方块之间的点数差异，如果这个差异为负数，说明自己赢了，否则说明让对方赢了。

如果差异为0，则说明这是一场平局。

另外，通过这个函数，还可以在游戏中增加一个全局变量 `n`，记录当前游戏的轮数。

总结起来，这个函数是为了在棋盘游戏中进行结束处理，确保游戏可以正确结束，并且可以在游戏结束后进行记录和统计。


```
def game_over(board) -> None:
    print()
    print("GAME OVER")

    pit_difference = board[6] - board[13]
    if pit_difference < 0:
        print(f"I WIN BY {-pit_difference} POINTS")

    else:
        global n
        n = n + 1

        if pit_difference == 0:
            print("DRAWN GAME")
        else:
            print(f"YOU WIN BY {pit_difference} POINTS")


```

这是一个Python语言下的函数，定义了两个函数do_capture()和do_move()。

do_capture()函数的功能是移动石头，将其从原来的位置(board[m])移动到一个新的位置(board[12-m])。同时，该函数在执行完移动操作后，将原来移动过的石头(board[m])的值清零，并将新移动的位置(board[12-m])的值清零。

do_move()函数用于根据需要移动石头，具体实现是，获取需要移动的石头的数量(board[m])，将其从board中删除，然后遍历与该数量相邻的位置，如果当前位置与需要移动的石头的数量相邻(包括上下左右)，则将该位置向后或向前移动一个单位，移动后，如果当前位置剩余的空位置数小于需要移动的石头的数量，则递归调用do_capture()函数处理该位置。函数返回移动后剩余的空位置数。


```
def do_capture(m, home, board) -> None:
    board[home] += board[12 - m] + 1
    board[m] = 0
    board[12 - m] = 0


def do_move(m, home, board) -> int:
    move_stones = board[m]
    board[m] = 0

    for _stones in range(move_stones, 0, -1):
        m = m + 1
        if m > 13:
            m = m - 14
        board[m] += 1
    if board[m] == 1 and (m != 6) and (m != 13) and (board[12 - m] != 0):
        do_capture(m, home, board)
    return m


```

这段代码定义了两个函数，`player_has_stones` 和 `computer_has_stones`，它们用于检查游戏棋盘上是否有玩家剩余的方块和电脑当前落下的位置是否在棋盘上。

接着定义了一个 `execute_move` 函数，它接受一个移动（move）选项（home），棋盘（board）和一个整数参数，然后执行移动并返回执行结果。

`execute_move` 函数的实现比较复杂，主要包括以下步骤：

1. 如果移动是 7 以内的数字，则执行移动并返回。
2. 如果移动是 7 或者更大的数字，则使用 home 减去移动得到的结果，并将移动结果加 1，同时将移动计数器加 1，以便存储棋盘状态。
3. 如果棋盘上所有玩家都没有棋子或者移动后的位置不在棋盘上，则判断玩家和电脑当前棋盘状态，如果两个位置的棋盘状态相同，则认为游戏还有继续进行的可能，返回 True；如果两个位置的棋盘状态不同，则认为游戏已经结束，返回 False。


```
def player_has_stones(board) -> bool:
    return any(board[i] > 0 for i in range(6))


def computer_has_stones(board: Dict[int, int]) -> bool:
    return any(board[i] > 0 for i in range(7, 13))


def execute_move(move, home: int, board) -> Tuple[int, bool, int]:
    move_digit = move
    last_location = do_move(move, home, board)

    if move_digit > 6:
        move_digit = move_digit - 7

    global move_count
    move_count += 1
    if move_count < MAX_HISTORY:
        # The computer keeps a chain of moves in losing_book by
        # storing a sequence of moves as digits in a base-6 number.
        #
        # game_number represents the current game,
        # losing_book[game_number] records the history of the ongoing
        # game.  When the computer evaluates moves, it tries to avoid
        # moves that will lead it into paths that have led to previous
        # losses.
        losing_book[game_number] = losing_book[game_number] * 6 + move_digit

    if player_has_stones(board) and computer_has_stones(board):
        is_still_going = True
    else:
        is_still_going = False
    return last_location, is_still_going, home


```

这段代码定义了一个名为 `player_move_again` 的函数，它接受一个名为 `board` 的二维列表作为参数。

函数的作用是调用另一个名为 `player_move` 的函数，这个函数会尝试在 board 上移动，直到它抵达一个有效的位置，然后返回移动后的状态、是否还有移动的可能性以及移动的目标位置。

`player_move` 函数的实现比较复杂，这里简单解释一下：

-它会提示玩家有 6 个选择：1-6
-然后程序会读取玩家的输入，并将其赋值给一个名为 `m` 的变量。
-程序会检查 `m` 是否大于 5 或小于 0，如果是，那么会提示非法移动并继续尝试下一个输入。
-如果 `m` 在有效范围内，程序会尝试使用 `execute_move` 函数执行动作，并获取移动后的位置、是否还有移动的可能性以及移动的目标位置。
-如果 `execute_move` 函数返回没有找到有效的位置或者移动指令失败，那么 `player_move` 函数会继续尝试，直到成功或所有可能的移动都被尝试过。

`draw_board` 函数用于在移动之后清空 board，以便下一轮移动做出更好的决策。


```
def player_move_again(board) -> Tuple[int, bool, int]:
    print("AGAIN")
    return player_move(board)


def player_move(board) -> Tuple[int, bool, int]:
    while True:
        print("SELECT MOVE 1-6")
        m = int(input()) - 1

        if m > 5 or m < 0 or board[m] == 0:
            print("ILLEGAL MOVE")
            continue

        break

    ending_spot, is_still_going, home = execute_move(m, 6, board)

    draw_board(board)

    return ending_spot, is_still_going, home


```

这段代码定义了一个名为 "main" 的函数，它接受一个 None 类型的参数。函数内部执行以下操作：

1. 在屏幕上打印出 " " * 34 + "AWARI "，其中 " " 代表一个空格，34 是一个变量，用于存储在屏幕上打印的单词数量。
2. 在屏幕上打印出 " " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"，同样地，这里的 " " 也是一个变量，用于在屏幕上打印的单词数量。
3. 在屏幕上打印出 " " * 29 + "AWARI "，这里同样使用了变量，打印出 " " * 20 + "Error! "，这里使用了 "Error!"，而不是 "ERROR"。
4. 在一个 while 循环中，内部调用了名为 "play\_game" 的函数，这个函数接受一个列表作为参数，这个列表用于存储游戏的棋盘。
5. 最后，在函数外部，当程序被调用时，它将进入 main 函数。


```
def main() -> None:
    print(" " * 34 + "AWARI")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n")

    board = [0] * 14  # clear the board representation
    global losing_book
    losing_book = [0] * LOSING_BOOK_SIZE  # clear the "machine learning" state

    while True:
        play_game(board)


if __name__ == "__main__":
    main()

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/) by [Alex Scown](https://github.com/TheScown)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Bagels

In this game, the computer picks a 3-digit secret number using the digits 0 to 9 and you attempt to guess what it is. You are allowed up to twenty guesses. No digit is repeated. After each guess the computer will give you clues about your guess as follows:

- PICO    One digit is correct, but in the wrong place
- FERMI    One digit is in the correct place
- BAGELS   No digit is correct

You will learn to draw inferences from the clues and, with practice, you’ll learn to improve your score. There are several good strategies for playing Bagels. After you have found a good strategy, see if you can improve it. Or try a different strategy altogether to see if it is any better. While the program allows up to twenty guesses, if you use a good strategy it should not take more than eight guesses to get any number.

The original authors of this program are D. Resek and P. Rowe of the Lawrence Hall of Science, Berkeley, California.

---

As published in Basic Computer Games (1978)
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=9)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=21)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

(please note any difficulties or challenges in porting here)


# `05_Bagels/csharp/BagelNumber.cs`

This is a class definition for a `BagelNumber` that has a constructor that takes an array of `digits` and implements the `IComparer<BagelNumber, int>` interface.

The `BagelNumber` class has several methods that are relevant to the `IComparer<BagelNumber, int>` interface, including:

* `CompareTo(BagelNumber other)`: This method is used by the `IComparer<BagelNumber, int>` interface to compare two `BagelNumber` objects. It takes in another `BagelNumber` object called `other`, and returns a tuple of two values: the `pico` and `fermi` values of the `BagelNumber` object that is larger than `other`.
* `ToArray()`: This method returns an array of integers that is the same as the `ToArray()` method of the `BagelNumber` class.
* `GetDigits()`: This method returns an array of integers that is the same as the `GetDigits()` method of the `BagelNumber` class.
* `Shuffle()`: This method is used by the `IComparer<BagelNumber, int>` interface to shuffle the digits of a `BagelNumber` object.

Note that this implementation is not optimized and may not be the most efficient in terms of performance.


```
﻿using System;
using System.Collections.Generic;
using System.Linq;

namespace BasicComputerGames.Bagels
{
	public enum BagelValidation
	{
		Valid,
		WrongLength,
		NotUnique,
		NonDigit
	};
	public class BagelNumber
	{
		private static readonly Random Rnd = new Random();

		private readonly int[] _digits;
		public override string ToString()
		{
			return String.Join('-', _digits);
		}

		public static BagelNumber CreateSecretNumber(int numDigits)
		{
			if (numDigits < 3 || numDigits > 9)
				throw new ArgumentOutOfRangeException(nameof(numDigits),
					"Number of digits must be between 3 and 9, inclusive");

			var digits = GetDigits(numDigits);
			return new BagelNumber(digits);
		}



		public static BagelValidation IsValid(string number, int length)
		{
			if (number.Length != length)
				return BagelValidation.WrongLength;

			if (!number.All(Char.IsDigit))
				return BagelValidation.NonDigit;

			if (new HashSet<char>(number).Count != length)
				return BagelValidation.NotUnique;

			return BagelValidation.Valid;
		}

		public BagelNumber(string number)
		{
			if (number.Any(d => !Char.IsDigit(d)))
				throw new ArgumentException("Number must be all unique digits", nameof(number));

			_digits = number.Select(d => d - '0').ToArray();
		}

		//public BagelNumber(long number)
		//{
		//	var digits = new List<int>();
		//	if (number >= 1E10)
		//		throw new ArgumentOutOfRangeException(nameof(number), "Number can be no more than 9 digits");

		//	while (number > 0)
		//	{
		//		long num = number / 10;
		//		int digit = (int)(number - (num * 10));
		//		number = num;
		//		digits.Add(digit);
		//	}

		//	_digits = digits.ToArray();
		//}

		public BagelNumber(int[] digits)
		{
			_digits = digits;
		}

		private static  int[] GetDigits(int numDigits)
		{
			int[] digits = {1, 2, 3, 4, 5, 6, 7, 8, 9};
			Shuffle(digits);
			return digits.Take(numDigits).ToArray();

		}

		private static void Shuffle(int[] digits)
		{
			for (int i = digits.Length - 1; i > 0; --i)
			{
				int pos = Rnd.Next(i);
				var t = digits[i];
				digits[i] = digits[pos];
				digits[pos] = t;
			}

		}

		public (int pico, int fermi) CompareTo(BagelNumber other)
		{
			int pico = 0;
			int fermi = 0;
			for (int i = 0; i < _digits.Length; i++)
			{
				for (int j = 0; j < other._digits.Length; j++)
				{
					if (_digits[i] == other._digits[j])
					{
						if (i == j)
							++fermi;
						else
							++pico;
					}
				}
			}

			return (pico, fermi);
		}
	}
}

```