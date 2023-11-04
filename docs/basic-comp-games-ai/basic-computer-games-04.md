# BasicComputerGames源码解析 4

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.julialang.org/)

Original source downloaded from [Vintage Basic](http://www.vintage-basic.net/games.html).

Conversion to [MiniScript](https://miniscript.org).

Ways to play:

1. Command-Line MiniScript:
Download for your system from https://miniscript.org/cmdline/, install, and then run the program with a command such as:

```
	miniscript dice.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "dice"
	run
```
3. "Try-It!" page on the web:
Go to https://miniscript.org/tryit/, clear the default program from the source code editor, paste in the contents of dice.ms, and click the "Run Script" button.


Please refer to the `readme.md` in the parent folder. 

Each subfolder represents a port of this program to a language which is _not_ one of the agreed upon 10 languages, which are intended to meet these three criteria:

1. Popular (by TIOBE index)
2. Memory safe
3. Generally considered a 'scripting' language

We welcome additional ports, but these additional ports are for educational purposes only.

# `00_Alternate_Languages/34_Digits/go/main.go`

这段代码的主要作用是定义了一个名为`printIntro`的函数，该函数用于在屏幕上打印出一些文本，以便在游戏中向玩家介绍游戏的玩法和规则。

具体来说，这段代码导入了`bufio`、`fmt`、`math/rand`、`os`、`strconv`和`time`等标准库，然后定义了一个`printIntro`函数，该函数使用`fmt`库将一些字符串打印到屏幕上，然后使用`time`库来让文本具有随机化的效果。

`printIntro`函数的主要目的是在玩家启动游戏时显示一些游戏相关的信息和介绍，例如游戏名称、版本、计分方式等。这些信息有助于玩家了解游戏的特点和规则，从而更好地享受游戏。


```
package main

import (
	"bufio"
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"time"
)

func printIntro() {
	fmt.Println("                                DIGITS")
	fmt.Println("              CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
	fmt.Println()
	fmt.Println()
	fmt.Println("THIS IS A GAME OF GUESSING.")
}

```

这段代码定义了一个名为 `readInteger` 的函数，接受一个字符串参数 `prompt`，并返回一个整数类型的值。函数内部使用了一个 `bufio.NewScanner` 来从标准输入（通常是键盘）读取字符串，并逐个字符地将其存储在 `scanner` 变量中。

在 `for` 循环中，函数先输出一个 prompt，然后等待用户从标准输入中读取并存储在 `scanner.Text()` 方法中。接着，函数调用 `strconv.Atoi` 函数将 `scanner.Text()` 方法中的字符串转换为一个整数，并将其存储在 `response` 变量中。

如果转换过程中出现错误，函数将打印一个错误消息并尝试再次读取输入。最后，函数返回 `response`，即用户输入的整数。


```
func readInteger(prompt string) int {
	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Println(prompt)
		scanner.Scan()
		response, err := strconv.Atoi(scanner.Text())

		if err != nil {
			fmt.Println("INVALID INPUT, TRY AGAIN... ")
			continue
		}

		return response
	}
}

```

这段代码定义了两个函数，printInstructions()和readTenNumbers()。

printInstructions()函数的作用是打印一系列的提示信息，告诉用户应该如何按照特定的格式打印数字。这个函数会输出字符串："TAKE A PIECE OF PAPER AND WRITE DOWN\nPLEASE TAKE YOUR DIGITS IN THREE LINES OF TEN DIGITS EACH\nTHIS IS A sample code.\n"

而readTenNumbers()函数的作用是从用户那里获取10个数字，并返回这些数字的切片。这个函数会提示用户输入第一个数字，然后循环获取剩余的9个数字，最终把它们放置在一个长度为10的切片数组中并返回。


```
func printInstructions() {
	fmt.Println()
	fmt.Println("PLEASE TAKE A PIECE OF PAPER AND WRITE DOWN")
	fmt.Println("THE DIGITS '0', '1', OR '2' THIRTY TIMES AT RANDOM.")
	fmt.Println("ARRANGE THEM IN THREE LINES OF TEN DIGITS EACH.")
	fmt.Println("I WILL ASK FOR THEN TEN AT A TIME.")
	fmt.Println("I WILL ALWAYS GUESS THEM FIRST AND THEN LOOK AT YOUR")
	fmt.Println("NEXT NUMBER TO SEE IF I WAS RIGHT. BY PURE LUCK,")
	fmt.Println("I OUGHT TO BE RIGHT TEN TIMES. BUT I HOPE TO DO BETTER")
	fmt.Println("THAN THAT *****")
	fmt.Println()
}

func readTenNumbers() []int {
	numbers := make([]int, 10)

	numbers[0] = readInteger("FIRST NUMBER: ")
	for i := 1; i < 10; i++ {
		numbers[i] = readInteger("NEXT NUMBER:")
	}

	return numbers
}

```

这段代码定义了一个名为 printSummary 的函数，它接受一个整数参数 correct，并输出一系列文字。

函数首先打印一个空行，然后根据 correct 值与 10 的比较，输出不同的消息。如果 correct 大于 10，则输出 "I GUESSED MORE THAN 1/3 OF YOUR NUMBERS。" 和 "I WIN."；如果 correct 小于 10，则输出 "I GUESSED LESS THAN 1/3 OF YOUR NUMBERS." 和 "YOU BEAT ME."；如果 correct 等于 10，则输出 "I GUESSED EXACTLY 1/3 OF YOUR NUMBERS." 和 "IT'S A TIE GAME."。

输出完所有消息后，函数会结束。


```
func printSummary(correct int) {
	fmt.Println()

	if correct > 10 {
		fmt.Println()
		fmt.Println("I GUESSED MORE THAN 1/3 OF YOUR NUMBERS.")
		fmt.Println("I WIN.\u0007")
	} else if correct < 10 {
		fmt.Println("I GUESSED LESS THAN 1/3 OF YOUR NUMBERS.")
		fmt.Println("YOU BEAT ME.  CONGRATULATIONS *****")
	} else {
		fmt.Println("I GUESSED EXACTLY 1/3 OF YOUR NUMBERS.")
		fmt.Println("IT'S A TIE GAME.")
	}
}

```

This appears to be a Go program that is meant to calculate the害羞指数， also known as the Akkroyd index, of a given number. The program takes into account a variety of factors, such as the number of runs completed, the number of runs that were not right, and the number of runs that were incorrect but were correct. The program also takes into account the number of runs that were completed before the user tried again.

The program first sets the害羞指数 to 0 and then loops through all possible runs. In each run, the program calculates the correct number of runs, prints the result, and updates the running corrector counter. If the user tries again, the program prints a summary of the runs and then loops through all runs again.

The program also includes a loop that attempts to determine if the user wants to try again. If the user chooses to try again, the program prints a message and then exit the program.

Overall, the program appears to be well-structured and easy to understand.


```
func buildArray(val, row, col int) [][]int {
	a := make([][]int, row)
	for r := 0; r < row; r++ {
		b := make([]int, col)
		for c := 0; c < col; c++ {
			b[c] = val
		}
		a[r] = b
	}
	return a
}

func main() {
	rand.Seed(time.Now().UnixNano())

	printIntro()
	if readInteger("FOR INSTRUCTIONS, TYPE '1', ELSE TYPE '0' ? ") == 1 {
		printInstructions()
	}

	a := 0
	b := 1
	c := 3

	m := buildArray(1, 27, 3)
	k := buildArray(9, 3, 3)
	l := buildArray(3, 9, 3)

	for {
		l[0][0] = 2
		l[4][1] = 2
		l[8][2] = 2

		z := float64(26)
		z1 := float64(8)
		z2 := 2
		runningCorrect := 0

		var numbers []int
		for round := 1; round <= 4; round++ {
			validNumbers := false
			for !validNumbers {
				numbers = readTenNumbers()
				validNumbers = true
				for _, n := range numbers {
					if n < 0 || n > 2 {
						fmt.Println("ONLY USE THE DIGITS '0', '1', OR '2'.")
						fmt.Println("LET'S TRY AGAIN.")
						validNumbers = false
						break
					}
				}
			}

			fmt.Printf("\n%-14s%-14s%-14s%-14s\n", "MY GUESS", "YOUR NO.", "RESULT", "NO. RIGHT")

			for _, n := range numbers {
				s := 0
				myGuess := 0

				for j := 0; j < 3; j++ {
					s1 := a*k[z2][j] + b*l[int(z1)][j] + c*m[int(z)][j]

					if s < s1 {
						s = s1
						myGuess = j
					} else if s1 == s && rand.Float64() > 0.5 {
						myGuess = j
					}
				}
				result := ""

				if myGuess != n {
					result = "WRONG"
				} else {
					runningCorrect += 1
					result = "RIGHT"
					m[int(z)][n] = m[int(z)][n] + 1
					l[int(z1)][n] = l[int(z1)][n] + 1
					k[int(z2)][n] = k[int(z2)][n] + 1
					z = z - (z/9)*9
					z = 3.0*z + float64(n)
				}
				fmt.Printf("\n%-14d%-14d%-14s%-14d\n", myGuess, n, result, runningCorrect)

				z1 = z - (z/9)*9
				z2 = n
			}
			printSummary(runningCorrect)
			if readInteger("\nDO YOU WANT TO TRY AGAIN (1 FOR YES, 0 FOR NO) ? ") != 1 {
				fmt.Println("\nTHANKS FOR THE GAME.")
				os.Exit(0)
			}
		}
	}
}

```

Original source downloaded from [Vintage Basic](http://www.vintage-basic.net/games.html).

Conversion to [MiniScript](https://miniscript.org).

Ways to play:

1. Command-Line MiniScript:
Download for your system from https://miniscript.org/cmdline/, install, and then run the program with a command such as:

```
	miniscript digits.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "digits"
	run
```

Please refer to the `readme.md` in the parent folder. 

Each subfolder represents a port of this program to a language which is _not_ one of the agreed upon 10 languages, which are intended to meet these three criteria:

1. Popular (by TIOBE index)
2. Memory safe
3. Generally considered a 'scripting' language

We welcome additional ports, but these additional ports are for educational purposes only.

# `00_Alternate_Languages/35_Even_Wins/go/evenwins.go`

这段代码定义了一个名为 "PlayerType" 的枚举类型，表示玩家的类型，包括 HUMAN 和 COMPUTER 两种。同时，还定义了一个名为 "MAXTAKE" 的常量，表示允许玩家在每一轮游戏中最多可以抽取多少卡牌。

接下来，定义了一个名为 "Player" 的结构体，用于表示每个玩家的信息，包括他们的 ID、抽卡次数以及卡牌类型的组合。在结构体中，使用了两个整型变量 "id" 和 "turn" 来表示玩家 ID 和当前轮到的玩家。

然后，通过一个名为 "fmt" 的函数输出了一行字符串，包含了游戏的名称和版本信息。

接着，定义了一个名为 "抽卡" 的函数，用于从玩家手牌中抽取一张卡牌，并输出玩家 ID 和抽到的卡牌类型。

在 "抽卡" 函数中，使用了两个循环来遍历玩家手牌和卡牌库存。第一个循环从玩家手牌中抽取一张卡牌，第二个循环从卡牌库存中检索该卡牌所属的玩家。

在 "抽卡" 函数中，还使用了一个名为 "PlayerType" 的变量，用于表示当前玩家是 HUMAN 还是 COMPUTER。在 "MAXTAKE" 常量中，使用了变量 "turn" 来表示当前轮到的玩家。

最后，通过一个名为 "main" 的函数来进入游戏的主循环。在主循环中，首先让玩家输入游戏名称和版本信息，然后初始化玩家手牌和卡牌库存。接着，让玩家抽取卡牌，并显示玩家 ID 和抽到的卡牌类型。


```
package main

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

const MAXTAKE = 4

type PlayerType int8

const (
	HUMAN PlayerType = iota
	COMPUTER
)

```

这段代码定义了一个名为 "Game" 的结构体，其中包含一个整数类型的 table、一个整数类型的 human 和一个整数类型的 computer。

接下来，定义了一个名为 "NewGame" 的函数，该函数返回一个名为 "Game" 的实例化对象。

在 "NewGame" 函数内部，创建了一个空白的 Game 对象，然后将其 table 成员设置为 27。

接着，定义了一个名为 "printIntro" 的函数，该函数用于输出一段游戏介绍信息。

最后，在 "printIntro" 函数内部，使用格式化字符串Println() 函数输出了一些关于游戏规则的信息，包括游戏的参与者、游戏规则以及游戏结束的条件。


```
type Game struct {
	table    int
	human    int
	computer int
}

func NewGame() Game {
	g := Game{}
	g.table = 27

	return g
}

func printIntro() {
	fmt.Println("Welcome to Even Wins!")
	fmt.Println("Based on evenwins.bas from Creative Computing")
	fmt.Println()
	fmt.Println("Even Wins is a two-person game. You start with")
	fmt.Println("27 marbles in the middle of the table.")
	fmt.Println()
	fmt.Println("Players alternate taking marbles from the middle.")
	fmt.Println("A player can take 1 to 4 marbles on their turn, and")
	fmt.Println("turns cannot be skipped. The game ends when there are")
	fmt.Println("no marbles left, and the winner is the one with an even")
	fmt.Println("number of marbles.")
	fmt.Println()
}

```

这两段代码定义了一个名为Game的类，该类包含一个名为printBoard的函数和一个名为gameOver的函数。

printBoard函数用于打印游戏棋盘的状态，并输出其中所拥有的棋子数量，包括玩家和电脑各自拥有的棋子数量。

gameOver函数用于判断游戏是否结束，如果玩家的棋子数量为奇数，则认为玩家获胜，否则电脑获胜。函数会先输出一段信息，然后输出一条消息，告诉玩家游戏是否结束，并决定胜负。


```
func (g *Game) printBoard() {
	fmt.Println()
	fmt.Printf(" marbles in the middle: %d\n", g.table)
	fmt.Printf("    # marbles you have: %d\n", g.human)
	fmt.Printf("# marbles computer has: %d\n", g.computer)
	fmt.Println()
}

func (g *Game) gameOver() {
	fmt.Println()
	fmt.Println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
	fmt.Println("!! All the marbles are taken: Game Over!")
	fmt.Println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
	fmt.Println()
	g.printBoard()
	if g.human%2 == 0 {
		fmt.Println("You are the winner! Congratulations!")
	} else {
		fmt.Println("The computer wins: all hail mighty silicon!")
	}
	fmt.Println()
}

```

该代码定义了两个函数。第一个函数 `getPlural` 接受一个整数参数 `count`, 返回一个字符串，表示 `count` 可爱的动物。第二个函数 `humanTurn` 接受一个 `Game` 对象，实现了 ` human` 玩家的回合操作。以下是该函数的实现细节：

1. 从标准输入(通常是键盘)读取一行字符。
2. 定义 `maxAvailable` 变量，其值限制玩家的可选择物品的最大数量。
3. 如果 `g.table` 的值小于 `maxAvailable`，将 `maxAvailable` 设置为 `g.table` 的值。
4. 打印 "It's your turn!"。
5. 循环让玩家输入选择，每次最多可以输入 `maxAvailable` 个字符。
6. 如果玩家输入的数字不正确，或者玩家输入的数量超过物品数量，给出错误提示并结束游戏。
7. 如果玩家输入的数量正确，将 `g.table` 减小 `maxAvailable` 个物品，并将 `g.human` 增加 `maxAvailable` 个物品。
8. 返回游戏中玩家的人性化操作。


```
func getPlural(count int) string {
	m := "marble"
	if count > 1 {
		m += "s"
	}
	return m
}

func (g *Game) humanTurn() {
	scanner := bufio.NewScanner(os.Stdin)
	maxAvailable := MAXTAKE
	if g.table < MAXTAKE {
		maxAvailable = g.table
	}

	fmt.Println("It's your turn!")
	for {
		fmt.Printf("Marbles to take? (1 - %d) --> ", maxAvailable)
		scanner.Scan()
		n, err := strconv.Atoi(scanner.Text())
		if err != nil {
			fmt.Printf("\n  Please enter a whole number from 1 to %d\n", maxAvailable)
			continue
		}
		if n < 1 {
			fmt.Println("\n  You must take at least 1 marble!")
			continue
		}
		if n > maxAvailable {
			fmt.Printf("\n  You can take at most %d %s\n", maxAvailable, getPlural(maxAvailable))
			continue
		}
		fmt.Printf("\nOkay, taking %d %s ...\n", n, getPlural(n))
		g.table -= n
		g.human += n
		return
	}
}

```

这段代码定义了一个名为 computerTurn 的函数，接收一个名为 g 的 Game 对象作为参数。这个函数的主要作用是控制计算机在游戏中的行动。

函数内部首先计算出计算机需要取走的棋子数量，这个数量基于棋盘大小和当前棋子数量的比例关系。接着，根据棋盘大小和当前棋子数量，判断计算机需要取走的棋子数量是整数还是浮点数。如果是整数，则说明计算机需要取走的棋子数量是原来的两倍，否则就是原来的数量减一。

接下来，函数会输出计算机需要取走的棋子数量，并使用 getPlural 函数来获取棋子数量的数量级（比如 "个"、"辆" 等）。最后，函数会将棋盘大小减去需要取走的棋子数量，并将计算机的计数增加需要取走的棋子数量。


```
func (g *Game) computerTurn() {
	marblesToTake := 0

	fmt.Println("It's the computer's turn ...")
	r := float64(g.table - 6*int((g.table)/6))

	if int(g.human/2) == g.human/2 {
		if r < 1.5 || r > 5.3 {
			marblesToTake = 1
		} else {
			marblesToTake = int(r - 1)
		}
	} else if float64(g.table) < 4.2 {
		marblesToTake = 4
	} else if r > 3.4 {
		if r < 4.7 || r > 3.5 {
			marblesToTake = 4
		}
	} else {
		marblesToTake = int(r + 1)
	}

	fmt.Printf("Computer takes %d %s ...\n", marblesToTake, getPlural(marblesToTake))
	g.table -= marblesToTake
	g.computer += marblesToTake
}

```

这段代码是一个函数，名为“play”，接受一个名为“g”的游戏对象作为参数，并且传入一个名为“playersTurn”的整数类型参数。

函数内部先打印游戏板的当前状态，然后进行一个无限循环。在循环中，根据当前轮到玩家发球还是电脑发球来决定下一步的操作。

如果是玩家发球，则会调用函数内部的一个名为“humanTurn”的函数，并且打印游戏板并发送给玩家。接着再次调用“printBoard”函数打印游戏板，然后将轮到玩家发球的标识“playersTurn”改为“comp的象征”。

如果是电脑发球，则会调用函数内部的一个名为“computerTurn”的函数，并且打印游戏板并发送给电脑。接着再次调用“printBoard”函数打印游戏板，然后将轮到玩家发球的标识“playersTurn”改为“HUMAN”。


```
func (g *Game) play(playersTurn PlayerType) {
	g.printBoard()

	for {
		if g.table == 0 {
			g.gameOver()
			return
		} else if playersTurn == HUMAN {
			g.humanTurn()
			g.printBoard()
			playersTurn = COMPUTER
		} else {
			g.computerTurn()
			g.printBoard()
			playersTurn = HUMAN
		}
	}
}

```

这段代码定义了一个名为 `getFirstPlayer` 的函数，它接受一个 `PlayerType` 类型的参数。函数使用 `bufio.NewScanner` 函数从标准输入(通常是键盘)读取输入。

函数内部使用一个 for 循环，循环多次向用户询问是否想先手。每次循环，函数首先输出 "Do you want to play first? (y/n) --> "，然后等待用户从标准输入中读取输入。

如果用户从标准输入中读取的字符串以大写音调开头，函数将返回 `HUMAN` 类型。如果字符串以大写音调开头，函数将返回 `COMPUTER` 类型。如果用户没有在输入中输入任何有效的字符，函数将输出并提示用户重新输入。


```
func getFirstPlayer() PlayerType {
	scanner := bufio.NewScanner(os.Stdin)

	for {
		fmt.Println("Do you want to play first? (y/n) --> ")
		scanner.Scan()

		if strings.ToUpper(scanner.Text()) == "Y" {
			return HUMAN
		} else if strings.ToUpper(scanner.Text()) == "N" {
			return COMPUTER
		} else {
			fmt.Println()
			fmt.Println("Please enter 'y' if you want to play first,")
			fmt.Println("or 'n' if you want to play second.")
			fmt.Println()
		}
	}
}

```

这段代码的作用是实现了一个简单的文本游戏，名为 "Guess the Number"，其中玩家需要猜测一个随机生成的整数，每次猜测后，程序会告诉玩家是否再玩一次，直到玩家回答 "Y" 表示再玩，否则结束游戏并返回。

具体来说，代码中首先定义了一个名为 "main" 的函数，该函数接受一个空字符串作为参数，表示输入结束。函数内部创建了一个 "bufio.NewScanner" 实例，用于从标准输入（通常是键盘输入）中读取玩家输入的整数。

接着，代码中使用一个 for 循环，该循环会不断地创建一个新的 "GuessGame" 对象，并调用该对象的 "play" 方法来让玩家输入猜测的数字。每次循环结束后，程序会提示玩家是否要再次玩游戏，如果玩家输入 "y"，则表示再玩一次，否则结束游戏并返回。

在循环内部，程序还会输出一段欢迎消息并等待玩家的输入。当玩家猜测完毕后，程序会根据玩家的输入提示是否再玩一次。如果玩家再次输入 "y"，则程序会再次输出欢迎消息并循环等待玩家输入。如果玩家不再输入 "y"，则程序会结束游戏并返回。


```
func main() {
	scanner := bufio.NewScanner(os.Stdin)

	printIntro()

	for {
		g := NewGame()

		g.play(getFirstPlayer())

		fmt.Println("\nWould you like to play again? (y/n) --> ")
		scanner.Scan()
		if strings.ToUpper(scanner.Text()) == "Y" {
			fmt.Println("\nOk, let's play again ...")
		} else {
			fmt.Println("\nOk, thanks for playing ... goodbye!")
			return
		}

	}

}

```

Original source downloaded from [Vintage Basic](http://www.vintage-basic.net/games.html).

Conversion to [MiniScript](https://miniscript.org).

Note that this folder (like the original BASIC programs) contains TWO different programs based on the same idea.  evenwins.ms plays deterministically; gameofevenwins.ms learns from its failures over multiple games.

Ways to play:

1. Command-Line MiniScript:
Download for your system from https://miniscript.org/cmdline/, install, and then run the program with a command such as:

```
	miniscript evenwins.ms
```
or

```
	miniscript gameofevenwins.ms
```

2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "evenwins"
	run
```
or

```
	load "gameofevenwins"
	run
```

Please refer to the `readme.md` in the parent folder. 

Each subfolder represents a port of this program to a language which is _not_ one of the agreed upon 10 languages, which are intended to meet these three criteria:

1. Popular (by TIOBE index)
2. Memory safe
3. Generally considered a 'scripting' language

We welcome additional ports, but these additional ports are for educational purposes only.

Original source downloaded from [Vintage Basic](http://www.vintage-basic.net/games.html).

Conversion to [MiniScript](https://miniscript.org).

Ways to play:

1. Command-Line MiniScript:
Download for your system from https://miniscript.org/cmdline/, install, and then run the program with a command such as:

```
	miniscript flipflop.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "flipflop"
	run
```

Please refer to the `readme.md` in the parent folder. 

Each subfolder represents a port of this program to a language which is _not_ one of the agreed upon 10 languages, which are intended to meet these three criteria:

1. Popular (by TIOBE index)
2. Memory safe
3. Generally considered a 'scripting' language

We welcome additional ports, but these additional ports are for educational purposes only.

Original source downloaded from [Vintage Basic](http://www.vintage-basic.net/games.html).

Conversion to [MiniScript](https://miniscript.org).

Ways to play:

1. Command-Line MiniScript:
Download for your system from https://miniscript.org/cmdline/, install, and then run the program with a command such as:

```
	miniscript football.ms
```or
```
	miniscript ftball.ms
```
2. Mini Micro:
Download Mini Micro from https://miniscript.org/MiniMicro/, launch, and then click the top disk slot and chose "Mount Folder..."  Select the folder containing the MiniScript program and this README file.  Then, at the Mini Micro command prompt, enter:

```
	load "football"
	run
```or
```
	load "ftball"
	run
```

#### Apology from the Translator

These MiniScript programs were actually ported from the JavaScript ports of the original BASIC programs.  I did that because the BASIC code (of both programs) was incomprehensible spaghetti.  The JavaScript port, however, was essentially the same — and so are the MiniScript ports.  The very structure of these programs makes them near-impossible to untangle.

If I were going to write a football simulation from scratch, I would approach it very differently.  But in that case I would have either a detailed specification of how the program should behave, or at least enough understanding of American football to design it myself as I go.  Neither is the case here (and we're supposed to be porting the original programs, not making up our own).

So, I'm sorry.  Please take these programs as proof that you can write bad code even in the most simple, elegant languages.  And I promise to try harder on future translations!


Please refer to the `readme.md` in the parent folder. 

Each subfolder represents a port of this program to a language which is _not_ one of the agreed upon 10 languages, which are intended to meet these three criteria:

1. Popular (by TIOBE index)
2. Memory safe
3. Generally considered a 'scripting' language

We welcome additional ports, but these additional ports are for educational purposes only.

# `00_Alternate_Languages/38_Fur_Trader/c/furtrader.c`

This code is a C implementation of a simple "print your name" program that prompts the user to enter their name and prints it. The program is first ported from a BSR-style program called "furtrader.bas" to ANSI C (C99) standard.

The program then includes several standard input/output library functions like `printf()`, `scanf()`, `strlen()`, and `time.h`. These functions are likely used to read input from the user, store the input in a variable, and display the input using `printf()`.

The program also includes a few custom functions:

1. `print_name()`: This function takes two arguments: the name entered by the user and the current time. It uses the current time to format the user's name and prints it using `printf()`.
2. `main()`: This is the starting point of the program. It is called when the program is run.

The `main()` function is responsible for running the program. It first prompts the user to enter their name using `printf()` and then enters a loop that calls the `print_name()` function with the user's name and the current time, formatting and printing it.

In summary, this code is a simple program that prompts the user to enter their name and prints it using a custom function `print_name()`.


```

/*
 * Ported from furtrader.bas to ANSI C (C99) by krt@krt.com.au
 *
 * compile with:
 *    gcc -g -Wall -Werror furtrader.c -o furtrader
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>


/* Constants */
```

这段代码定义了一系列常量，包括 fur 类型计数器、 fur 动物名称常量数组和 fort 类型计数器、fort 动物名称常量数组。然后，定义了一些变量，包括 fur 类型名称数组和 fort 类型名称数组。最后，给 fur 类型名称数组和 fort 类型名称数组赋值，使得可以通过下标访问它们的元素。


```
#define FUR_TYPE_COUNT    4
#define FUR_MINK          0
#define FUR_BEAVER        1
#define FUR_ERMINE        2
#define FUR_FOX           3
#define MAX_FURS        190
const char *FUR_NAMES[FUR_TYPE_COUNT] = { "MINK", "BEAVER", "ERMINE", "FOX" };

#define FORT_TYPE_COUNT 3
#define FORT_MONTREAL   1
#define FORT_QUEBEC     2
#define FORT_NEWYORK    3
const char *FORT_NAMES[FORT_TYPE_COUNT] = { "HOCHELAGA (MONTREAL)", "STADACONA (QUEBEC)", "NEW YORK" };



```

这两段代码定义了两个函数 printAtColumn 和 print。printAtColumn 函数接受一个整数 column 和一个字符串 words，并输出 words 中从 column 开始的位置。print 函数接受一个字符串 words，并输出其中所有单词。

printAtColumn 函数的作用是打印 words 中每个单词的第一行，然后输出一个换行符。print 函数的作用是打印 words 中的所有单词。


```
/* Print the words at the specified column */
void printAtColumn( int column, const char *words )
{
    int i;
    for ( i=0; i<column; i++ )
        printf( " " );
    printf( "%s\n", words );
}

/* trivial function to output a line with a \n */
void print( const char *words )
{
    printf( "%s\n", words );
}

```

这段代码是一个C语言函数，名为`showIntroduction`。其作用是向玩家显示一段介绍性的信息。这段信息包括一个法语句子、一个小时、一个星期和一个 fort。

具体来说，这段信息告诉玩家他们将在1776年离开伊利诺伊湖地区进行一次法语贸易探险活动，他们可以选择三个堡垒进行贸易。贸易商品包括供应商提供的商品和玩家可以用 fur 换取的商品。这段信息还提到了商品的价格和数量取决于玩家选择的堡垒。


```
/* Show the player the introductory message */
void showIntroduction()
{
    print( "YOU ARE THE LEADER OF A FRENCH FUR TRADING EXPEDITION IN " );
    print( "1776 LEAVING THE LAKE ONTARIO AREA TO SELL FURS AND GET" );
    print( "SUPPLIES FOR THE NEXT YEAR.  YOU HAVE A CHOICE OF THREE" );
    print( "FORTS AT WHICH YOU MAY TRADE.  THE COST OF SUPPLIES" );
    print( "AND THE AMOUNT YOU RECEIVE FOR YOUR FURS WILL DEPEND" );
    print( "ON THE FORT THAT YOU CHOOSE." );
    print( "" );
}


/*
 * Prompt the user for input.
 * When input is given, try to conver it to an integer
 * return the integer converted value or 0 on error
 */
```

这段代码的作用是获取用户输入的数字，并将其存储在变量中。它使用了 C 语言和 C 库函数，主要通过以下几个步骤来实现：

1. 读取用户从标准输入(通常是键盘)输入的一行字符串，并将其存储在一个字符数组中(存储在变量 buffer 中)。
2. 使用一个指向字符串结束标记的指针变量 endstr  来定位字符串的结束位置(即不包含在字符串中的最后一个字符)，然后使用 strtol 函数将字符串转换为整数类型，并将其存储在变量 result 中。
3. 如果转换成功，即读取的字符串可以转换为整数，那么将 result 设置为正确的整数，否则将 result 设置为 -1。

该代码的输出结果将是一个整数类型，即用户输入的数字。


```
int getNumericInput()
{
    int  result = -1;
    char buffer[64];   /* somewhere to store user input */
    char *endstr;

    while ( result == -1 )
    {
        printf( ">> " );                                 /* prompt the user */
        fgets( buffer, sizeof( buffer ), stdin );        /* read from the console into the buffer */
        result = (int)strtol( buffer, &endstr, 10 );     /* only simple error checking */

        if ( endstr == buffer )                          /* was the string -> integer ok? */
            result = -1;
    }

    return result;
}


```

这段代码的作用是询问用户是否回答了 "YES" 或 "NO"，如果用户回答了 "YES" 或 "NO"，则尝试将输入转换为一个大写单字母 "Y" 或 "N"。如果无法确定用户回答，则返回 "NO"。

代码中使用了两个循环来读取用户输入。第一个循环用于显示 "ANSWER YES OR NO"，并将用户输入存储在缓冲区中。第二个循环用于从缓冲区中读取用户输入，并将其存储在结果变量中。

如果缓冲区中的第一个字符是 "Y" 或 "N"，则将结果变量设置为相应的字符。否则，如果缓冲区中的第一个字符是其他字符，则将结果变量设置为初始字符 "!"。

最终，代码将返回用户输入的字符，以 "Y" 或 "N" 的形式输出。


```
/*
 * Prompt the user for YES/NO input.
 * When input is given, try to work out if it's YES, Yes, yes, Y, etc.
 * And convert to a single upper-case letter
 * Returns a character of 'Y' or 'N'.
 */
char getYesOrNo()
{
    char result = '!';
    char buffer[64];   /* somewhere to store user input */

    while ( !( result == 'Y' || result == 'N' ) )       /* While the answer was not Yes or No */
    {
        print( "ANSWER YES OR NO" );
        printf( ">> " );

        fgets( buffer, sizeof( buffer ), stdin );            /* read from the console into the buffer */
        if ( buffer[0] == 'Y' || buffer[0] == 'y' )
            result = 'Y';
        else if ( buffer[0] == 'N' || buffer[0] == 'n' )
            result = 'N';
    }

    return result;
}



```

这段代码的作用是向玩家显示Fort的位置，并获取他们的选择。如果玩家的输入是有效的选择（1，2，3），则返回选择，否则继续提示玩家。在循环中，代码会输出Fort的位置和相关的保护措施，并询问玩家选择一个或多个Fort位置。如果玩家选择的有效，则返回选择，否则继续等待玩家的输入。


```
/*
 * Show the player the choices of Fort, get their input, if the
 * input is a valid choice (1,2,3) return it, otherwise keep
 * prompting the user.
 */
int getFortChoice()
{
    int result = 0;

    while ( result == 0 )
    {
        print( "" );
        print( "YOU MAY TRADE YOUR FURS AT FORT 1, FORT 2," );
        print( "OR FORT 3.  FORT 1 IS FORT HOCHELAGA (MONTREAL)" );
        print( "AND IS UNDER THE PROTECTION OF THE FRENCH ARMY." );
        print( "FORT 2 IS FORT STADACONA (QUEBEC) AND IS UNDER THE" );
        print( "PROTECTION OF THE FRENCH ARMY.  HOWEVER, YOU MUST" );
        print( "MAKE A PORTAGE AND CROSS THE LACHINE RAPIDS." );
        print( "FORT 3 IS FORT NEW YORK AND IS UNDER DUTCH CONTROL." );
        print( "YOU MUST CROSS THROUGH IROQUOIS LAND." );
        print( "ANSWER 1, 2, OR 3." );

        result = getNumericInput();   /* get input from the player */
    }

    return result;
}


```

这段代码是一个函数，名为 `showFortComment`，其作用是输出指定堡垒（Fort）的描述。它根据选择的堡垒类型不同，输出不同的描述。如果选择的堡垒不存在的，函数会输出一个错误消息并退出。


```
/*
 * Print the description for the fort
 */
void showFortComment( int which_fort )
{
    print( "" );
    if ( which_fort == FORT_MONTREAL )
    {
        print( "YOU HAVE CHOSEN THE EASIEST ROUTE.  HOWEVER, THE FORT" );
        print( "IS FAR FROM ANY SEAPORT.  THE VALUE" );
        print( "YOU RECEIVE FOR YOUR FURS WILL BE LOW AND THE COST" );
        print( "OF SUPPLIES HIGHER THAN AT FORTS STADACONA OR NEW YORK." );
    }
    else if ( which_fort == FORT_QUEBEC )
    {
        print( "YOU HAVE CHOSEN A HARD ROUTE.  IT IS, IN COMPARSION," );
        print( "HARDER THAN THE ROUTE TO HOCHELAGA BUT EASIER THAN" );
        print( "THE ROUTE TO NEW YORK.  YOU WILL RECEIVE AN AVERAGE VALUE" );
        print( "FOR YOUR FURS AND THE COST OF YOUR SUPPLIES WILL BE AVERAGE." );
    }
    else if ( which_fort == FORT_NEWYORK )
    {
        print( "YOU HAVE CHOSEN THE MOST DIFFICULT ROUTE.  AT" );
        print( "FORT NEW YORK YOU WILL RECEIVE THE HIGHEST VALUE" );
        print( "FOR YOUR FURS.  THE COST OF YOUR SUPPLIES" );
        print( "WILL BE LOWER THAN AT ALL THE OTHER FORTS." );
    }
    else
    {
        printf( "Internal error #1, fort %d does not exist\n", which_fort );
        exit( 1 );  /* you have a bug */
    }
    print( "" );
}


```

这段代码的作用是要求玩家输入狐狸皮、羊皮、马皮和狗皮的数量，如果输入的值不正确，会重新提示玩家输入。然后，程序会统计玩家选择的皮毛种类及其数量，并将它们存储在furs数组中。


```
/*
 * Prompt the player for how many of each fur type they want.
 * Accept numeric inputs, re-prompting on incorrect input values
 */
void getFursPurchase( int *furs )
{
    int i;

    printf( "YOUR %d FURS ARE DISTRIBUTED AMONG THE FOLLOWING\n", FUR_TYPE_COUNT );
    print( "KINDS OF PELTS: MINK, BEAVER, ERMINE AND FOX." );
    print( "" );

    for ( i=0; i<FUR_TYPE_COUNT; i++ )
    {
        printf( "HOW MANY %s DO YOU HAVE\n", FUR_NAMES[i] );
        furs[i] = getNumericInput();
    }
}


```

这段代码有两个主要的作用：

1. 将玩家皮毛库存的值设为零，即使用i从0到FUR_TYPE_COUNT-1遍历，将每个皮毛库存的值设为0。
2. 统计玩家皮毛库存的总量，即使用i从0到FUR_TYPE_COUNT-1遍历，将每个皮毛库存的值相加，然后使用FUR_TYPE_COUNT-1作为循环变量，将总和赋值给一个变量count。


```
/*
 * (Re)Set the player's inventory to zero
 */
void zeroInventory( int *player_fur_count )
{
    int i;
    for ( i=0; i<FUR_TYPE_COUNT; i++ )
    {
        player_fur_count[i] = 0;
    }
}


/*
 * Tally the player's inventory
 */
```

这段代码定义了一个名为`sumInventory`的函数，其功能是计算玩家毛发库存的总数。函数接受一个整数类型的指针`player_fur_count`作为参数，然后对库存中的每一种类型的毛发进行累加，最后返回毛发库存总数。

该函数使用了嵌套的循环来遍历所有的毛发类型。外层循环变量`i`用于跟踪库存中毛发的种类数，内层循环变量`player_fur_count[i]`用于存储每个毛发类型的数量。在循环体内，将当前类型的毛发数量`player_fur_count[i]`累加到结果变量`result`中。

函数的另一个部分定义了一个名为`randomNumber`的函数，它接受两个整数类型的参数`a`和`b`，然后返回一个介于它们之间的随机整数。这个随机整数的范围在`a`和`b`之间，包括`a`和`b`本身。


```
int sumInventory( int *player_fur_count )
{
    int result = 0;
    int i;
    for ( i=0; i<FUR_TYPE_COUNT; i++ )
    {
        result += player_fur_count[i];
    }

    return result;
}


/*
 * Return a random number between a & b
 * Ref: https://stackoverflow.com/a/686376/1730895
 */
```

这段代码定义了两个函数，一个用于生成随机浮点数，另一个用于生成随机整数。

另外，还定义了三个常量，用于标识游戏中的不同状态。

函数randomAB接受两个float类型的参数a和b，并返回一个float类型的结果。函数使用rand()函数生成一个0到1之间的随机数，然后将其乘以(b-a)，再加上a，最终得到一个0到1之间的随机数。

函数randFloat()与randomAB类似，但使用rand()函数生成一个0到1之间的随机数。

定义的三个常量STATE_STARTING、STATE_CHOOSING_FORT和STATE_TRAVELLING分别表示游戏中的三种状态，可以在游戏主循环中切换。


```
float randomAB(float a, float b)
{
    return ((b - a) * ((float)rand() / (float)RAND_MAX)) + a;
}
/* Random floating point number between 0 and 1 */
float randFloat()
{
    return randomAB( 0, 1 );
}


/* States to allow switching in main game-loop */
#define STATE_STARTING      1
#define STATE_CHOOSING_FORT 2
#define STATE_TRAVELLING    3
```

This appears to be a game of fort construction where the player is given a list of furs (beaver, fox, ermine, and mink) and a starting position for these furs. The player must then build a fort using these furs and keep track of their savings. The game has different levels of difficulty, and the player will have to deal with errors if they use the wrong fort or don't build a valid fort. The game ends when the player decides to trade their furs or when they have saved enough money to leave.


```
#define STATE_TRADING       4

int main( void )
{
    /* variables for storing player's status */
    float player_funds = 0;                              /* no money */
    int   player_furs[FUR_TYPE_COUNT]  = { 0, 0, 0, 0 }; /* no furs */

    /* player input holders */
    char  yes_or_no;
    int   event_picker;
    int   which_fort;

    /* what part of the game is in play */
    int   game_state = STATE_STARTING;

    /* commodity prices */
    float mink_price   = -1;
    float beaver_price = -1;
    float ermine_price = -1;
    float fox_price    = -1;  /* sometimes this takes the "last" price (probably this was a bug) */

    float mink_value;
    float beaver_value;
    float ermine_value;
    float fox_value;      /* for calculating sales results */


    srand( time( NULL ) );  /* seed the random number generator */

    printAtColumn( 31, "FUR TRADER" );
    printAtColumn( 15, "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY" );
    printAtColumn( 15, "(Ported to ANSI-C Oct 2012 krt@krt.com.au)" );
    print( "\n\n\n" );

    /* Loop forever until the player asks to quit */
    while ( 1 )
    {
        if ( game_state == STATE_STARTING )
        {
            showIntroduction();

            player_funds = 600;            /* Initial player start money */
            zeroInventory( player_furs );  /* Player fur inventory */

            print( "DO YOU WISH TO TRADE FURS?" );
            yes_or_no = getYesOrNo();
            if ( yes_or_no == 'N' )
                exit( 0 );                 /* STOP */
            game_state = STATE_TRADING;
        }

        else if ( game_state == STATE_TRADING )
        {
            print( "" );
            printf( "YOU HAVE $ %1.2f IN SAVINGS\n", player_funds );
            printf( "AND %d FURS TO BEGIN THE EXPEDITION\n", MAX_FURS );
            getFursPurchase( player_furs );

            if ( sumInventory( player_furs ) > MAX_FURS )
            {
                print( "" );
                print( "YOU MAY NOT HAVE THAT MANY FURS." );
                print( "DO NOT TRY TO CHEAT.  I CAN ADD." );
                print( "YOU MUST START AGAIN." );
                print( "" );
                game_state = STATE_STARTING;   /* T/N: Wow, harsh. */
            }
            else
            {
                game_state = STATE_CHOOSING_FORT;
            }
        }

        else if ( game_state == STATE_CHOOSING_FORT )
        {
            which_fort = getFortChoice();
            showFortComment( which_fort );
            print( "DO YOU WANT TO TRADE AT ANOTHER FORT?" );
            yes_or_no = getYesOrNo();
            if ( yes_or_no == 'N' )
                game_state = STATE_TRAVELLING;
        }

        else if ( game_state == STATE_TRAVELLING )
        {
            print( "" );
            if ( which_fort == FORT_MONTREAL )
            {
                mink_price   = ( ( 0.2 * randFloat() + 0.70 ) * 100 + 0.5 ) / 100;
                ermine_price = ( ( 0.2 * randFloat() + 0.65 ) * 100 + 0.5 ) / 100;
                beaver_price = ( ( 0.2 * randFloat() + 0.75 ) * 100 + 0.5 ) / 100;
                fox_price    = ( ( 0.2 * randFloat() + 0.80 ) * 100 + 0.5 ) / 100;

                print( "SUPPLIES AT FORT HOCHELAGA COST $150.00." );
                print( "YOUR TRAVEL EXPENSES TO HOCHELAGA WERE $10.00." );
                player_funds -= 160;
            }

            else if ( which_fort == FORT_QUEBEC )
            {
                mink_price   = ( ( 0.30 * randFloat() + 0.85 ) * 100 + 0.5 ) / 100;
                ermine_price = ( ( 0.15 * randFloat() + 0.80 ) * 100 + 0.5 ) / 100;
                beaver_price = ( ( 0.20 * randFloat() + 0.90 ) * 100 + 0.5 ) / 100;
                fox_price    = ( ( 0.25 * randFloat() + 1.10 ) * 100 + 0.5 ) / 100;
                event_picker = ( 10 * randFloat() ) + 1;

                if ( event_picker <= 2 )
                {
                    print( "YOUR BEAVER WERE TOO HEAVY TO CARRY ACROSS" );
                    print( "THE PORTAGE.  YOU HAD TO LEAVE THE PELTS, BUT FOUND" );
                    print( "THEM STOLEN WHEN YOU RETURNED." );
                    player_furs[ FUR_BEAVER ] = 0;
                }
                else if ( event_picker <= 6 )
                {
                    print( "YOU ARRIVED SAFELY AT FORT STADACONA." );
                }
                else if ( event_picker <= 8 )
                {
                    print( "YOUR CANOE UPSET IN THE LACHINE RAPIDS.  YOU" );
                    print( "LOST ALL YOUR FURS." );
                    zeroInventory( player_furs );
                }
                else if ( event_picker <= 10 )
                {
                    print( "YOUR FOX PELTS WERE NOT CURED PROPERLY." );
                    print( "NO ONE WILL BUY THEM." );
                    player_furs[ FUR_FOX ] = 0;
                }
                else
                {
                    printf( "Internal Error #3, Out-of-bounds event_picker %d\n", event_picker );
                    exit( 1 );  /* you have a bug */
                }

                print( "" );
                print( "SUPPLIES AT FORT STADACONA COST $125.00." );
                print( "YOUR TRAVEL EXPENSES TO STADACONA WERE $15.00." );
                player_funds -= 140;
            }

            else if ( which_fort == FORT_NEWYORK )
            {
                mink_price   = ( ( 0.15 * randFloat() + 1.05 ) * 100 + 0.5 ) / 100;
                ermine_price = ( ( 0.15 * randFloat() + 0.95 ) * 100 + 0.5 ) / 100;
                beaver_price = ( ( 0.25 * randFloat() + 1.00 ) * 100 + 0.5 ) / 100;
                if ( fox_price < 0 )
                {
                    /* Original Bug?  There is no Fox price generated for New York,
                       it will use any previous "D1" price.
                       So if there was no previous value, make one up */
                    fox_price = ( ( 0.25 * randFloat() + 1.05 ) * 100 + 0.5 ) / 100; /* not in orginal code */
                }
                event_picker = ( 10 * randFloat() ) + 1;

                if ( event_picker <= 2 )
                {
                    print( "YOU WERE ATTACKED BY A PARTY OF IROQUOIS." );
                    print( "ALL PEOPLE IN YOUR TRADING GROUP WERE" );
                    print( "KILLED.  THIS ENDS THE GAME." );
                    exit( 0 );
                }
                else if ( event_picker <= 6 )
                {
                    print( "YOU WERE LUCKY.  YOU ARRIVED SAFELY" );
                    print( "AT FORT NEW YORK." );
                }
                else if ( event_picker <= 8 )
                {
                    print( "YOU NARROWLY ESCAPED AN IROQUOIS RAIDING PARTY." );
                    print( "HOWEVER, YOU HAD TO LEAVE ALL YOUR FURS BEHIND." );
                    zeroInventory( player_furs );
                }
                else if ( event_picker <= 10 )
                {
                    mink_price /= 2;
                    fox_price  /= 2;
                    print( "YOUR MINK AND BEAVER WERE DAMAGED ON YOUR TRIP." );
                    print( "YOU RECEIVE ONLY HALF THE CURRENT PRICE FOR THESE FURS." );
                }
                else
                {
                    print( "Internal Error #4, Out-of-bounds event_picker %d\n" );
                    exit( 1 );  /* you have a bug */
                }

                print( "" );
                print( "SUPPLIES AT NEW YORK COST $85.00." );
                print( "YOUR TRAVEL EXPENSES TO NEW YORK WERE $25.00." );
                player_funds -= 105;
            }

            else
            {
                printf( "Internal error #2, fort %d does not exist\n", which_fort );
                exit( 1 );  /* you have a bug */
            }

            /* Calculate sales */
            beaver_value = beaver_price * player_furs[ FUR_BEAVER ];
            fox_value    = fox_price    * player_furs[ FUR_FOX ];
            ermine_value = ermine_price * player_furs[ FUR_ERMINE ];
            mink_value   = mink_price   * player_furs[ FUR_MINK ];

            print( "" );
            printf( "YOUR BEAVER SOLD FOR $%6.2f\n", beaver_value );
            printf( "YOUR FOX SOLD FOR    $%6.2f\n", fox_value );
            printf( "YOUR ERMINE SOLD FOR $%6.2f\n", ermine_value );
            printf( "YOUR MINK SOLD FOR   $%6.2f\n", mink_value );

            player_funds += beaver_value + fox_value + ermine_value + mink_value;

            print( "" );
            printf( "YOU NOW HAVE $ %1.2f INCLUDING YOUR PREVIOUS SAVINGS\n", player_funds );

            print( "" );
            print( "DO YOU WANT TO TRADE FURS NEXT YEAR?" );
            yes_or_no = getYesOrNo();
            if ( yes_or_no == 'N' )
                exit( 0 );             /* STOP */
            else
                game_state = STATE_TRADING;

        }
    }

    return 0; /* exit OK */
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [ANSI-C](https://en.wikipedia.org/wiki/ANSI_C)

##### Translator Notes:
I tried to preserve as much of the original layout and flow of the code
as possible.  However I did use enumerated types for the Fort numbers
and Fur types.  I think this was certainly a change for the better, and
makes the code much easier to read.

I also tried to minimise the use of pointers, and stuck with old-school
C formatting, because you never know how old the compiler is.

Interestingly the code seems to have a bug around the prices of Fox Furs.
The commodity-rate for these is stored in the variable `D1`, however some
paths through the code do not set this price.  So there was a chance of
using this uninitialised, or whatever the previous loop set.  I don't
think this was the original authors intent.  So I preserved the original flow
of the code (using the previous `D1` value), but also catching the
uninitialised path, and assigning a "best guess" value.

krt@krt.com.au 2020-10-10


# `00_Alternate_Languages/38_Fur_Trader/go/main.go`

该代码的主要作用是计算并输出树状数组中的所有叶子节点。

具体实现方式如下：

1. 首先定义了两个常量 MAXFURS 和 STARTFUNDS，分别表示树状数组的最大容量和最小起始资金。

2. 定义了一个名为 startFund 的变量，其值为 STARTFUNDS。

3. 定义了一个名为 bufio 的包，该包用于输入输出流操作。

4. 定义了一个名为 "fmt" 的包，该包用于格式化字符串。

5. 定义了一个名为 "log" 的包，该包用于输出日志信息。

6. 定义了一个名为 "math/rand" 的包，该包用于生成随机数。

7. 定义了一个名为 "os" 的包，该包用于操作系统交互操作。

8. 定义了一个名为 "time" 的包，该包用于时间计算。

9. 在 "main" 函数中，先创建一个名为 "bufio.Reader" 的实例，用于读取树状数组中的节点。

10. 循环读取树状数组中的每个节点，先计算当前节点到最小启动资金的差值，然后将当前节点设为叶子节点并输出。

11. 在 "main" 函数中，使用 "fmt.Println" 将计算得到的叶子节点信息输出到控制台。

12. 在 "main" 函数中，使用 "log.SetPrefix" 将输出日志信息统一设置为 "2022-01-01 09:30:00 [INFO] "。

13. 在 "main" 函数中，使用 "time.sleep" 对当前时间进行随机等待，使得输出时随机化。


```
package main

import (
	"bufio"
	"fmt"
	"log"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"
)

const (
	MAXFURS    = 190
	STARTFUNDS = 600
)

```

这段代码定义了两个嵌套的枚举类型Fur和Fort，分别定义了四个不同的类型Fur和四个不同的类型Fort，每个类型都定义了一个名为iota的常量，用于生成独立的名称。

这里使用iota作为名称生成器的灵感来自编程语言中的匿名变量生成器，iota是一个接受一个整数参数并生成一个唯一的名称的库，这在很多编程语言中都有使用。

type Fur int8

type Fort int8

const (
	FUR_MINK Fur = iota
	FUR_BEAVER
	FUR_ERMINE
	FUR_FOX
)

type Forti int8

const (
	FORT_MONTREAL Fort = iota
	FORT_QUEBEC
	FORT_NEWYORK
)

在这段注释中，显示了每个枚举类型的初始常量，以及每个枚举类型的名称。

例如，枚举类型Fur中定义了Fur,Fur_Mink,Fur_Beaver和Fur_Fox。同时，定义了一个常量Fort_Mink，它使用了iota，并从枚举类型Fur中继承了名称。

此外，还定义了一个常量Forti，它使用了iota，并从枚举类型Fort中继承了名称。


```
type Fur int8

const (
	FUR_MINK Fur = iota
	FUR_BEAVER
	FUR_ERMINE
	FUR_FOX
)

type Fort int8

const (
	FORT_MONTREAL Fort = iota
	FORT_QUEBEC
	FORT_NEWYORK
)

```

这段代码定义了一个名为 `GameState` 的枚举类型，它有五个不同的状态，分别为 `STARTING`、`TRADING`、`CHOOSINGFORT` 和 `TRAVELLING`。这些状态用数字 `iota` 初始化，使得 `iota` 每次循环时都递增。

另外，代码中还定义了两个函数：`FURS` 和 `FORTS`。这些函数没有明确的返回类型，但它们的参数列表却是有明确返回类型的。

具体来说，`FURS` 函数返回了一个字符串数组，它代表了动物 `MINK`、`BEAVER`、`ERMINE` 和 `FOX`。

`FORTS` 函数返回了一个字符串数组，它代表了三个城市的名称，分别是 `HOCHELAGA (MONTREAL)`、`STADACONA (QUEBEC)` 和 `NEW YORK`。


```
type GameState int8

const (
	STARTING GameState = iota
	TRADING
	CHOOSINGFORT
	TRAVELLING
)

func FURS() []string {
	return []string{"MINK", "BEAVER", "ERMINE", "FOX"}
}

func FORTS() []string {
	return []string{"HOCHELAGA (MONTREAL)", "STADACONA (QUEBEC)", "NEW YORK"}
}

```

这是一个定义了一个名为Player的结构体，其中包含一个float32类型的变量funds和一个整数类型的数组furs。

在另一个名为NewPlayer的函数中，使用结构体类型的Player变量创建了一个新的Player实例，并将其赋值为调用者提供的值STARTFUNDS。

此外，还定义了一个名为totalFurs的函数，它接受一个Player类型的变量作为实参，并计算该Player实例中所有furs值的数量。

totalFurs函数的实现比较简单，它遍历了Player实例中的furs数组，将每个furs值的数量加到一个名为f的变量中，并返回该变量的值。最后，totalFurs函数返回f的值，即Player实例中所有furs值的数量。


```
type Player struct {
	funds float32
	furs  []int
}

func NewPlayer() Player {
	p := Player{}
	p.funds = STARTFUNDS
	p.furs = make([]int, 4)
	return p
}

func (p *Player) totalFurs() int {
	f := 0
	for _, v := range p.furs {
		f += v
	}
	return f
}

```

这段代码定义了三个函数：lostFurs()、printTitle() 和 printIntro()。

lostFurs() 函数的作用是输出所有 `Player` 对象中的 fur 值，并将其设置为 0。这个函数对于每个 `Player` 对象都执行一次，因此它不会对单个对象产生影响。

printTitle() 函数用于输出一个标题字符串。它使用 `fmt.Println()` 函数打印字符串，并使用多个 `fmt.Println()` 函数来在字符串中添加新行。

printIntro() 函数用于输出关于 fur trading expedition 的介绍信息。它使用 `fmt.Println()` 函数打印字符串，并使用多个 `fmt.Println()` 函数来在字符串中添加新行。


```
func (p *Player) lostFurs() {
	for f := 0; f < len(p.furs); f++ {
		p.furs[f] = 0
	}
}

func printTitle() {
	fmt.Println("                               FUR TRADER")
	fmt.Println("               CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
	fmt.Println()
	fmt.Println()
	fmt.Println()
}

func printIntro() {
	fmt.Println("YOU ARE THE LEADER OF A FRENCH FUR TRADING EXPEDITION IN ")
	fmt.Println("1776 LEAVING THE LAKE ONTARIO AREA TO SELL FURS AND GET")
	fmt.Println("SUPPLIES FOR THE NEXT YEAR.  YOU HAVE A CHOICE OF THREE")
	fmt.Println("FORTS AT WHICH YOU MAY TRADE.  THE COST OF SUPPLIES")
	fmt.Println("AND THE AMOUNT YOU RECEIVE FOR YOUR FURS WILL DEPEND")
	fmt.Println("ON THE FORT THAT YOU CHOOSE.")
	fmt.Println()
}

```

这段代码使用了Fort库，用于在控制台输出选择性较高的堡垒。函数名为getFortChoice，返回一个Fort类型的变量。

主要步骤如下：

1. 使用bufio.NewScanner从标准输入（os.Stdin）读取输入数据。
2. 使用for循环遍历输入数据。
3. 在循环内，首先输出字符串"YOU MAY TRADE YOUR FURS AT FORT 1, FORT 2,"，接着输出字符串"OR FORT 3. Fort 1 IS FORT HOCHELAGA (MONTREAL) AND IS UNDER THE PROTECTION OF THE FRENCH ARMY."，然后输出字符串"AND IS UNDER THE PROTECTION OF THE FRENCH ARMY."，接着输出"FORT 2 IS FORT STADACONA (QUEBEC) AND IS UNDER THE PROTECTION OF THE FRENCH ARMY. HOWEVER, YOU MUST MAKE A PORTAGE AND CROSS THE LACHINE RAPIDS."，然后输出"FORT 3 IS FORT NEW YORK AND IS UNDER DUTCH CONTROL. YOU MUST CROSS THROUGH IROQUOIS LAND."，最后输出"ANSWER 1, 2, OR 3。"。
4. 从输入数据中提取整数并转换为字符串，如果错误则输出提示信息并继续。
5. 返回字符串"2"。

函数的作用是在控制台输出堡垒的选择，并提示用户输入正确的数字。如果用户输入的不是数字1、2或3，则会提示并重新尝试用户输入。


```
func getFortChoice() Fort {
	scanner := bufio.NewScanner(os.Stdin)

	for {
		fmt.Println()
		fmt.Println("YOU MAY TRADE YOUR FURS AT FORT 1, FORT 2,")
		fmt.Println("OR FORT 3.  FORT 1 IS FORT HOCHELAGA (MONTREAL)")
		fmt.Println("AND IS UNDER THE PROTECTION OF THE FRENCH ARMY.")
		fmt.Println("FORT 2 IS FORT STADACONA (QUEBEC) AND IS UNDER THE")
		fmt.Println("PROTECTION OF THE FRENCH ARMY.  HOWEVER, YOU MUST")
		fmt.Println("MAKE A PORTAGE AND CROSS THE LACHINE RAPIDS.")
		fmt.Println("FORT 3 IS FORT NEW YORK AND IS UNDER DUTCH CONTROL.")
		fmt.Println("YOU MUST CROSS THROUGH IROQUOIS LAND.")
		fmt.Println("ANSWER 1, 2, OR 3.")
		fmt.Print(">> ")
		scanner.Scan()

		f, err := strconv.Atoi(scanner.Text())
		if err != nil || f < 1 || f > 3 {
			fmt.Println("Invalid input, Try again ... ")
			continue
		}
		return Fort(f)
	}
}

```

这段代码定义了一个名为 printFortComment 的函数，它接受一个 Fort 类型的参数 f。函数的主要作用是打印关于不同路线的信息，以便用户了解选择路线的影响。

具体来说，当 f 为 FORT_MONTREAL 时，函数会打印出 "YOU HAVE CHOSEN THE EASIEST ROUTE.  HOWEVER, THE FORT IS FAR FROM ANY SEAPORT.  THE VALUE OF SUPPLIES FOR YOUR FURS WILL BE LOW AND THE COST OF SUPPLIES WILL BE HIGHER THAN AT FORTS STADACONA OR NEW YORK。" 的信息。当 f 为 FORT_QUEBEC 时，函数会打印出 "YOU HAVE CHOSEN A HARD ROUTE.  IT IS, IN COMPERSHIP, HARDER THAN THE ROUTE TO HOCHELAGA BUT EASIER THAN THE ROUTE TO NEW YORK.  YOU WILL RECEIVE AN AVERAGE VALUE FOR YOUR FURS AND THE COST OF YOUR SUPPLIES。" 的信息。当 f 为 FORT_NEWYORK 时，函数会打印出 "YOU HAVE CHOSEN THE MOST DIFFICULAR ROUTE.  AT FORT NEW YORK YOU WILL RECEIVE THE HIGHEST VALUE FOR YOUR FURS.  THE COST OF YOUR SUPPLIES WILL BE LOWER THAN AT ALL THE OTHER FORTS。" 的信息。

打印结束后，函数使用 fmt.Println() 函数打印空行，以使输出更易于阅读。


```
func printFortComment(f Fort) {
	fmt.Println()
	switch f {
	case FORT_MONTREAL:
		fmt.Println("YOU HAVE CHOSEN THE EASIEST ROUTE.  HOWEVER, THE FORT")
		fmt.Println("IS FAR FROM ANY SEAPORT.  THE VALUE")
		fmt.Println("YOU RECEIVE FOR YOUR FURS WILL BE LOW AND THE COST")
		fmt.Println("OF SUPPLIES HIGHER THAN AT FORTS STADACONA OR NEW YORK.")
	case FORT_QUEBEC:
		fmt.Println("YOU HAVE CHOSEN A HARD ROUTE.  IT IS, IN COMPARSION,")
		fmt.Println("HARDER THAN THE ROUTE TO HOCHELAGA BUT EASIER THAN")
		fmt.Println("THE ROUTE TO NEW YORK.  YOU WILL RECEIVE AN AVERAGE VALUE")
		fmt.Println("FOR YOUR FURS AND THE COST OF YOUR SUPPLIES WILL BE AVERAGE.")
	case FORT_NEWYORK:
		fmt.Println("YOU HAVE CHOSEN THE MOST DIFFICULT ROUTE.  AT")
		fmt.Println("FORT NEW YORK YOU WILL RECEIVE THE HIGHEST VALUE")
		fmt.Println("FOR YOUR FURS.  THE COST OF YOUR SUPPLIES")
		fmt.Println("WILL BE LOWER THAN AT ALL THE OTHER FORTS.")
	}
	fmt.Println()
}

```

这两段代码都是用于获取用户输入的Yes或No答案以及Furs购买数量。

第一段代码是一个函数getYesOrNo()，它使用了bufio.NewScanner从标准输入（通常是键盘输入）中读取字符串，然后循环输出"ANSWER YES OR NO"并等待用户输入。如果用户输入的字符串是"Y"，则返回字符串"Y"，否则返回字符串"N"。

第二段代码是一个函数getFursPurchase()，它同样使用了bufio.NewScanner从标准输入中读取字符串，然后循环输出"YOUR %d FURS ARE DISTRIBUTED AMONG THE FOLLOWING KINDS OF PELTS"并等待用户输入。然后它读取用户的输入并尝试将输入转换为整数，如果转换失败，则输出"INVALID INPUT，TRY AGAIN..."并继续循环。最后，它返回输入值的总和，即购买Furs的数量。


```
func getYesOrNo() string {
	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Println("ANSWER YES OR NO")
		scanner.Scan()
		if strings.ToUpper(scanner.Text())[0:1] == "Y" {
			return "Y"
		} else if strings.ToUpper(scanner.Text())[0:1] == "N" {
			return "N"
		}
	}
}

func getFursPurchase() []int {
	scanner := bufio.NewScanner(os.Stdin)
	fmt.Printf("YOUR %d FURS ARE DISTRIBUTED AMONG THE FOLLOWING\n", MAXFURS)
	fmt.Println("KINDS OF PELTS: MINK, BEAVER, ERMINE AND FOX.")
	fmt.Println()

	purchases := make([]int, 4)

	for i, f := range FURS() {
	retry:
		fmt.Printf("HOW MANY %s DO YOU HAVE: ", f)
		scanner.Scan()
		count, err := strconv.Atoi(scanner.Text())
		if err != nil {
			fmt.Println("INVALID INPUT, TRY AGAIN ...")
			goto retry
		}
		purchases[i] = count
	}

	return purchases
}

```

This is a function definition for a game in the area of交易系统。
It takes in a log message if an error happens.
It then prints the supplies,
then it calculates the value of the different animals and updates the funds.
Then it asks the user if they want to trade furors in the next year and then exits the program using os.Exit.
It seems like the game is a simple trading simulation, where the user can trade different types of furors and the game updates the user's funds based on the value of the furors they trade.
It also has some logging mechanism to check if any unexpected errors happen and to indicate the user if it does.


```
func main() {
	rand.Seed(time.Now().UnixNano())

	printTitle()

	gameState := STARTING
	whichFort := FORT_NEWYORK
	var (
		minkPrice   int
		erminePrice int
		beaverPrice int
		foxPrice    int
	)
	player := NewPlayer()

	for {
		switch gameState {
		case STARTING:
			printIntro()
			fmt.Println("DO YOU WISH TO TRADE FURS?")
			if getYesOrNo() == "N" {
				os.Exit(0)
			}
			gameState = TRADING
		case TRADING:
			fmt.Println()
			fmt.Printf("YOU HAVE $ %1.2f IN SAVINGS\n", player.funds)
			fmt.Printf("AND %d FURS TO BEGIN THE EXPEDITION\n", MAXFURS)
			player.furs = getFursPurchase()

			if player.totalFurs() > MAXFURS {
				fmt.Println()
				fmt.Println("YOU MAY NOT HAVE THAT MANY FURS.")
				fmt.Println("DO NOT TRY TO CHEAT.  I CAN ADD.")
				fmt.Println("YOU MUST START AGAIN.")
				gameState = STARTING
			} else {
				gameState = CHOOSINGFORT
			}
		case CHOOSINGFORT:
			whichFort = getFortChoice()
			printFortComment(whichFort)
			fmt.Println("DO YOU WANT TO TRADE AT ANOTHER FORT?")
			changeFort := getYesOrNo()
			if changeFort == "N" {
				gameState = TRAVELLING
			}
		case TRAVELLING:
			switch whichFort {
			case FORT_MONTREAL:
				minkPrice = (int((0.2*rand.Float64()+0.70)*100+0.5) / 100)
				erminePrice = (int((0.2*rand.Float64()+0.65)*100+0.5) / 100)
				beaverPrice = (int((0.2*rand.Float64()+0.75)*100+0.5) / 100)
				foxPrice = (int((0.2*rand.Float64()+0.80)*100+0.5) / 100)

				fmt.Println("SUPPLIES AT FORT HOCHELAGA COST $150.00.")
				fmt.Println("YOUR TRAVEL EXPENSES TO HOCHELAGA WERE $10.00.")
				player.funds -= 160
			case FORT_QUEBEC:
				minkPrice = (int((0.30*rand.Float64()+0.85)*100+0.5) / 100)
				erminePrice = (int((0.15*rand.Float64()+0.80)*100+0.5) / 100)
				beaverPrice = (int((0.20*rand.Float64()+0.90)*100+0.5) / 100)
				foxPrice = (int((0.25*rand.Float64()+1.10)*100+0.5) / 100)

				event := int(10*rand.Float64()) + 1
				if event <= 2 {
					fmt.Println("YOUR BEAVER WERE TOO HEAVY TO CARRY ACROSS")
					fmt.Println("THE PORTAGE. YOU HAD TO LEAVE THE PELTS, BUT FOUND")
					fmt.Println("THEM STOLEN WHEN YOU RETURNED.")
					player.furs[FUR_BEAVER] = 0
				} else if event <= 6 {
					fmt.Println("YOU ARRIVED SAFELY AT FORT STADACONA.")
				} else if event <= 8 {
					fmt.Println("YOUR CANOE UPSET IN THE LACHINE RAPIDS.  YOU")
					fmt.Println("LOST ALL YOUR FURS.")
					player.lostFurs()
				} else if event <= 10 {
					fmt.Println("YOUR FOX PELTS WERE NOT CURED PROPERLY.")
					fmt.Println("NO ONE WILL BUY THEM.")
					player.furs[FUR_FOX] = 0
				} else {
					log.Fatal("Unexpected error")
				}

				fmt.Println()
				fmt.Println("SUPPLIES AT FORT STADACONA COST $125.00.")
				fmt.Println("YOUR TRAVEL EXPENSES TO STADACONA WERE $15.00.")
				player.funds -= 140
			case FORT_NEWYORK:
				minkPrice = (int((0.15*rand.Float64()+1.05)*100+0.5) / 100)
				erminePrice = (int((0.15*rand.Float64()+0.95)*100+0.5) / 100)
				beaverPrice = (int((0.25*rand.Float64()+1.00)*100+0.5) / 100)
				foxPrice = (int((0.25*rand.Float64()+1.05)*100+0.5) / 100) // not in original code

				event := int(10*rand.Float64()) + 1
				if event <= 2 {
					fmt.Println("YOU WERE ATTACKED BY A PARTY OF IROQUOIS.")
					fmt.Println("ALL PEOPLE IN YOUR TRADING GROUP WERE")
					fmt.Println("KILLED.  THIS ENDS THE GAME.")
					os.Exit(0)
				} else if event <= 6 {
					fmt.Println("YOU WERE LUCKY.  YOU ARRIVED SAFELY")
					fmt.Println("AT FORT NEW YORK.")
				} else if event <= 8 {
					fmt.Println("YOU NARROWLY ESCAPED AN IROQUOIS RAIDING PARTY.")
					fmt.Println("HOWEVER, YOU HAD TO LEAVE ALL YOUR FURS BEHIND.")
					player.lostFurs()
				} else if event <= 10 {
					minkPrice /= 2
					foxPrice /= 2
					fmt.Println("YOUR MINK AND BEAVER WERE DAMAGED ON YOUR TRIP.")
					fmt.Println("YOU RECEIVE ONLY HALF THE CURRENT PRICE FOR THESE FURS.")
				} else {
					log.Fatal("Unexpected error")
				}

				fmt.Println()
				fmt.Println("SUPPLIES AT NEW YORK COST $85.00.")
				fmt.Println("YOUR TRAVEL EXPENSES TO NEW YORK WERE $25.00.")
				player.funds -= 110
			}

			beaverValue := beaverPrice * player.furs[FUR_BEAVER]
			foxValue := foxPrice * player.furs[FUR_FOX]
			ermineValue := erminePrice * player.furs[FUR_ERMINE]
			minkValue := minkPrice * player.furs[FUR_MINK]

			fmt.Println()
			fmt.Printf("YOUR BEAVER SOLD FOR $%6.2f\n", float64(beaverValue))
			fmt.Printf("YOUR FOX SOLD FOR    $%6.2f\n", float64(foxValue))
			fmt.Printf("YOUR ERMINE SOLD FOR $%6.2f\n", float64(ermineValue))
			fmt.Printf("YOUR MINK SOLD FOR   $%6.2f\n", float64(minkValue))

			player.funds += float32(beaverValue + foxValue + ermineValue + minkValue)

			fmt.Println()
			fmt.Printf("YOU NOW HAVE $%1.2f INCLUDING YOUR PREVIOUS SAVINGS\n", player.funds)
			fmt.Println("\nDO YOU WANT TO TRADE FURS NEXT YEAR?")
			if getYesOrNo() == "N" {
				os.Exit(0)
			} else {
				gameState = TRADING
			}
		}
	}
}

```