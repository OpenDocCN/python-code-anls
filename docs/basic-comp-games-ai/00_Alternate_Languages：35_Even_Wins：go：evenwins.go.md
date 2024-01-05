# `d:/src/tocomm/basic-computer-games\00_Alternate_Languages\35_Even_Wins\go\evenwins.go`

```
package main  // 声明当前文件属于 main 包

import (
	"bufio"  // 导入 bufio 包，用于读取输入
	"fmt"  // 导入 fmt 包，用于格式化输入输出
	"os"  // 导入 os 包，用于操作系统功能
	"strconv"  // 导入 strconv 包，用于字符串和数字之间的转换
	"strings"  // 导入 strings 包，用于处理字符串
)

const MAXTAKE = 4  // 声明常量 MAXTAKE，值为 4

type PlayerType int8  // 声明 PlayerType 类型为 int8

const (
	HUMAN PlayerType = iota  // 声明 HUMAN 值为 0
	COMPUTER  // 声明 COMPUTER 值为 1
)

type Game struct {  // 声明 Game 结构体
	table    int  // 定义整型变量 table
	human    int  // 定义整型变量 human
	computer int  // 定义整型变量 computer
}

func NewGame() Game {  // 定义一个名为 NewGame 的函数，返回类型为 Game 结构体
	g := Game{}  // 创建一个 Game 结构体实例
	g.table = 27  // 将实例的 table 属性赋值为 27

	return g  // 返回创建的 Game 结构体实例
}

func printIntro() {  // 定义一个名为 printIntro 的函数
	fmt.Println("Welcome to Even Wins!")  // 打印欢迎信息
	fmt.Println("Based on evenwins.bas from Creative Computing")  // 打印基于 Creative Computing 的 evenwins.bas
	fmt.Println()  // 打印空行
	fmt.Println("Even Wins is a two-person game. You start with")  // 打印游戏介绍
	fmt.Println("27 marbles in the middle of the table.")  // 打印游戏介绍
	fmt.Println()  // 打印空行
	fmt.Println("Players alternate taking marbles from the middle.")  // 打印游戏玩法介绍
}
	fmt.Println("A player can take 1 to 4 marbles on their turn, and")  // 打印玩家每次可以拿1到4个弹珠
	fmt.Println("turns cannot be skipped. The game ends when there are")  // 打印不能跳过回合。当没有弹珠时游戏结束
	fmt.Println("no marbles left, and the winner is the one with an even")  // 打印没有弹珠剩余，胜利者是拥有偶数个弹珠的玩家
	fmt.Println("number of marbles.")  // 打印弹珠数量
	fmt.Println()
}

func (g *Game) printBoard() {
	fmt.Println()  // 打印空行
	fmt.Printf(" marbles in the middle: %d\n", g.table)  // 打印中间的弹珠数量
	fmt.Printf("    # marbles you have: %d\n", g.human)  // 打印玩家拥有的弹珠数量
	fmt.Printf("# marbles computer has: %d\n", g.computer)  // 打印电脑拥有的弹珠数量
	fmt.Println()  // 打印空行
}

func (g *Game) gameOver() {
	fmt.Println()  // 打印空行
	fmt.Println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")  // 打印游戏结束的提示
	fmt.Println("!! All the marbles are taken: Game Over!")  // 打印所有弹珠都被拿走了，游戏结束
	fmt.Println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")  // 打印游戏结束的提示
	fmt.Println()  # 打印空行
	g.printBoard()  # 调用 printBoard 方法打印游戏板
	if g.human%2 == 0:  # 如果 human 变量除以 2 的余数为 0
		fmt.Println("You are the winner! Congratulations!")  # 打印玩家获胜的消息
	else:
		fmt.Println("The computer wins: all hail mighty silicon!")  # 打印计算机获胜的消息
	fmt.Println()  # 打印空行
}

func getPlural(count int) string:  # 定义一个函数，根据 count 参数返回单数或复数的字符串
	m := "marble"  # 初始化 m 变量为 "marble"
	if count > 1:  # 如果 count 大于 1
		m += "s"  # 在 m 后面加上 "s"
	return m  # 返回 m 变量的值

func (g *Game) humanTurn():  # 定义一个方法，表示玩家的回合
	scanner := bufio.NewScanner(os.Stdin)  # 创建一个从标准输入读取数据的 Scanner 对象
	maxAvailable := MAXTAKE  // 设置最大可取的数量为 MAXTAKE
	if g.table < MAXTAKE {  // 如果桌子上的大理石数量小于 MAXTAKE
		maxAvailable = g.table  // 则最大可取的数量为桌子上的大理石数量
	}

	fmt.Println("It's your turn!")  // 打印提示信息，轮到你了
	for {  // 进入循环
		fmt.Printf("Marbles to take? (1 - %d) --> ", maxAvailable)  // 打印提示信息，输入要取的大理石数量范围
		scanner.Scan()  // 从标准输入中扫描用户输入
		n, err := strconv.Atoi(scanner.Text())  // 将用户输入的文本转换为整数
		if err != nil {  // 如果转换出错
			fmt.Printf("\n  Please enter a whole number from 1 to %d\n", maxAvailable)  // 打印错误提示信息
			continue  // 继续下一次循环
		}
		if n < 1 {  // 如果输入小于1
			fmt.Println("\n  You must take at least 1 marble!")  // 打印错误提示信息
			continue  // 继续下一次循环
		}
		if n > maxAvailable {  // 如果输入大于最大可取数量
			fmt.Printf("\n  You can take at most %d %s\n", maxAvailable, getPlural(maxAvailable))  // 打印错误提示信息
			continue
		}
		fmt.Printf("\nOkay, taking %d %s ...\n", n, getPlural(n))
		g.table -= n
		g.human += n
		return
	}
}

func (g *Game) computerTurn() {
	marblesToTake := 0

	fmt.Println("It's the computer's turn ...")
	r := float64(g.table - 6*int((g.table)/6))

	if int(g.human/2) == g.human/2 {  // 检查玩家拿的弹珠数量是否为偶数
		if r < 1.5 || r > 5.3 {  // 如果余数小于1.5或大于5.3
			marblesToTake = 1  // 则电脑拿1个弹珠
		} else {
			marblesToTake = int(r - 1)  // 否则电脑拿余数减1个弹珠
		}
	} else if float64(g.table) < 4.2 {  # 如果游戏桌上的弹珠数量小于4.2
		marblesToTake = 4  # 则电脑取4个弹珠
	} else if r > 3.4 {  # 否则如果随机数大于3.4
		if r < 4.7 || r > 3.5 {  # 并且随机数小于4.7或者大于3.5
			marblesToTake = 4  # 则电脑取4个弹珠
		}
	} else {  # 否则
		marblesToTake = int(r + 1)  # 电脑取随机数加1个弹珠
	}

	fmt.Printf("Computer takes %d %s ...\n", marblesToTake, getPlural(marblesToTake))  # 打印电脑取的弹珠数量和对应的单复数形式
	g.table -= marblesToTake  # 更新游戏桌上的弹珠数量
	g.computer += marblesToTake  # 更新电脑的弹珠数量
}

func (g *Game) play(playersTurn PlayerType) {  # 定义游戏进行的方法，参数为玩家类型
	g.printBoard()  # 打印游戏桌面

	for {  # 进入循环
		if g.table == 0 {  # 如果游戏结束，调用游戏结束函数并返回
			g.gameOver()
			return
		} else if playersTurn == HUMAN {  # 如果轮到玩家
			g.humanTurn()  # 调用玩家回合函数
			g.printBoard()  # 打印游戏棋盘
			playersTurn = COMPUTER  # 轮到电脑
		} else {  # 如果轮到电脑
			g.computerTurn()  # 调用电脑回合函数
			g.printBoard()  # 打印游戏棋盘
			playersTurn = HUMAN  # 轮到玩家
		}
	}
}

func getFirstPlayer() PlayerType {  # 定义获取先手玩家的函数
	scanner := bufio.NewScanner(os.Stdin)  # 创建一个从标准输入读取数据的扫描器

	for {  # 循环
		fmt.Println("Do you want to play first? (y/n) --> ")  # 打印提示信息
		scanner.Scan()  # 从标准输入中扫描下一行文本

		if strings.ToUpper(scanner.Text()) == "Y":  # 将输入的文本转换为大写，如果等于"Y"，则返回HUMAN
			return HUMAN
		} else if strings.ToUpper(scanner.Text()) == "N":  # 如果输入的文本转换为大写等于"N"，则返回COMPUTER
			return COMPUTER
		} else {  # 如果输入的文本既不是"Y"也不是"N"
			fmt.Println()  # 打印空行
			fmt.Println("Please enter 'y' if you want to play first,")  # 提示输入'y'来先手
			fmt.Println("or 'n' if you want to play second.")  # 提示输入'n'来后手
			fmt.Println()  # 打印空行
		}
	}
}

func main() {
	scanner := bufio.NewScanner(os.Stdin)  # 创建一个新的扫描器来读取标准输入

	printIntro()  # 调用打印介绍的函数
	for {
		# 创建一个新的游戏对象
		g := NewGame()

		# 开始游戏，传入第一个玩家
		g.play(getFirstPlayer())

		# 打印提示信息，询问玩家是否想再玩一次
		fmt.Println("\nWould you like to play again? (y/n) --> ")
		scanner.Scan()
		
		# 如果玩家输入的是"y"或"Y"，则继续下一轮游戏
		if strings.ToUpper(scanner.Text()) == "Y" {
			fmt.Println("\nOk, let's play again ...")
		} else {
			# 如果玩家输入的不是"y"或"Y"，则打印结束信息并结束游戏
			fmt.Println("\nOk, thanks for playing ... goodbye!")
			return
		}

	}

}
```