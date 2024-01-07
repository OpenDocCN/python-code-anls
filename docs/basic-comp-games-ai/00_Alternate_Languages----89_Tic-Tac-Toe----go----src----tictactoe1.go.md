# `basic-computer-games\00_Alternate_Languages\89_Tic-Tac-Toe\go\src\tictactoe1.go`

```

package main

import (
	"fmt"
	"strconv"
)

func main() {
	// 在屏幕上打印文本，文本前有30个空格
	fmt.Printf("%30s\n", "TIC TAC TOE")
	// 在屏幕上打印文本，文本前有15个空格，并且在屏幕上打印三行空行
	fmt.Printf("%15s\n\n\n\n", "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
	// 这个程序玩井字棋
	fmt.Printf("THE GAME BOARD IS NUMBERED:\n\n")
	fmt.Println("1  2  3")
	fmt.Println("8  9  4")
	fmt.Println("7  6  5")

	// 主程序
	for {
		var (
			a, b, c, d, e int
			p, q, r, s    int
		)
		a = 9
		fmt.Printf("\n\n")
		// 机器先走
		computerMoves(a)
		p = readYourMove()
		b = move(p + 1)
		computerMoves(b)
		q = readYourMove()
		if q == move(b+4) {
			c = move(b + 2)
			computerMoves(c)
			r = readYourMove()
			if r == move(c+4) {
				if p%2 != 0 {
					d = move(c + 3)
					computerMoves(d)
					s = readYourMove()
					if s == move(d+4) {
						e = move(d + 6)
						computerMoves(e)
						fmt.Println("THE GAME IS A DRAW.")
					} else {
						e = move(d + 4)
						computerMoves(e)
						fmt.Println("AND WINS ********")
					}
				} else {
					d = move(c + 7)
					computerMoves(d)
					fmt.Println("AND WINS ********")
				}
			} else {
				d = move(c + 4)
				computerMoves(d)
				fmt.Println("AND WINS ********")
			}
		} else {
			c = move(b + 4)
			computerMoves(c)
			fmt.Println("AND WINS ********")
		}
	}
}
func computerMoves(move int) {
	// 打印机器走的步数
	fmt.Printf("COMPUTER MOVES %v\n", move)
}

func readYourMove() int {
	for {
		// 提示用户输入步数
		fmt.Printf("YOUR MOVE?")
		var input string
		// 读取用户输入
		fmt.Scan(&input)
		// 将输入转换为整数
		number, err := strconv.Atoi(input)
		if err == nil {
			return number
		}
	}
}

func move(number int) int {
	// 计算移动后的位置
	return number - 8*(int)((number-1)/8)
}

```