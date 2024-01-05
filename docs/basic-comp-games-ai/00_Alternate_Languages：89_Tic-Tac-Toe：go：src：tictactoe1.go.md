# `d:/src/tocomm/basic-computer-games\00_Alternate_Languages\89_Tic-Tac-Toe\go\src\tictactoe1.go`

```
package main  // 声明包名为 main

import (  // 导入需要的包
	"fmt"  // 导入 fmt 包，用于格式化输出
	"strconv"  // 导入 strconv 包，用于字符串和数字之间的转换
)

func main() {  // 主函数入口
	// 在屏幕上打印文本，文本前面有 30 个空格
	fmt.Printf("%30s\n", "TIC TAC TOE")
	// 在屏幕上打印文本，文本前面有 15 个空格
	// 并在屏幕上打印三行换行
	fmt.Printf("%15s\n\n\n\n", "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
	// 打印文本
	fmt.Printf("THE GAME BOARD IS NUMBERED:\n\n")
	// 打印文本
	fmt.Println("1  2  3")
	// 打印文本
	fmt.Println("8  9  4")
	// 打印文本
	fmt.Println("7  6  5")

	// 主程序
}
	for {
		var (
			a, b, c, d, e int  // 声明整型变量 a, b, c, d, e
			p, q, r, s    int  // 声明整型变量 p, q, r, s
		)
		a = 9  // 将变量 a 赋值为 9
		fmt.Printf("\n\n")  // 打印两个换行符
		// THE MACHINE GOES FIRST  // 打印提示信息
		computerMoves(a)  // 调用 computerMoves 函数并传入参数 a
		p = readYourMove()  // 调用 readYourMove 函数并将返回值赋给变量 p
		b = move(p + 1)  // 调用 move 函数并将 p+1 的结果赋给变量 b
		computerMoves(b)  // 调用 computerMoves 函数并传入参数 b
		q = readYourMove()  // 调用 readYourMove 函数并将返回值赋给变量 q
		if q == move(b+4) {  // 如果 q 等于 move(b+4) 的结果
			c = move(b + 2)  // 将 move(b+2) 的结果赋给变量 c
			computerMoves(c)  // 调用 computerMoves 函数并传入参数 c
			r = readYourMove()  // 调用 readYourMove 函数并将返回值赋给变量 r
			if r == move(c+4) {  // 如果 r 等于 move(c+4) 的结果
				if p%2 != 0 {  // 如果 p 除以 2 的余数不等于 0
					d = move(c + 3)  // 将 move(c+3) 的结果赋给变量 d
					# 让计算机根据给定的位置进行移动
					computerMoves(d)
					# 读取玩家的移动
					s = readYourMove()
					# 如果玩家的移动等于计算机移动位置加4
					if s == move(d+4) {
						# 计算机移动到位置d+6
						e = move(d + 6)
						# 让计算机根据新位置e进行移动
						computerMoves(e)
						# 打印游戏平局的消息
						fmt.Println("THE GAME IS A DRAW.")
					} else {
						# 否则，计算机移动到位置d+4
						e = move(d + 4)
						# 让计算机根据新位置e进行移动
						computerMoves(e)
						# 打印计算机获胜的消息
						fmt.Println("AND WINS ********")
					}
				} else {
					# 如果玩家的移动不等于计算机移动位置加4，则计算机移动到位置c+7
					d = move(c + 7)
					# 让计算机根据新位置d进行移动
					computerMoves(d)
					# 打印计算机获胜的消息
					fmt.Println("AND WINS ********")
				}
			} else {
				# 如果玩家的移动不等于计算机移动位置加4，则计算机移动到位置c+4
				d = move(c + 4)
				# 让计算机根据新位置d进行移动
				computerMoves(d)
				# 打印计算机获胜的消息
				fmt.Println("AND WINS ********")
		}
	} else {
		# 从位置 b + 4 开始移动，将移动结果赋值给变量 c
		c = move(b + 4)
		# 调用 computerMoves 函数，传入参数 c
		computerMoves(c)
		# 打印信息："AND WINS ********"
		fmt.Println("AND WINS ********")
	}
}
# 定义函数 computerMoves，接受一个参数 move
func computerMoves(move int) {
	# 打印信息："COMPUTER MOVES" 后跟参数 move 的值
	fmt.Printf("COMPUTER MOVES %v\n", move)
}

# 定义函数 readYourMove，返回一个整数
func readYourMove() int {
	# 进入无限循环
	for {
		# 打印信息："YOUR MOVE?"
		fmt.Printf("YOUR MOVE?")
		# 声明变量 input 为字符串类型
		var input string
		# 从标准输入中读取一行，并将其存储在 input 变量中
		fmt.Scan(&input)
		# 将 input 转换为整数类型，存储在 number 变量中，并检查是否有错误发生
		number, err := strconv.Atoi(input)
		# 如果没有错误发生
		if err == nil {
			# 返回转换后的整数值
			return number
		}
		}  # 结束第一个 for 循环的代码块
	}  # 结束第二个 for 循环的代码块
}  # 结束函数 move 的代码块

func move(number int) int {  # 定义一个名为 move 的函数，接受一个整数参数，返回一个整数
	return number - 8*(int)((number-1)/8)  # 返回 number 减去 8 乘以 (number-1) 除以 8 的整数部分
}
```