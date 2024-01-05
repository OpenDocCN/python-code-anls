# `d:/src/tocomm/basic-computer-games\00_Alternate_Languages\34_Digits\go\main.go`

```
package main  // 声明当前文件属于 main 包

import (  // 导入需要使用的包
	"bufio"  // 用于读取输入
	"fmt"  // 用于格式化输出
	"math/rand"  // 用于生成随机数
	"os"  // 用于处理文件和目录
	"strconv"  // 用于字符串和数字之间的转换
	"time"  // 用于处理时间
)

func printIntro() {  // 定义函数 printIntro
	fmt.Println("                                DIGITS")  // 输出字符串
	fmt.Println("              CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")  // 输出字符串
	fmt.Println()  // 输出空行
	fmt.Println()  // 输出空行
	fmt.Println("THIS IS A GAME OF GUESSING.")  // 输出字符串
}

func readInteger(prompt string) int {  // 定义函数 readInteger，接收一个字符串参数 prompt，返回一个整数
	scanner := bufio.NewScanner(os.Stdin)  // 创建一个从标准输入读取数据的扫描器
	for {  // 无限循环，直到遇到 return 语句
		fmt.Println(prompt)  // 打印提示信息
		scanner.Scan()  // 从标准输入读取一行数据
		response, err := strconv.Atoi(scanner.Text())  // 将读取的数据转换为整数

		if err != nil {  // 如果转换出错
			fmt.Println("INVALID INPUT, TRY AGAIN... ")  // 打印错误信息
			continue  // 继续下一次循环
		}

		return response  // 返回读取的整数值
	}
}

func printInstructions() {
	fmt.Println()  // 打印空行
	fmt.Println("PLEASE TAKE A PIECE OF PAPER AND WRITE DOWN")  // 打印提示信息
	fmt.Println("THE DIGITS '0', '1', OR '2' THIRTY TIMES AT RANDOM.")  // 打印提示信息
	fmt.Println("ARRANGE THEM IN THREE LINES OF TEN DIGITS EACH.")  // 打印提示信息
}
	fmt.Println("I WILL ASK FOR THEN TEN AT A TIME.")  // 打印提示信息
	fmt.Println("I WILL ALWAYS GUESS THEM FIRST AND THEN LOOK AT YOUR")  // 打印提示信息
	fmt.Println("NEXT NUMBER TO SEE IF I WAS RIGHT. BY PURE LUCK,")  // 打印提示信息
	fmt.Println("I OUGHT TO BE RIGHT TEN TIMES. BUT I HOPE TO DO BETTER")  // 打印提示信息
	fmt.Println("THAN THAT *****")  // 打印提示信息
	fmt.Println()  // 打印空行
}

func readTenNumbers() []int {
	numbers := make([]int, 10)  // 创建一个包含10个整数的切片

	numbers[0] = readInteger("FIRST NUMBER: ")  // 读取第一个整数并存入切片中
	for i := 1; i < 10; i++ {  // 循环读取下一个9个整数
		numbers[i] = readInteger("NEXT NUMBER:")  // 读取下一个整数并存入切片中
	}

	return numbers  // 返回包含10个整数的切片
}

func printSummary(correct int) {
	fmt.Println()  // 打印空行

	if correct > 10 {  // 如果正确猜对的数字大于10
		fmt.Println()  // 打印空行
		fmt.Println("I GUESSED MORE THAN 1/3 OF YOUR NUMBERS.")  // 打印消息
		fmt.Println("I WIN.\u0007")  // 打印消息并发出响铃
	} else if correct < 10 {  // 如果正确猜对的数字小于10
		fmt.Println("I GUESSED LESS THAN 1/3 OF YOUR NUMBERS.")  // 打印消息
		fmt.Println("YOU BEAT ME.  CONGRATULATIONS *****")  // 打印消息
	} else {  // 其他情况
		fmt.Println("I GUESSED EXACTLY 1/3 OF YOUR NUMBERS.")  // 打印消息
		fmt.Println("IT'S A TIE GAME.")  // 打印消息
	}
}

func buildArray(val, row, col int) [][]int {  // 定义一个名为buildArray的函数，接受三个整数参数，返回一个二维整数数组
	a := make([][]int, row)  // 创建一个包含row个元素的切片，每个元素是一个整数切片
	for r := 0; r < row; r++ {  // 遍历行
		b := make([]int, col)  // 创建一个包含col个元素的整数切片
		for c := 0; c < col; c++ {  // 遍历列
			b[c] = val  # 将变量val赋值给数组b的索引为c的元素
		}
		a[r] = b  # 将数组b赋值给数组a的索引为r的元素
	}
	return a  # 返回数组a
}

func main() {
	rand.Seed(time.Now().UnixNano())  # 使用当前时间的纳秒数作为随机数种子

	printIntro()  # 调用打印介绍的函数
	if readInteger("FOR INSTRUCTIONS, TYPE '1', ELSE TYPE '0' ? ") == 1 {  # 调用读取整数的函数，根据用户输入的值判断是否打印说明
		printInstructions()  # 调用打印说明的函数
	}

	a := 0  # 初始化变量a为0
	b := 1  # 初始化变量b为1
	c := 3  # 初始化变量c为3

	m := buildArray(1, 27, 3)  # 调用构建数组的函数，传入参数1, 27, 3，并将结果赋值给变量m
	k := buildArray(9, 3, 3)  # 创建一个 9x3x3 的数组 k
	l := buildArray(3, 9, 3)  # 创建一个 3x9x3 的数组 l

	for {  # 无限循环
		l[0][0] = 2  # 将 l 数组中的第一个元素设置为 2
		l[4][1] = 2  # 将 l 数组中的第五个元素设置为 2
		l[8][2] = 2  # 将 l 数组中的第九个元素设置为 2

		z := float64(26)  # 创建一个浮点数变量 z 并赋值为 26
		z1 := float64(8)  # 创建一个浮点数变量 z1 并赋值为 8
		z2 := 2  # 创建一个整数变量 z2 并赋值为 2
		runningCorrect := 0  # 创建一个整数变量 runningCorrect 并赋值为 0

		var numbers []int  # 创建一个整数数组 numbers

		for round := 1; round <= 4; round++ {  # 循环 4 次
			validNumbers := false  # 创建一个布尔变量 validNumbers 并赋值为 false
			for !validNumbers {  # 当 validNumbers 为 false 时执行循环
				numbers = readTenNumbers()  # 调用 readTenNumbers 函数并将返回值赋给 numbers
				validNumbers = true  # 将 validNumbers 设置为 true
				for _, n := range numbers {  # 遍历 numbers 数组
					if n < 0 || n > 2 {  // 检查数字是否小于0或大于2
						fmt.Println("ONLY USE THE DIGITS '0', '1', OR '2'.")  // 打印错误提示信息
						fmt.Println("LET'S TRY AGAIN.")  // 打印提示信息
						validNumbers = false  // 将validNumbers标记为false
						break  // 跳出循环
					}
				}
			}

			fmt.Printf("\n%-14s%-14s%-14s%-14s\n", "MY GUESS", "YOUR NO.", "RESULT", "NO. RIGHT")  // 打印表头

			for _, n := range numbers {  // 遍历numbers切片
				s := 0  // 初始化s为0
				myGuess := 0  // 初始化myGuess为0

				for j := 0; j < 3; j++ {  // 循环3次
					s1 := a*k[z2][j] + b*l[int(z1)][j] + c*m[int(z)][j]  // 计算s1的值

					if s < s1 {  // 比较s和s1的大小
						s = s1  // 将s的值更新为s1
					myGuess = j  # 将变量 j 的值赋给变量 myGuess
				} else if s1 == s && rand.Float64() > 0.5 {  # 如果 s1 等于 s 并且随机生成的浮点数大于 0.5
					myGuess = j  # 将变量 j 的值赋给变量 myGuess
				}
			}
			result := ""  # 初始化变量 result 为空字符串

			if myGuess != n {  # 如果 myGuess 不等于 n
				result = "WRONG"  # 将 result 设置为 "WRONG"
			} else {
				runningCorrect += 1  # runningCorrect 加 1
				result = "RIGHT"  # 将 result 设置为 "RIGHT"
				m[int(z)][n] = m[int(z)][n] + 1  # 将 m[int(z)][n] 的值加 1
				l[int(z1)][n] = l[int(z1)][n] + 1  # 将 l[int(z1)][n] 的值加 1
				k[int(z2)][n] = k[int(z2)][n] + 1  # 将 k[int(z2)][n] 的值加 1
				z = z - (z/9)*9  # z 减去 z 除以 9 的整数部分乘以 9
				z = 3.0*z + float64(n)  # z 乘以 3.0 再加上 n 的浮点数值
			}
			fmt.Printf("\n%-14d%-14d%-14s%-14d\n", myGuess, n, result, runningCorrect)  # 格式化输出 myGuess, n, result, runningCorrect
				z1 = z - (z/9)*9  # 计算 z 除以 9 的余数
				z2 = n  # 将 n 赋值给 z2
			}
			printSummary(runningCorrect)  # 调用 printSummary 函数，打印运行结果
			if readInteger("\nDO YOU WANT TO TRY AGAIN (1 FOR YES, 0 FOR NO) ? ") != 1 {  # 如果用户输入的不是 1，则执行下面的代码
				fmt.Println("\nTHANKS FOR THE GAME.")  # 打印感谢信息
				os.Exit(0)  # 退出程序
			}
		}
	}
}
```