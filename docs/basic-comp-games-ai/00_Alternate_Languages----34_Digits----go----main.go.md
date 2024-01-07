# `basic-computer-games\00_Alternate_Languages\34_Digits\go\main.go`

```

package main

import (
	"bufio" // 导入用于读取输入的包
	"fmt" // 导入用于格式化输出的包
	"math/rand" // 导入用于生成随机数的包
	"os" // 导入用于操作系统功能的包
	"strconv" // 导入用于字符串转换的包
	"time" // 导入用于处理时间的包
)

func printIntro() {
	// 打印游戏介绍
	fmt.Println("                                DIGITS")
	fmt.Println("              CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
	fmt.Println()
	fmt.Println()
	fmt.Println("THIS IS A GAME OF GUESSING.")
}

func readInteger(prompt string) int {
	// 读取整数输入
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

func printInstructions() {
	// 打印游戏说明
	fmt.Println()
	fmt.Println("PLEASE TAKE A PIECE OF PAPER AND WRITE DOWN")
	// 省略部分打印内容
	fmt.Println("THIRTY TIMES AT RANDOM.")
	fmt.Println("ARRANGE THEM IN THREE LINES OF TEN DIGITS EACH.")
	// 省略部分打印内容
	fmt.Println("I HOPE TO DO BETTER THAN THAT *****")
	fmt.Println()
}

func readTenNumbers() []int {
	// 读取十个数字输入
	numbers := make([]int, 10)

	numbers[0] = readInteger("FIRST NUMBER: ")
	for i := 1; i < 10; i++ {
		numbers[i] = readInteger("NEXT NUMBER:")
	}

	return numbers
}

func printSummary(correct int) {
	// 打印游戏总结
	fmt.Println()

	if correct > 10 {
		// 猜对超过1/3的数字
		fmt.Println()
		fmt.Println("I GUESSED MORE THAN 1/3 OF YOUR NUMBERS.")
		fmt.Println("I WIN.\u0007")
	} else if correct < 10 {
		// 猜对少于1/3的数字
		fmt.Println("I GUESSED LESS THAN 1/3 OF YOUR NUMBERS.")
		fmt.Println("YOU BEAT ME.  CONGRATULATIONS *****")
	} else {
		// 猜对1/3的数字
		fmt.Println("I GUESSED EXACTLY 1/3 OF YOUR NUMBERS.")
		fmt.Println("IT'S A TIE GAME.")
	}
}

func buildArray(val, row, col int) [][]int {
	// 构建二维数组
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
	rand.Seed(time.Now().UnixNano()) // 使用当前时间作为随机数种子

	printIntro() // 打印游戏介绍
	if readInteger("FOR INSTRUCTIONS, TYPE '1', ELSE TYPE '0' ? ") == 1 {
		printInstructions() // 打印游戏说明
	}

	// 省略部分变量声明和数组初始化

	for {
		// 省略部分游戏逻辑
	}
}

```