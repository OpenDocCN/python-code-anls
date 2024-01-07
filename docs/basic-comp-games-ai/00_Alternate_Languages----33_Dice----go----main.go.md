# `basic-computer-games\00_Alternate_Languages\33_Dice\go\main.go`

```

package main

import (
	"bufio" // 导入 bufio 包，用于读取输入
	"fmt" // 导入 fmt 包，用于格式化输出
	"math/rand" // 导入 math/rand 包，用于生成随机数
	"os" // 导入 os 包，用于操作系统功能
	"strconv" // 导入 strconv 包，用于字符串和基本数据类型之间的转换
	"strings" // 导入 strings 包，用于处理字符串
)

func printWelcome() {
	// 打印欢迎信息
	fmt.Println("\n                   Dice")
	fmt.Println("Creative Computing  Morristown, New Jersey")
	fmt.Println()
	fmt.Println()
	fmt.Println("This program simulates the rolling of a")
	fmt.Println("pair of dice.")
	fmt.Println("You enter the number of times you want the computer to")
	fmt.Println("'roll' the dice.   Watch out, very large numbers take")
	fmt.Println("a long time.  In particular, numbers over 5000.")
	fmt.Println()
}

func main() {
	printWelcome() // 调用打印欢迎信息的函数
	scanner := bufio.NewScanner(os.Stdin) // 创建一个用于读取输入的 Scanner 对象

	for {
		fmt.Println("\nHow many rolls? ") // 提示用户输入掷骰子的次数
		scanner.Scan() // 读取用户输入
		numRolls, err := strconv.Atoi(scanner.Text()) // 将用户输入的字符串转换为整数
		if err != nil { // 如果转换出错
			fmt.Println("Invalid input, try again...") // 提示用户输入无效，重新输入
			continue
		}

		// 创建一个长度为13的整数切片，用于统计每种掷骰子结果的次数
		results := make([]int, 13)

		for n := 0; n < numRolls; n++ { // 循环掷骰子
			d1 := rand.Intn(6) + 1 // 生成1到6之间的随机数，模拟第一个骰子的结果
			d2 := rand.Intn(6) + 1 // 生成1到6之间的随机数，模拟第二个骰子的结果
			results[d1+d2] += 1 // 统计掷骰子结果的次数
		}

		// 显示最终结果
		fmt.Println("\nTotal Spots   Number of Times")
		for i := 2; i < 13; i++ { // 遍历掷骰子结果的次数
			fmt.Printf(" %-14d%d\n", i, results[i]) // 格式化输出掷骰子结果和次数
		}

		fmt.Println("\nTry again? ") // 提示用户是否再次进行掷骰子
		scanner.Scan() // 读取用户输入
		if strings.ToUpper(scanner.Text()) == "Y" { // 如果用户输入是"Y"
			continue // 继续下一轮掷骰子
		} else {
			os.Exit(1) // 否则退出程序
		}

	}
}

```