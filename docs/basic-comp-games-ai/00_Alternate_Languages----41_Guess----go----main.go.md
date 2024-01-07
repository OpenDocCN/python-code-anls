# `basic-computer-games\00_Alternate_Languages\41_Guess\go\main.go`

```

package main

import (
	"bufio" // 导入 bufio 包，用于读取输入
	"fmt" // 导入 fmt 包，用于格式化输出
	"math" // 导入 math 包，用于数学计算
	"math/rand" // 导入 math/rand 包，用于生成随机数
	"os" // 导入 os 包，用于操作系统功能
	"strconv" // 导入 strconv 包，用于字符串和基本数据类型之间的转换
	"time" // 导入 time 包，用于时间相关操作
)

func printIntro() {
	// 打印游戏介绍信息
	fmt.Println("                   Guess")
	fmt.Println("Creative Computing  Morristown, New Jersey")
	fmt.Println()
	fmt.Println()
	fmt.Println()
	fmt.Println("This is a number guessing game. I'll think")
	fmt.Println("of a number between 1 and any limit you want.")
	fmt.Println("Then you have to guess what it is")
}

func getLimit() (int, int) {
	// 获取用户输入的限制值
	scanner := bufio.NewScanner(os.Stdin)

	for {
		fmt.Println("What limit do you want?")
		scanner.Scan()

		// 将用户输入的字符串转换为整数
		limit, err := strconv.Atoi(scanner.Text())
		if err != nil || limit < 0 {
			fmt.Println("Please enter a number greater or equal to 1")
			continue
		}

		// 根据用户输入的限制值计算猜测次数的目标值
		limitGoal := int((math.Log(float64(limit)) / math.Log(2)) + 1)
		return limit, limitGoal
	}

}

func main() {
	// 设置随机数种子
	rand.Seed(time.Now().UnixNano())
	// 打印游戏介绍
	printIntro()

	// 获取用户输入的限制值和猜测次数的目标值
	scanner := bufio.NewScanner(os.Stdin)
	limit, limitGoal := getLimit()

	// 初始化猜测次数、是否继续猜测、是否赢得游戏、计算机猜测的数字
	guessCount := 1
	stillGuessing := true
	won := false
	myGuess := int(float64(limit)*rand.Float64() + 1)

	// 打印计算机思考的数字范围
	fmt.Printf("I'm thinking of a number between 1 and %d\n", limit)
	fmt.Println("Now you try to guess what it is.")

	// 循环进行猜测
	for stillGuessing {
		scanner.Scan()
		n, err := strconv.Atoi(scanner.Text())
		if err != nil {
			fmt.Println("Please enter a number greater or equal to 1")
			continue
		}

		if n < 0 {
			break
		}

		fmt.Print("\n\n\n")
		if n < myGuess {
			fmt.Println("Too low. Try a bigger answer")
			guessCount += 1
		} else if n > myGuess {
			fmt.Println("Too high. Try a smaller answer")
			guessCount += 1
		} else {
			fmt.Printf("That's it! You got it in %d tries\n", guessCount)
			won = true
			stillGuessing = false
		}
	}

	// 根据猜测次数判断游戏结果
	if won {
		if guessCount < limitGoal {
			fmt.Println("Very good.")
		} else if guessCount == limitGoal {
			fmt.Println("Good.")
		} else {
			fmt.Printf("You should have been able to get it in only %d guesses.\n", limitGoal)
		}
		fmt.Print("\n\n\n")
	}
}

```