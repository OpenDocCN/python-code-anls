# `basic-computer-games\00_Alternate_Languages\47_Hi-Lo\go\main.go`

```

package main

import (
	"bufio" // 导入 bufio 包，用于读取输入
	"fmt" // 导入 fmt 包，用于格式化输出
	"math/rand" // 导入 rand 包，用于生成随机数
	"os" // 导入 os 包，用于访问操作系统功能
	"strconv" // 导入 strconv 包，用于字符串和基本数据类型之间的转换
	"strings" // 导入 strings 包，用于处理字符串
	"time" // 导入 time 包，用于处理时间
)

const MAX_ATTEMPTS = 6 // 定义常量，表示最大尝试次数

func printIntro() {
	// 打印游戏介绍
	fmt.Println("HI LO")
	fmt.Println("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
	fmt.Println("\n\n\nTHIS IS THE GAME OF HI LO.")
	fmt.Println("\nYOU WILL HAVE 6 TRIES TO GUESS THE AMOUNT OF MONEY IN THE")
	fmt.Println("HI LO JACKPOT, WHICH IS BETWEEN 1 AND 100 DOLLARS.  IF YOU")
	fmt.Println("GUESS THE AMOUNT, YOU WIN ALL THE MONEY IN THE JACKPOT!")
	fmt.Println("THEN YOU GET ANOTHER CHANCE TO WIN MORE MONEY.  HOWEVER,")
	fmt.Println("IF YOU DO NOT GUESS THE AMOUNT, THE GAME ENDS.")
	fmt.Println()
	fmt.Println()
}

func main() {
	rand.Seed(time.Now().UnixNano()) // 使用当前时间作为随机数种子
	scanner := bufio.NewScanner(os.Stdin) // 创建一个从标准输入读取数据的 Scanner 对象

	printIntro() // 调用打印游戏介绍的函数

	totalWinnings := 0 // 初始化总奖金为 0

	for {
		fmt.Println()
		secret := rand.Intn(1000) + 1 // 生成一个 1 到 1000 之间的随机数作为秘密数字

		guessedCorrectly := false // 初始化猜测结果为 false

		for attempt := 0; attempt < MAX_ATTEMPTS; attempt++ { // 循环进行最多 6 次猜测
			fmt.Println("YOUR GUESS?") // 提示用户输入猜测
			scanner.Scan() // 读取用户输入
			guess, err := strconv.Atoi(scanner.Text()) // 将用户输入的字符串转换为整数
			if err != nil { // 如果转换出错
				fmt.Println("INVALID INPUT") // 提示输入无效
			}

			if guess == secret { // 如果猜测正确
				fmt.Printf("GOT IT!!!!!!!!!!   YOU WIN %d DOLLARS.\n", secret) // 输出猜测正确的消息
				guessedCorrectly = true // 设置猜测结果为 true
				break // 跳出循环
			} else if guess > secret { // 如果猜测值大于秘密数字
				fmt.Println("YOUR GUESS IS TOO HIGH.") // 提示猜测值过高
			} else { // 否则
				fmt.Println("YOUR GUESS IS TOO LOW.") // 提示猜测值过低
			}
		}

		if guessedCorrectly { // 如果猜测结果为 true
			totalWinnings += secret // 将奖金累加上猜测的数字
			fmt.Printf("YOUR TOTAL WINNINGS ARE NOW $%d.\n", totalWinnings) // 输出当前总奖金
		} else { // 否则
			fmt.Printf("YOU BLEW IT...TOO BAD...THE NUMBER WAS %d\n", secret) // 提示猜测失败
		}

		fmt.Println()
		fmt.Println("PLAYAGAIN (YES OR NO)?") // 提示用户是否再玩一次
		scanner.Scan() // 读取用户输入

		if strings.ToUpper(scanner.Text())[0:1] != "Y" { // 如果用户输入的不是以 Y 开头的字符串
			break // 退出循环
		}
	}
	fmt.Println("\nSO LONG.  HOPE YOU ENJOYED YOURSELF!!!") // 输出结束语
}

```