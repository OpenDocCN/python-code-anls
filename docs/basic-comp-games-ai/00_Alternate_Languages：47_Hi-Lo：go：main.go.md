# `d:/src/tocomm/basic-computer-games\00_Alternate_Languages\47_Hi-Lo\go\main.go`

```
package main  // 声明包名为 main

import (
	"bufio"  // 导入 bufio 包，提供了缓冲 I/O 的功能
	"fmt"  // 导入 fmt 包，提供了格式化 I/O 的功能
	"math/rand"  // 导入 math/rand 包，提供了伪随机数生成的功能
	"os"  // 导入 os 包，提供了操作系统功能的接口
	"strconv"  // 导入 strconv 包，提供了字符串和基本数据类型之间的转换功能
	"strings"  // 导入 strings 包，提供了操作字符串的功能
	"time"  // 导入 time 包，提供了时间的功能
)

const MAX_ATTEMPTS = 6  // 声明常量 MAX_ATTEMPTS，值为 6

func printIntro() {  // 定义函数 printIntro
	fmt.Println("HI LO")  // 打印字符串 "HI LO"
	fmt.Println("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")  // 打印字符串 "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"
	fmt.Println("\n\n\nTHIS IS THE GAME OF HI LO.")  // 打印字符串 "THIS IS THE GAME OF HI LO."
	fmt.Println("\nYOU WILL HAVE 6 TRIES TO GUESS THE AMOUNT OF MONEY IN THE")  // 打印字符串 "YOU WILL HAVE 6 TRIES TO GUESS THE AMOUNT OF MONEY IN THE"
	fmt.Println("HI LO JACKPOT, WHICH IS BETWEEN 1 AND 100 DOLLARS.  IF YOU")  // 打印字符串 "HI LO JACKPOT, WHICH IS BETWEEN 1 AND 100 DOLLARS.  IF YOU"
	fmt.Println("GUESS THE AMOUNT, YOU WIN ALL THE MONEY IN THE JACKPOT!")  // 打印字符串，提示玩家猜测金额
	fmt.Println("THEN YOU GET ANOTHER CHANCE TO WIN MORE MONEY.  HOWEVER,")  // 打印字符串，提示玩家有机会赢得更多的钱
	fmt.Println("IF YOU DO NOT GUESS THE AMOUNT, THE GAME ENDS.")  // 打印字符串，提示玩家如果没有猜对金额，游戏结束
	fmt.Println()  // 打印空行
	fmt.Println()  // 打印空行
}

func main() {
	rand.Seed(time.Now().UnixNano())  // 使用当前时间的纳秒数作为随机数种子
	scanner := bufio.NewScanner(os.Stdin)  // 创建一个从标准输入读取数据的 Scanner 对象

	printIntro()  // 调用函数，打印游戏介绍

	totalWinnings := 0  // 初始化总奖金为 0

	for {  // 进入无限循环
		fmt.Println()  // 打印空行
		secret := rand.Intn(1000) + 1  // 生成一个 1 到 1000 之间的随机数作为秘密金额

		guessedCorrectly := false  // 初始化猜测是否正确的标志为 false
		for attempt := 0; attempt < MAX_ATTEMPTS; attempt++ {
			# 打印提示信息，要求用户输入猜测的数字
			fmt.Println("YOUR GUESS?")
			# 从标准输入中读取用户输入的内容
			scanner.Scan()
			# 将用户输入的内容转换为整数类型
			guess, err := strconv.Atoi(scanner.Text())
			# 如果转换过程中出现错误，打印提示信息
			if err != nil {
				fmt.Println("INVALID INPUT")
			}

			# 判断用户猜测的数字是否与秘密数字相等
			if guess == secret {
				# 如果相等，打印中奖信息，并结束循环
				fmt.Printf("GOT IT!!!!!!!!!!   YOU WIN %d DOLLARS.\n", secret)
				guessedCorrectly = true
				break
			} else if guess > secret {
				# 如果猜测的数字大于秘密数字，打印提示信息
				fmt.Println("YOUR GUESS IS TOO HIGH.")
			} else {
				# 如果猜测的数字小于秘密数字，打印提示信息
				fmt.Println("YOUR GUESS IS TOO LOW.")
			}
		}
		# 如果猜对了数字
		if guessedCorrectly {
			# 将奖金加上猜对的数字
			totalWinnings += secret
			# 打印出当前的总奖金
			fmt.Printf("YOUR TOTAL WINNINGS ARE NOW $%d.\n", totalWinnings)
		} else {
			# 如果猜错了，打印出正确的数字
			fmt.Printf("YOU BLEW IT...TOO BAD...THE NUMBER WAS %d\n", secret)
		}

		# 打印空行
		fmt.Println()
		# 打印出是否要再玩一次的提示
		fmt.Println("PLAYAGAIN (YES OR NO)?")
		# 读取用户输入
		scanner.Scan()

		# 如果用户输入的不是以"Y"开头的字符串，则跳出循环
		if strings.ToUpper(scanner.Text())[0:1] != "Y" {
			break
		}
	}
	# 打印结束语
	fmt.Println("\nSO LONG.  HOPE YOU ENJOYED YOURSELF!!!")
}
```