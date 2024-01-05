# `00_Alternate_Languages\05_Bagels\go\main.go`

```
package main  // 声明包名为 main

import (  // 导入所需的包
	"bufio"  // 用于读取输入
	"fmt"  // 用于格式化输出
	"math/rand"  // 用于生成随机数
	"os"  // 提供对操作系统功能的访问
	"strconv"  // 用于字符串和数字之间的转换
	"strings"  // 提供对字符串的操作
	"time"  // 提供时间相关的功能
)

const MAXGUESSES int = 20  // 声明一个常量 MAXGUESSES，并赋值为 20

func printWelcome() {  // 定义一个名为 printWelcome 的函数
	fmt.Println("\n                Bagels")  // 打印欢迎信息
	fmt.Println("Creative Computing  Morristown, New Jersey")  // 打印创意计算的信息
	fmt.Println()  // 打印空行
}
func printRules() {  // 定义一个名为 printRules 的函数
	fmt.Println()  # 打印空行
	fmt.Println("I am thinking of a three-digit number.  Try to guess")  # 打印提示信息
	fmt.Println("my number and I will give you clues as follows:")  # 打印提示信息
	fmt.Println("   PICO   - One digit correct but in the wrong position")  # 打印提示信息
	fmt.Println("   FERMI  - One digit correct and in the right position")  # 打印提示信息
	fmt.Println("   BAGELS - No digits correct")  # 打印提示信息
}  # 结束函数定义

func getNumber() []string {  # 定义函数，返回一个字符串数组
	numbers := []string{"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}  # 创建包含数字字符串的数组
	rand.Shuffle(len(numbers), func(i, j int) { numbers[i], numbers[j] = numbers[j], numbers[i] })  # 使用随机函数打乱数组顺序

	return numbers[:3]  # 返回数组的前三个元素
}  # 结束函数定义

func getValidGuess(guessNumber int) string {  # 定义函数，接受一个整数参数，返回一个字符串
	var guess string  # 声明一个字符串变量
	scanner := bufio.NewScanner(os.Stdin)  # 创建一个从标准输入读取数据的扫描器
	valid := false  # 初始化一个布尔变量为假
	for !valid {  # 循环直到 valid 变为真
		// 打印猜测的次数
		fmt.Printf("Guess # %d?\n", guessNumber)
		// 从标准输入中读取用户的猜测
		scanner.Scan()
		guess = strings.TrimSpace(scanner.Text())

		// 猜测必须是3个字符
		if len(guess) == 3 {
			// 并且应该是数字
			_, err := strconv.Atoi(guess)
			if err != nil {
				fmt.Println("What?")
			} else {
				// 并且数字应该是唯一的
				if (guess[0:1] != guess[1:2]) && (guess[0:1] != guess[2:3]) && (guess[1:2] != guess[2:3]) {
					valid = true
				} else {
					fmt.Println("Oh, I forgot to tell you that the number I have in mind")
					fmt.Println("has no two digits the same.")
				}
			}
		} else {
			fmt.Println("Try guessing a three-digit number.")  // 打印提示信息，提示用户猜测一个三位数
		}
	}

	return guess  // 返回用户猜测的结果
}

func buildResultString(num []string, guess string) string {
	result := ""  // 初始化结果字符串为空

	// correct digits in wrong place
	for i := 0; i < 2; i++ {  // 遍历数字和猜测结果的每一位
		if num[i] == guess[i+1:i+2] {  // 如果数字和猜测结果的当前位和下一位相等
			result += "PICO "  // 在结果字符串中添加"PICO "
		}
		if num[i+1] == guess[i:i+1] {  // 如果数字和猜测结果的下一位和当前位相等
			result += "PICO "  // 在结果字符串中添加"PICO "
		}
	}
	if num[0] == guess[2:3] {  // 如果数字的第一位和猜测结果的最后一位相等
		result += "PICO "  // 如果猜测的数字中有一个数字与目标数字的第一个数字相同，则将 "PICO " 添加到结果字符串中
	}
	if num[2] == guess[0:1] {  // 如果猜测的数字的第一个数字与目标数字的最后一个数字相同，则将 "PICO " 添加到结果字符串中
		result += "PICO "
	}

	// correct digits in right place
	for i := 0; i < 3; i++ {  // 遍历猜测的数字和目标数字的每一位
		if num[i] == guess[i:i+1] {  // 如果猜测的数字和目标数字的当前位相同
			result += "FERMI "  // 将 "FERMI " 添加到结果字符串中
		}
	}

	// nothing right?
	if result == "" {  // 如果结果字符串为空
		result = "BAGELS"  // 将结果字符串设置为 "BAGELS"
	}

	return result  // 返回结果字符串
}
func main() {
	// 设置随机数种子
	rand.Seed(time.Now().UnixNano())
	// 创建标准输入的扫描器
	scanner := bufio.NewScanner(os.Stdin)

	// 打印欢迎信息
	printWelcome()

	// 询问是否需要游戏规则
	fmt.Println("Would you like the rules (Yes or No)? ")
	// 扫描用户输入
	scanner.Scan()
	// 获取用户输入的响应
	response := scanner.Text()
	// 如果用户输入不为空
	if len(response) > 0 {
		// 如果用户输入的第一个字符不是"N"
		if strings.ToUpper(response[0:1]) != "N" {
			// 打印游戏规则
			printRules()
		}
	} else {
		// 如果用户输入为空，打印游戏规则
		printRules()
	}

	// 初始化游戏获胜次数
	gamesWon := 0
	// 游戏仍在运行
	stillRunning := true
}
		for stillRunning {  # 使用 for 循环来检查程序是否仍在运行
		num := getNumber()  # 调用 getNumber 函数来获取一个数字
		numStr := strings.Join(num, "")  # 将获取的数字转换为字符串
		guesses := 1  # 初始化猜测次数为1

		fmt.Println("\nO.K.  I have a number in mind.")  # 打印消息
		guessing := true  # 初始化猜测状态为真
		for guessing {  # 使用 for 循环来进行猜测
			guess := getValidGuess(guesses)  # 调用 getValidGuess 函数来获取有效的猜测

			if guess == numStr {  # 检查猜测是否与数字相等
				fmt.Println("You got it!!")  # 打印消息
				gamesWon++  # 增加游戏胜利次数
				guessing = false  # 设置猜测状态为假，结束猜测循环
			} else {
				fmt.Println(buildResultString(num, guess))  # 调用 buildResultString 函数来构建结果字符串并打印
				guesses++  # 增加猜测次数
				if guesses > MAXGUESSES {  # 检查猜测次数是否超过最大次数
					fmt.Println("Oh well")  # 打印消息
					fmt.Printf("That's %d guesses. My number was %s\n", MAXGUESSES, numStr)  # 打印猜测次数和正确数字
					guessing = false  # 将猜测状态设置为假，结束猜测
				}
			}
		}

		validRespone := false  # 初始化有效回答为假
		for !validRespone:  # 循环直到得到有效回答
			fmt.Println("Play again (Yes or No)?")  # 打印提示信息
			scanner.Scan()  # 扫描用户输入
			response := scanner.Text()  # 获取用户输入的回答
			if len(response) > 0:  # 如果回答不为空
				validRespone = true  # 将有效回答设置为真
				if strings.ToUpper(response[0:1]) != "Y":  # 如果回答的第一个字符不是Y
					stillRunning = false  # 将游戏状态设置为假，结束游戏
				}
			}
		}
	}
	if gamesWon > 0 {  # 如果游戏获胜次数大于0
		fmt.Printf("\nA %d point Bagels buff!!\n", gamesWon)  # 打印获胜次数对应的消息
	}

	fmt.Println("Hope you had fun.  Bye")  # 打印结束游戏的消息
}
```