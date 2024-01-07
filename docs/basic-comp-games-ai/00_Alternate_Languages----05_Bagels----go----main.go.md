# `basic-computer-games\00_Alternate_Languages\05_Bagels\go\main.go`

```

package main

import (
	"bufio" // 导入 bufio 包，用于读取输入
	"fmt" // 导入 fmt 包，用于格式化输出
	"math/rand" // 导入 math/rand 包，用于生成随机数
	"os" // 导入 os 包，用于操作系统功能
	"strconv" // 导入 strconv 包，用于字符串和数字之间的转换
	"strings" // 导入 strings 包，用于处理字符串
	"time" // 导入 time 包，用于处理时间
)

const MAXGUESSES int = 20 // 定义常量 MAXGUESSES，表示最大猜测次数为 20

func printWelcome() {
	fmt.Println("\n                Bagels") // 打印欢迎信息
	fmt.Println("Creative Computing  Morristown, New Jersey") // 打印创意计算的信息
	fmt.Println()
}
func printRules() {
	fmt.Println() // 打印空行
	fmt.Println("I am thinking of a three-digit number.  Try to guess") // 打印游戏规则
	fmt.Println("my number and I will give you clues as follows:") // 打印游戏规则
	fmt.Println("   PICO   - One digit correct but in the wrong position") // 打印游戏规则
	fmt.Println("   FERMI  - One digit correct and in the right position") // 打印游戏规则
	fmt.Println("   BAGELS - No digits correct") // 打印游戏规则
}

func getNumber() []string {
	numbers := []string{"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"} // 创建包含数字字符串的切片
	rand.Shuffle(len(numbers), func(i, j int) { numbers[i], numbers[j] = numbers[j], numbers[i] }) // 打乱切片中的数字顺序

	return numbers[:3] // 返回打乱顺序后的前三个数字
}

func getValidGuess(guessNumber int) string {
	var guess string // 定义变量 guess，用于存储玩家的猜测
	scanner := bufio.NewScanner(os.Stdin) // 创建用于读取输入的 Scanner 对象
	valid := false // 定义变量 valid，表示猜测是否有效
	for !valid {
		fmt.Printf("Guess # %d?\n", guessNumber) // 提示玩家输入猜测
		scanner.Scan() // 读取玩家输入
		guess = strings.TrimSpace(scanner.Text()) // 去除输入字符串两端的空格

		// 确保猜测为三个字符
		if len(guess) == 3 {
			// 确保猜测为数字
			_, err := strconv.Atoi(guess) // 尝试将猜测转换为数字
			if err != nil {
				fmt.Println("What?") // 如果转换失败，提示玩家重新输入
			} else {
				// 确保猜测中的数字唯一
				if (guess[0:1] != guess[1:2]) && (guess[0:1] != guess[2:3]) && (guess[1:2] != guess[2:3]) {
					valid = true // 如果猜测有效，设置 valid 为 true
				} else {
					fmt.Println("Oh, I forgot to tell you that the number I have in mind") // 如果猜测中有重复数字，提示玩家重新输入
					fmt.Println("has no two digits the same.")
				}
			}
		} else {
			fmt.Println("Try guessing a three-digit number.") // 如果猜测不是三个字符，提示玩家重新输入
		}
	}

	return guess // 返回有效的猜测
}

func buildResultString(num []string, guess string) string {
	result := "" // 定义变量 result，用于存储猜测结果

	// 正确数字但位置错误
	for i := 0; i < 2; i++ {
		if num[i] == guess[i+1:i+2] {
			result += "PICO " // 如果数字正确但位置错误，添加 PICO 到结果中
		}
		if num[i+1] == guess[i:i+1] {
			result += "PICO " // 如果数字正确但位置错误，添加 PICO 到结果中
		}
	}
	if num[0] == guess[2:3] {
		result += "PICO " // 如果数字正确但位置错误，添加 PICO 到结果中
	}
	if num[2] == guess[0:1] {
		result += "PICO " // 如果数字正确但位置错误，添加 PICO 到结果中
	}

	// 正确数字且位置正确
	for i := 0; i < 3; i++ {
		if num[i] == guess[i:i+1] {
			result += "FERMI " // 如果数字正确且位置正确，添加 FERMI 到结果中
		}
	}

	// 没有正确的数字
	if result == "" {
		result = "BAGELS" // 如果没有正确的数字，设置结果为 BAGELS
	}

	return result // 返回猜测结果
}

func main() {
	rand.Seed(time.Now().UnixNano()) // 使用当前时间作为随机数种子
	scanner := bufio.NewScanner(os.Stdin) // 创建用于读取输入的 Scanner 对象

	printWelcome() // 打印欢迎信息

	fmt.Println("Would you like the rules (Yes or No)? ") // 提示玩家是否需要游戏规则
	scanner.Scan() // 读取玩家输入
	response := scanner.Text() // 获取玩家输入的响应
	if len(response) > 0 {
		if strings.ToUpper(response[0:1]) != "N" { // 如果玩家不想要游戏规则
			printRules() // 打印游戏规则
		}
	} else {
		printRules() // 打印游戏规则
	}

	gamesWon := 0 // 定义变量 gamesWon，用于存储赢得的游戏次数
	stillRunning := true // 定义变量 stillRunning，表示游戏是否继续进行

	for stillRunning {
		num := getNumber() // 获取一个三位数作为目标数字
		numStr := strings.Join(num, "") // 将目标数字转换为字符串
		guesses := 1 // 定义变量 guesses，表示猜测次数

		fmt.Println("\nO.K.  I have a number in mind.") // 提示玩家开始猜测
		guessing := true // 定义变量 guessing，表示是否在猜测中
		for guessing {
			guess := getValidGuess(guesses) // 获取有效的玩家猜测

			if guess == numStr { // 如果猜测正确
				fmt.Println("You got it!!") // 提示玩家猜对了
				gamesWon++ // 赢得的游戏次数加一
				guessing = false // 结束猜测
			} else {
				fmt.Println(buildResultString(num, guess)) // 打印猜测结果
				guesses++ // 猜测次数加一
				if guesses > MAXGUESSES { // 如果猜测次数超过最大次数
					fmt.Println("Oh well") // 提示玩家猜测失败
					fmt.Printf("That's %d guesses. My number was %s\n", MAXGUESSES, numStr) // 打印目标数字
					guessing = false // 结束猜测
				}
			}
		}

		validRespone := false // 定义变量 validRespone，表示玩家响应是否有效
		for !validRespone {
			fmt.Println("Play again (Yes or No)?") // 提示玩家是否继续游戏
			scanner.Scan() // 读取玩家输入
			response := scanner.Text() // 获取玩家输入的响应
			if len(response) > 0 {
				validRespone = true // 设置玩家响应为有效
				if strings.ToUpper(response[0:1]) != "Y" { // 如果玩家不想继续游戏
					stillRunning = false // 结束游戏
				}
			}
		}
	}

	if gamesWon > 0 {
		fmt.Printf("\nA %d point Bagels buff!!\n", gamesWon) // 打印赢得的游戏次数
	}

	fmt.Println("Hope you had fun.  Bye") // 结束游戏，打印结束语
}

```