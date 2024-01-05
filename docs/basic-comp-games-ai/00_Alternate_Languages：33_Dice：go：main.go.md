# `00_Alternate_Languages\33_Dice\go\main.go`

```
package main  // 声明当前文件所属的包为 main

import (
	"bufio"  // 导入 bufio 包，用于提供缓冲 I/O
	"fmt"  // 导入 fmt 包，用于格式化输入输出
	"math/rand"  // 导入 math/rand 包，用于生成随机数
	"os"  // 导入 os 包，提供对操作系统功能的访问
	"strconv"  // 导入 strconv 包，用于字符串和基本数据类型之间的转换
	"strings"  // 导入 strings 包，提供对字符串的操作
)

func printWelcome() {  // 定义一个名为 printWelcome 的函数
	fmt.Println("\n                   Dice")  // 打印输出字符串
	fmt.Println("Creative Computing  Morristown, New Jersey")  // 打印输出字符串
	fmt.Println()  // 打印空行
	fmt.Println()  // 打印空行
	fmt.Println("This program simulates the rolling of a")  // 打印输出字符串
	fmt.Println("pair of dice.")  // 打印输出字符串
	fmt.Println("You enter the number of times you want the computer to")  // 打印输出字符串
	fmt.Println("'roll' the dice.   Watch out, very large numbers take")  // 打印输出字符串
```
```go
	fmt.Println("a long time.  In particular, numbers over 5000.")  # 打印一条消息到标准输出

	fmt.Println()  # 打印一个空行到标准输出
}

func main() {  # 定义一个名为main的函数
	printWelcome()  # 调用printWelcome函数
	scanner := bufio.NewScanner(os.Stdin)  # 创建一个从标准输入读取数据的Scanner对象

	for {  # 进入一个无限循环
		fmt.Println("\nHow many rolls? ")  # 打印一条消息到标准输出
		scanner.Scan()  # 从标准输入读取一行数据
		numRolls, err := strconv.Atoi(scanner.Text())  # 将读取的数据转换为整数并检查是否出错
		if err != nil {  # 如果转换出错
			fmt.Println("Invalid input, try again...")  # 打印一条消息到标准输出
			continue  # 继续下一次循环
		}

		// We'll track counts of roll outcomes in a 13-element list.
		// The first two indices (0 & 1) are ignored, leaving just
		// the indices that match the roll values (2 through 12).
		// 我们将在一个包含13个元素的列表中跟踪掷骰子结果的次数。
		// 前两个索引（0和1）被忽略，只留下与掷骰子值匹配的索引（2到12）。
		results := make([]int, 13)  // 创建一个长度为13的整数切片，用于存储每个点数出现的次数

		for n := 0; n < numRolls; n++ {  // 循环投掷骰子的次数
			d1 := rand.Intn(6) + 1  // 生成1到6之间的随机整数，模拟第一个骰子的点数
			d2 := rand.Intn(6) + 1  // 生成1到6之间的随机整数，模拟第二个骰子的点数
			results[d1+d2] += 1  // 将投掷结果的点数对应的次数加1
		}

		// 显示最终结果
		fmt.Println("\nTotal Spots   Number of Times")
		for i := 2; i < 13; i++ {  // 遍历每个点数出现的次数
			fmt.Printf(" %-14d%d\n", i, results[i])  // 打印点数和出现的次数
		}

		fmt.Println("\nTry again? ")
		scanner.Scan()  // 读取用户输入
		if strings.ToUpper(scanner.Text()) == "Y" {  // 判断用户输入是否为"Y"
			continue  // 如果是，则继续循环
		} else {
			os.Exit(1)  // 如果不是，则退出程序
		# 关闭 ZIP 对象
		zip.close()
		# 返回结果字典
		return fdict
	}
}
```

这段代码是函数的结尾部分，其中包括了关闭 ZIP 对象和返回结果字典的操作。在这里，我们使用注释来解释这两行代码的作用。
```