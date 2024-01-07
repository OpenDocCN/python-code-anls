# `basic-computer-games\00_Alternate_Languages\20_Buzzword\go\main.go`

```

package main

import (
	"bufio" // 导入 bufio 包，用于读取输入
	"fmt" // 导入 fmt 包，用于格式化输出
	"math/rand" // 导入 math/rand 包，用于生成随机数
	"os" // 导入 os 包，用于访问操作系统功能
	"strings" // 导入 strings 包，用于处理字符串
	"time" // 导入 time 包，用于处理时间
)

func main() {
	rand.Seed(time.Now().UnixNano()) // 使用当前时间的纳秒数作为随机数种子
	words := [][]string{ // 定义一个二维字符串数组，包含多个短语
		{
			"Ability",
			"Basal",
			"Behavioral",
			// ... 省略部分内容
		}, {
			"learning",
			"evaluative",
			// ... 省略部分内容
		}, {
			"grouping",
			"modification",
			// ... 省略部分内容
		},
	}

	scanner := bufio.NewScanner(os.Stdin) // 创建一个从标准输入读取数据的 Scanner 对象

	// 显示介绍文本
	fmt.Println("\n           Buzzword Generator")
	fmt.Println("Creative Computing  Morristown, New Jersey")
	fmt.Println("\n\n")
	fmt.Println("This program prints highly acceptable phrases in")
	fmt.Println("'educator-speak' that you can work into reports")
	fmt.Println("and speeches.  Whenever a question mark is printed,")
	fmt.Println("type a 'Y' for another phrase or 'N' to quit.")
	fmt.Println("\n\nHere's the first phrase:")

	for {
		phrase := ""
		for _, section := range words { // 遍历短语数组
			if len(phrase) > 0 {
				phrase += " "
			}
			phrase += section[rand.Intn(len(section))] // 随机选择一个短语拼接到当前短语中
		}
		fmt.Println(phrase) // 输出当前短语
		fmt.Println()

		// 是否继续？
		fmt.Println("?")
		scanner.Scan() // 读取用户输入
		if strings.ToUpper(scanner.Text())[0:1] != "Y" { // 判断用户输入是否以 Y 开头
			break // 如果不是，则退出循环
		}
	}
	fmt.Println("Come back when you need help with another report!") // 输出结束语
}

```