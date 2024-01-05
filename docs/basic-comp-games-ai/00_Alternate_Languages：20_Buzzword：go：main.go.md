# `d:/src/tocomm/basic-computer-games\00_Alternate_Languages\20_Buzzword\go\main.go`

```
package main  // 声明包名为 main

import (
	"bufio"  // 导入 bufio 包，提供了缓冲 I/O 的功能
	"fmt"  // 导入 fmt 包，提供了格式化 I/O 的功能
	"math/rand"  // 导入 math/rand 包，提供了伪随机数生成的功能
	"os"  // 导入 os 包，提供了操作系统功能的接口
	"strings"  // 导入 strings 包，提供了操作字符串的功能
	"time"  // 导入 time 包，提供了时间的功能
)

func main() {  // 主函数
	rand.Seed(time.Now().UnixNano())  // 使用当前时间的纳秒数作为随机数种子
	words := [][]string{  // 定义一个二维字符串数组
		{
			"Ability",  // 第一个子数组包含一些单词
			"Basal",
			"Behavioral",
			"Child-centered",
			"Differentiated",
# 创建一个包含字符串的列表
adjectives = [
    "Discovery",  # 探索
    "Flexible",  # 灵活的
    "Heterogeneous",  # 异质的
    "Homogenous",  # 同质的
    "Manipulative",  # 操纵的
    "Modular",  # 模块化的
    "Tavistock",  # 塔维斯托克（地名）
    "Individualized",  # 个性化的
]

nouns = [
    "learning",  # 学习
    "evaluative",  # 评估的
    "objective",  # 客观的
    "cognitive",  # 认知的
    "enrichment",  # 丰富
    "scheduling",  # 安排
    "humanistic",  # 人文主义的
    "integrated",  # 整合的
    "non-graded",  # 非等级的
    "training",  # 培训
    "vertical age",  # 垂直年龄
]
			"motivational",  # 添加了一个字符串 "motivational" 到第一个集合中
			"creative",  # 添加了一个字符串 "creative" 到第一个集合中
		}, {  # 开始第二个集合
			"grouping",  # 添加了一个字符串 "grouping" 到第二个集合中
			"modification",  # 添加了一个字符串 "modification" 到第二个集合中
			"accountability",  # 添加了一个字符串 "accountability" 到第二个集合中
			"process",  # 添加了一个字符串 "process" 到第二个集合中
			"core curriculum",  # 添加了一个字符串 "core curriculum" 到第二个集合中
			"algorithm",  # 添加了一个字符串 "algorithm" 到第二个集合中
			"performance",  # 添加了一个字符串 "performance" 到第二个集合中
			"reinforcement",  # 添加了一个字符串 "reinforcement" 到第二个集合中
			"open classroom",  # 添加了一个字符串 "open classroom" 到第二个集合中
			"resource",  # 添加了一个字符串 "resource" 到第二个集合中
			"structure",  # 添加了一个字符串 "structure" 到第二个集合中
			"facility",  # 添加了一个字符串 "facility" 到第二个集合中
			"environment",  # 添加了一个字符串 "environment" 到第二个集合中
		},
	}

	scanner := bufio.NewScanner(os.Stdin)  # 创建了一个从标准输入读取数据的 Scanner 对象
    // Display intro text
    fmt.Println("\n           Buzzword Generator")  # 显示程序介绍文本
    fmt.Println("Creative Computing  Morristown, New Jersey")  # 显示创意计算的地点
    fmt.Println("\n\n")
    fmt.Println("This program prints highly acceptable phrases in")  # 显示程序功能介绍
    fmt.Println("'educator-speak' that you can work into reports")
    fmt.Println("and speeches.  Whenever a question mark is printed,")
    fmt.Println("type a 'Y' for another phrase or 'N' to quit.")
    fmt.Println("\n\nHere's the first phrase:")

    for {  # 进入循环
        phrase := ""  # 初始化短语为空
        for _, section := range words {  # 遍历单词列表
            if len(phrase) > 0:  # 如果短语长度大于0
                phrase += " "  # 在短语后面添加空格
            phrase += section[rand.Intn(len(section))]  # 从每个部分中随机选择一个单词添加到短语中
        }
        fmt.Println(phrase)  # 打印生成的短语
		fmt.Println()  // 打印空行

		// 继续吗？
		fmt.Println("?")  // 打印提示信息
		scanner.Scan()  // 从标准输入中扫描下一行
		if strings.ToUpper(scanner.Text())[0:1] != "Y" {  // 如果输入的内容不是以Y开头（不是Yes），则跳出循环
			break
		}
	}
	fmt.Println("Come back when you need help with another report!")  // 打印结束提示信息
}
```