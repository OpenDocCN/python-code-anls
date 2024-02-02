# `basic-computer-games\00_Alternate_Languages\20_Buzzword\go\main.go`

```py
package main

import (
    "bufio"  // 导入 bufio 包，用于读取输入
    "fmt"    // 导入 fmt 包，用于格式化输出
    "math/rand"  // 导入 math/rand 包，用于生成随机数
    "os"     // 导入 os 包，提供对操作系统功能的访问
    "strings"  // 导入 strings 包，提供对字符串的操作
    "time"   // 导入 time 包，提供时间相关的功能
)

func main() {
    rand.Seed(time.Now().UnixNano())  // 使用当前时间的纳秒数作为随机数种子
    words := [][]string{  // 定义一个二维字符串数组
        {
            "Ability",
            "Basal",
            "Behavioral",
            "Child-centered",
            "Differentiated",
            "Discovery",
            "Flexible",
            "Heterogeneous",
            "Homogenous",
            "Manipulative",
            "Modular",
            "Tavistock",
            "Individualized",
        }, {
            "learning",
            "evaluative",
            "objective",
            "cognitive",
            "enrichment",
            "scheduling",
            "humanistic",
            "integrated",
            "non-graded",
            "training",
            "vertical age",
            "motivational",
            "creative",
        }, {
            "grouping",
            "modification",
            "accountability",
            "process",
            "core curriculum",
            "algorithm",
            "performance",
            "reinforcement",
            "open classroom",
            "resource",
            "structure",
            "facility",
            "environment",
        },
    }

    scanner := bufio.NewScanner(os.Stdin)  // 创建一个从标准输入读取数据的 Scanner 对象

    // Display intro text
    fmt.Println("\n           Buzzword Generator")  // 打印标题
    fmt.Println("Creative Computing  Morristown, New Jersey")  // 打印创意计算的地点
    fmt.Println("\n\n")  // 打印空行
    fmt.Println("This program prints highly acceptable phrases in")  // 打印提示信息
    fmt.Println("'educator-speak' that you can work into reports")  // 打印提示信息
    fmt.Println("and speeches.  Whenever a question mark is printed,")  // 打印提示信息
    fmt.Println("type a 'Y' for another phrase or 'N' to quit.")  // 打印提示信息
    fmt.Println("\n\nHere's the first phrase:")  // 打印提示信息
}
    // 无限循环，生成短语
    for {
        // 初始化短语为空字符串
        phrase := ""
        // 遍历单词列表
        for _, section := range words {
            // 如果短语长度大于0，添加空格
            if len(phrase) > 0 {
                phrase += " "
            }
            // 随机选择一个单词添加到短语中
            phrase += section[rand.Intn(len(section))]
        }
        // 打印生成的短语
        fmt.Println(phrase)
        // 打印空行
        fmt.Println()

        // 是否继续生成短语？
        fmt.Println("?")
        // 读取用户输入
        scanner.Scan()
        // 如果用户输入的第一个字符不是"Y"，跳出循环
        if strings.ToUpper(scanner.Text())[0:1] != "Y" {
            break
        }
    }
    // 打印结束语
    fmt.Println("Come back when you need help with another report!")
# 闭合前面的函数定义
```