# `basic-computer-games\00_Alternate_Languages\33_Dice\go\main.go`

```py
package main

import (
    "bufio"  // 导入 bufio 包，用于读取输入
    "fmt"    // 导入 fmt 包，用于格式化输出
    "math/rand"  // 导入 math/rand 包，用于生成随机数
    "os"     // 导入 os 包，用于访问操作系统功能
    "strconv"    // 导入 strconv 包，用于字符串和基本数据类型之间的转换
    "strings"    // 导入 strings 包，用于处理字符串
)

func printWelcome() {
    fmt.Println("\n                   Dice")  // 输出欢迎信息
    fmt.Println("Creative Computing  Morristown, New Jersey")  // 输出欢迎信息
    fmt.Println()  // 输出空行
    fmt.Println()  // 输出空行
    fmt.Println("This program simulates the rolling of a")  // 输出提示信息
    fmt.Println("pair of dice.")  // 输出提示信息
    fmt.Println("You enter the number of times you want the computer to")  // 输出提示信息
    fmt.Println("'roll' the dice.   Watch out, very large numbers take")  // 输出提示信息
    fmt.Println("a long time.  In particular, numbers over 5000.")  // 输出提示信息
    fmt.Println()  // 输出空行
}

func main() {
    printWelcome()  // 调用打印欢迎信息的函数

    scanner := bufio.NewScanner(os.Stdin)  // 创建一个从标准输入读取数据的 Scanner 对象

    for {
        fmt.Println("\nHow many rolls? ")  // 输出提示信息
        scanner.Scan()  // 读取用户输入
        numRolls, err := strconv.Atoi(scanner.Text())  // 将用户输入的字符串转换为整数
        if err != nil {  // 如果转换出错
            fmt.Println("Invalid input, try again...")  // 输出错误提示信息
            continue  // 继续下一次循环
        }

        // We'll track counts of roll outcomes in a 13-element list.
        // The first two indices (0 & 1) are ignored, leaving just
        // the indices that match the roll values (2 through 12).
        results := make([]int, 13)  // 创建一个长度为 13 的整数切片，用于记录掷骰子的结果

        for n := 0; n < numRolls; n++ {  // 循环掷骰子 numRolls 次
            d1 := rand.Intn(6) + 1  // 生成一个1到6的随机数，模拟第一个骰子的点数
            d2 := rand.Intn(6) + 1  // 生成一个1到6的随机数，模拟第二个骰子的点数
            results[d1+d2] += 1  // 根据两个骰子的点数，更新结果切片中对应的索引处的值
        }

        // Display final results
        fmt.Println("\nTotal Spots   Number of Times")  // 输出提示信息
        for i := 2; i < 13; i++ {  // 遍历结果切片
            fmt.Printf(" %-14d%d\n", i, results[i])  // 格式化输出每个点数和出现的次数
        }

        fmt.Println("\nTry again? ")  // 输出提示信息
        scanner.Scan()  // 读取用户输入
        if strings.ToUpper(scanner.Text()) == "Y" {  // 如果用户输入的是"Y"
            continue  // 继续下一次循环
        } else {
            os.Exit(1)  // 退出程序
        }
    }
}
```