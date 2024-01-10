# `basic-computer-games\00_Alternate_Languages\41_Guess\go\main.go`

```
package main

import (
    "bufio"  // 导入用于读取输入的包
    "fmt"    // 导入用于格式化输出的包
    "math"   // 导入数学函数的包
    "math/rand"  // 导入随机数生成的包
    "os"     // 导入操作系统功能的包
    "strconv"  // 导入字符串转换为数字的包
    "time"   // 导入时间相关的包
)

func printIntro() {
    fmt.Println("                   Guess")  // 输出游戏标题
    fmt.Println("Creative Computing  Morristown, New Jersey")  // 输出游戏信息
    fmt.Println()  // 输出空行
    fmt.Println()  // 输出空行
    fmt.Println()  // 输出空行
    fmt.Println("This is a number guessing game. I'll think")  // 输出游戏说明
    fmt.Println("of a number between 1 and any limit you want.")  // 输出游戏说明
    fmt.Println("Then you have to guess what it is")  // 输出游戏说明
}

func getLimit() (int, int) {
    scanner := bufio.NewScanner(os.Stdin)  // 创建用于读取输入的扫描器

    for {
        fmt.Println("What limit do you want?")  // 提示用户输入限制值
        scanner.Scan()  // 读取用户输入

        limit, err := strconv.Atoi(scanner.Text())  // 将用户输入的字符串转换为整数
        if err != nil || limit < 0 {  // 如果转换出错或者输入值小于0
            fmt.Println("Please enter a number greater or equal to 1")  // 提示用户重新输入
            continue  // 继续循环
        }

        limitGoal := int((math.Log(float64(limit)) / math.Log(2)) + 1)  // 计算猜测次数的上限
        return limit, limitGoal  // 返回限制值和猜测次数的上限
    }

}

func main() {
    rand.Seed(time.Now().UnixNano())  // 使用当前时间作为随机数种子
    printIntro()  // 调用打印游戏介绍的函数

    scanner := bufio.NewScanner(os.Stdin)  // 创建用于读取输入的扫描器

    limit, limitGoal := getLimit()  // 调用获取限制值的函数

    guessCount := 1  // 初始化猜测次数为1
    stillGuessing := true  // 初始化是否继续猜测的标志为true
    won := false  // 初始化是否猜中的标志为false
    myGuess := int(float64(limit)*rand.Float64() + 1)  // 生成一个随机数作为答案

    fmt.Printf("I'm thinking of a number between 1 and %d\n", limit)  // 输出提示猜测范围
    fmt.Println("Now you try to guess what it is.")  // 输出提示玩家开始猜测

    for stillGuessing {  // 进入猜测循环
        scanner.Scan()  // 读取用户输入
        n, err := strconv.Atoi(scanner.Text())  // 将用户输入的字符串转换为整数
        if err != nil {  // 如果转换出错
            fmt.Println("Please enter a number greater or equal to 1")  // 提示用户重新输入
            continue  // 继续循环
        }

        if n < 0 {  // 如果输入值小于0
            break  // 退出循环
        }

        fmt.Print("\n\n\n")  // 输出空行
        if n < myGuess {  // 如果猜测值小于答案
            fmt.Println("Too low. Try a bigger answer")  // 提示猜测值过小
            guessCount += 1  // 猜测次数加1
        } else if n > myGuess {  // 如果猜测值大于答案
            fmt.Println("Too high. Try a smaller answer")  // 提示猜测值过大
            guessCount += 1  // 猜测次数加1
        } else {  // 如果猜测值等于答案
            fmt.Printf("That's it! You got it in %d tries\n", guessCount)  // 输出猜中提示和猜测次数
            won = true  // 设置猜中标志为true
            stillGuessing = false  // 设置继续猜测标志为false
        }
    }
    # 如果游戏胜利
    if won:
        # 如果猜测次数小于目标次数
        if guessCount < limitGoal:
            # 打印"非常好"
            fmt.Println("Very good.")
        # 如果猜测次数等于目标次数
        else if guessCount == limitGoal:
            # 打印"好"
            fmt.Println("Good.")
        # 如果猜测次数大于目标次数
        else:
            # 打印"你应该只需要%d次猜测就能猜中"
            fmt.Printf("You should have been able to get it in only %d guesses.\n", limitGoal)
        # 打印空行
        fmt.Print("\n\n\n")
# 闭合前面的函数定义
```