# `basic-computer-games\00_Alternate_Languages\47_Hi-Lo\go\main.go`

```py
package main

import (
    "bufio"  // 导入用于读取输入的包
    "fmt"    // 导入用于格式化输出的包
    "math/rand"  // 导入用于生成随机数的包
    "os"     // 导入用于操作系统功能的包
    "strconv"    // 导入用于字符串转换的包
    "strings"    // 导入用于处理字符串的包
    "time"   // 导入用于处理时间的包
)

const MAX_ATTEMPTS = 6  // 定义最大尝试次数为6

func printIntro() {
    fmt.Println("HI LO")  // 输出游戏名称
    fmt.Println("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")  // 输出游戏信息
    fmt.Println("\n\n\nTHIS IS THE GAME OF HI LO.")  // 输出游戏介绍
    fmt.Println("\nYOU WILL HAVE 6 TRIES TO GUESS THE AMOUNT OF MONEY IN THE")  // 输出游戏规则
    fmt.Println("HI LO JACKPOT, WHICH IS BETWEEN 1 AND 100 DOLLARS.  IF YOU")
    fmt.Println("GUESS THE AMOUNT, YOU WIN ALL THE MONEY IN THE JACKPOT!")
    fmt.Println("THEN YOU GET ANOTHER CHANCE TO WIN MORE MONEY.  HOWEVER,")
    fmt.Println("IF YOU DO NOT GUESS THE AMOUNT, THE GAME ENDS.")
    fmt.Println()
    fmt.Println()
}

func main() {
    rand.Seed(time.Now().UnixNano())  // 使用当前时间的纳秒数作为随机数种子
    scanner := bufio.NewScanner(os.Stdin)  // 创建用于读取标准输入的扫描器

    printIntro()  // 调用打印游戏介绍的函数

    totalWinnings := 0  // 初始化总奖金为0
    // 无限循环，直到用户选择不再玩游戏
    for {
        // 打印空行
        fmt.Println()
        // 生成一个1到1000之间的随机数作为秘密数字
        secret := rand.Intn(1000) + 1

        // 初始化猜对标志为false
        guessedCorrectly := false

        // 循环让用户猜数字，最多允许MAX_ATTEMPTS次
        for attempt := 0; attempt < MAX_ATTEMPTS; attempt++ {
            // 提示用户输入猜测的数字
            fmt.Println("YOUR GUESS?")
            scanner.Scan()
            // 将用户输入的字符串转换为整数
            guess, err := strconv.Atoi(scanner.Text())
            // 如果转换出错，提示用户输入无效
            if err != nil {
                fmt.Println("INVALID INPUT")
            }

            // 判断用户猜测的数字与秘密数字的关系
            if guess == secret {
                // 如果猜对了，打印提示信息，设置猜对标志为true，并跳出循环
                fmt.Printf("GOT IT!!!!!!!!!!   YOU WIN %d DOLLARS.\n", secret)
                guessedCorrectly = true
                break
            } else if guess > secret {
                // 如果猜测的数字大于秘密数字，提示用户猜测的数字过高
                fmt.Println("YOUR GUESS IS TOO HIGH.")
            } else {
                // 如果猜测的数字小于秘密数字，提示用户猜测的数字过低
                fmt.Println("YOUR GUESS IS TOO LOW.")
            }
        }

        // 根据用户是否猜对，更新总奖金并打印相应信息
        if guessedCorrectly {
            totalWinnings += secret
            fmt.Printf("YOUR TOTAL WINNINGS ARE NOW $%d.\n", totalWinnings)
        } else {
            fmt.Printf("YOU BLEW IT...TOO BAD...THE NUMBER WAS %d\n", secret)
        }

        // 提示用户是否再玩一次
        fmt.Println()
        fmt.Println("PLAYAGAIN (YES OR NO)?")
        scanner.Scan()

        // 如果用户输入的不是以Y开头的字符串，跳出循环
        if strings.ToUpper(scanner.Text())[0:1] != "Y" {
            break
        }
    }
    // 打印结束游戏的提示信息
    fmt.Println("\nSO LONG.  HOPE YOU ENJOYED YOURSELF!!!")
# 闭合前面的函数定义
```