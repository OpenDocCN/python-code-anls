# `basic-computer-games\00_Alternate_Languages\34_Digits\go\main.go`

```py
package main

import (
    "bufio"  // 导入用于读取输入的包
    "fmt"    // 导入用于格式化输出的包
    "math/rand"  // 导入用于生成随机数的包
    "os"     // 导入用于操作系统功能的包
    "strconv"    // 导入用于字符串转换的包
    "time"   // 导入用于时间相关功能的包
)

func printIntro() {
    fmt.Println("                                DIGITS")  // 打印游戏标题
    fmt.Println("              CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")  // 打印游戏信息
    fmt.Println()  // 打印空行
    fmt.Println()  // 打印空行
    fmt.Println("THIS IS A GAME OF GUESSING.")  // 打印游戏说明
}

func readInteger(prompt string) int {
    scanner := bufio.NewScanner(os.Stdin)  // 创建用于读取输入的扫描器
    for {
        fmt.Println(prompt)  // 打印提示信息
        scanner.Scan()  // 扫描输入
        response, err := strconv.Atoi(scanner.Text())  // 将输入转换为整数

        if err != nil {  // 如果转换出错
            fmt.Println("INVALID INPUT, TRY AGAIN... ")  // 打印错误提示
            continue  // 继续循环，等待有效输入
        }

        return response  // 返回输入的整数
    }
}

func printInstructions() {
    fmt.Println()  // 打印空行
    fmt.Println("PLEASE TAKE A PIECE OF PAPER AND WRITE DOWN")  // 打印游戏说明
    fmt.Println("THE DIGITS '0', '1', OR '2' THIRTY TIMES AT RANDOM.")  // 打印游戏说明
    fmt.Println("ARRANGE THEM IN THREE LINES OF TEN DIGITS EACH.")  // 打印游戏说明
    fmt.Println("I WILL ASK FOR THEN TEN AT A TIME.")  // 打印游戏说明
    fmt.Println("I WILL ALWAYS GUESS THEM FIRST AND THEN LOOK AT YOUR")  // 打印游戏说明
    fmt.Println("NEXT NUMBER TO SEE IF I WAS RIGHT. BY PURE LUCK,")  // 打印游戏说明
    fmt.Println("I OUGHT TO BE RIGHT TEN TIMES. BUT I HOPE TO DO BETTER")  // 打印游戏说明
    fmt.Println("THAN THAT *****")  // 打印游戏说明
    fmt.Println()  // 打印空行
}

func readTenNumbers() []int {
    numbers := make([]int, 10)  // 创建一个包含10个整数的切片

    numbers[0] = readInteger("FIRST NUMBER: ")  // 读取第一个数字
    for i := 1; i < 10; i++ {  // 循环读取剩下的9个数字
        numbers[i] = readInteger("NEXT NUMBER:")  // 读取下一个数字
    }

    return numbers  // 返回包含10个数字的切片
}

func printSummary(correct int) {
    fmt.Println()  // 打印空行

    if correct > 10 {  // 如果猜对的数字超过10个
        fmt.Println()  // 打印空行
        fmt.Println("I GUESSED MORE THAN 1/3 OF YOUR NUMBERS.")  // 打印提示信息
        fmt.Println("I WIN.\u0007")  // 打印提示信息
    } else if correct < 10 {  // 如果猜对的数字少于10个
        fmt.Println("I GUESSED LESS THAN 1/3 OF YOUR NUMBERS.")  // 打印提示信息
        fmt.Println("YOU BEAT ME.  CONGRATULATIONS *****")  // 打印提示信息
    } else {  // 如果猜对的数字等于10个
        fmt.Println("I GUESSED EXACTLY 1/3 OF YOUR NUMBERS.")  // 打印提示信息
        fmt.Println("IT'S A TIE GAME.")  // 打印提示信息
    }
}

func buildArray(val, row, col int) [][]int {
    # 创建一个二维切片，行数为row
    a := make([][]int, row)
    # 遍历每一行
    for r := 0; r < row; r++ {
        # 创建一个长度为col的切片
        b := make([]int, col)
        # 遍历当前行的每一列
        for c := 0; c < col; c++ {
            # 将val赋值给当前列
            b[c] = val
        }
        # 将当前行的切片赋值给二维切片a
        a[r] = b
    }
    # 返回二维切片a
    return a
# 主函数入口
func main() {
    # 设置随机数种子
    rand.Seed(time.Now().UnixNano())

    # 打印游戏介绍
    printIntro()
    # 根据用户输入判断是否打印游戏说明
    if readInteger("FOR INSTRUCTIONS, TYPE '1', ELSE TYPE '0' ? ") == 1 {
        printInstructions()
    }

    # 初始化变量a, b, c
    a := 0
    b := 1
    c := 3

    # 初始化数组m, k, l
    m := buildArray(1, 27, 3)
    k := buildArray(9, 3, 3)
    l := buildArray(3, 9, 3)
    }
}
```