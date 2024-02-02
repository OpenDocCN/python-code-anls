# `basic-computer-games\00_Alternate_Languages\25_Chief\go\main.go`

```py
package main

import (
    "bufio"  // 导入 bufio 包，提供了读写数据的缓冲区
    "fmt"    // 导入 fmt 包，提供了格式化 I/O 函数
    "os"     // 导入 os 包，提供了操作系统函数
    "strconv"    // 导入 strconv 包，提供了字符串和基本数据类型之间的转换函数
    "strings"    // 导入 strings 包，提供了操作字符串的函数
)

func printLightning() {
    fmt.Println("************************************")  // 打印分隔线
    n := 24  // 初始化变量 n 为 24
    for n > 16 {  // 当 n 大于 16 时执行循环
        var b strings.Builder  // 创建字符串构建器
        b.Grow(n + 3)  // 设置构建器的容量
        for i := 0; i < n; i++ {  // 循环向构建器中添加空格
            b.WriteString(" ")
        }
        b.WriteString("x x")  // 向构建器中添加字符串
        fmt.Println(b.String())  // 打印构建器中的字符串
        n--  // n 减一
    }
    fmt.Println("                x xxx")  // 打印指定字符串
    fmt.Println("               x   x")  // 打印指定字符串
    fmt.Println("              xx xx")  // 打印指定字符串
    n--  // n 减一
    for n > 8 {  // 当 n 大于 8 时执行循环
        var b strings.Builder  // 创建字符串构建器
        b.Grow(n + 3)  // 设置构建器的容量
        for i := 0; i < n; i++ {  // 循环向构建器中添加空格
            b.WriteString(" ")
        }
        b.WriteString("x x")  // 向构建器中添加字符串
        fmt.Println(b.String())  // 打印构建器中的字符串
        n--  // n 减一
    }
    fmt.Println("        xx")  // 打印指定字符串
    fmt.Println("       x")  // 打印指定字符串
    fmt.Println("************************************")  // 打印分隔线
}

func printSolution(n float64) {
    fmt.Printf("\n%f plus 3 gives %f. This divided by 5 equals %f\n", n, n+3, (n+3)/5)  // 格式化打印字符串
    fmt.Printf("This times 8 gives %f. If we divide 5 and add 5.\n", ((n+3)/5)*8)  // 格式化打印字符串
    fmt.Printf("We get %f, which, minus 1 equals %f\n", (((n+3)/5)*8)/5+5, ((((n+3)/5)*8)/5+5)-1)  // 格式化打印字符串
}

func play() {
    fmt.Println("\nTake a Number and ADD 3. Now, Divide this number by 5 and")  // 打印指定字符串
    fmt.Println("multiply by 8. Now, Divide by 5 and add the same. Subtract 1")  // 打印指定字符串

    youHave := getFloat("\nWhat do you have?")  // 调用 getFloat 函数获取用户输入的浮点数
    compGuess := (((youHave-4)*5)/8)*5 - 3  // 计算猜测的结果
    if getYesNo(fmt.Sprintf("\nI bet your number was %f was I right(Yes or No)? ", compGuess)) {  // 调用 getYesNo 函数获取用户输入的 Yes 或 No
        fmt.Println("\nHuh, I knew I was unbeatable")  // 打印指定字符串
        fmt.Println("And here is how i did it")  // 打印指定字符串
        printSolution(compGuess)  // 调用 printSolution 函数打印计算结果
    } else {
        // 如果猜测错误，提示用户输入原始数字
        originalNumber := getFloat("\nHUH!! what was you original number? ")
        // 如果用户输入的原始数字与计算的结果相等，输出相应信息
        if originalNumber == compGuess {
            fmt.Println("\nThat was my guess, AHA i was right")
            fmt.Println("Shamed to accept defeat i guess, don't worry you can master mathematics too")
            fmt.Println("Here is how i did it")
            // 输出计算结果的解决方案
            printSolution(compGuess)
        } else {
            fmt.Println("\nSo you think you're so smart, EH?")
            fmt.Println("Now, Watch")
            // 输出用户输入的原始数字的解决方案
            printSolution(originalNumber)

            // 提示用户是否相信计算结果
            if getYesNo("\nNow do you believe me? ") {
                print("\nOk, Lets play again sometime bye!!!!")
            } else {
                fmt.Println("\nYOU HAVE MADE ME VERY MAD!!!!!")
                fmt.Println("BY THE WRATH OF THE MATHEMATICS AND THE RAGE OF THE GODS")
                fmt.Println("THERE SHALL BE LIGHTNING!!!!!!!")
                // 输出闪电效果
                printLightning()
                fmt.Println("\nI Hope you believe me now, for your own sake")
            }
        }
    }
# 定义一个函数，用于获取用户输入的浮点数
func getFloat(prompt string) float64 {
    # 创建一个标准输入的扫描器
    scanner := bufio.NewScanner(os.Stdin)

    # 循环直到获取有效的输入
    for {
        # 打印提示信息
        fmt.Println(prompt)
        # 扫描用户输入
        scanner.Scan()
        # 尝试将用户输入的文本转换为浮点数
        val, err := strconv.ParseFloat(scanner.Text(), 64)
        # 如果转换出错，则提示用户重新输入
        if err != nil {
            fmt.Println("INVALID INPUT, TRY AGAIN")
            continue
        }
        # 返回有效的浮点数
        return val
    }
}

# 定义一个函数，用于获取用户输入的是或否
func getYesNo(prompt string) bool {
    # 创建一个标准输入的扫描器
    scanner := bufio.NewScanner(os.Stdin)
    # 打印提示信息
    fmt.Println(prompt)
    # 扫描用户输入
    scanner.Scan()
    # 将用户输入的文本转换为大写，并取第一个字符进行判断是否为"Y"
    return (strings.ToUpper(scanner.Text())[0:1] == "Y")
}

# 主函数
func main() {
    # 打印欢迎信息
    fmt.Println("I am CHIEF NUMBERS FREEK, The GREAT INDIAN MATH GOD.")

    # 如果用户准备好参加测试，则调用play函数，否则打印再见信息
    if getYesNo("\nAre you ready to take the test you called me out for(Yes or No)? ") {
        play()
    } else {
        fmt.Println("Ok, Nevermind. Let me go back to my great slumber, Bye")
    }
}
```