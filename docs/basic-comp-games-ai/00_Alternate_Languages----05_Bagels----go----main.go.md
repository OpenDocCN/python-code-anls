# `basic-computer-games\00_Alternate_Languages\05_Bagels\go\main.go`

```
package main

import (
    "bufio"  // 导入 bufio 包，提供读取输入的功能
    "fmt"    // 导入 fmt 包，提供格式化输入输出的功能
    "math/rand"  // 导入 math/rand 包，提供生成随机数的功能
    "os"     // 导入 os 包，提供操作系统功能
    "strconv"    // 导入 strconv 包，提供字符串和基本数据类型之间的转换功能
    "strings"    // 导入 strings 包，提供处理字符串的功能
    "time"   // 导入 time 包，提供时间相关的功能
)

const MAXGUESSES int = 20  // 定义常量 MAXGUESSES，表示最大猜测次数为 20 次

func printWelcome() {  // 定义函数 printWelcome，用于打印欢迎信息
    fmt.Println("\n                Bagels")  // 打印 Bagels
    fmt.Println("Creative Computing  Morristown, New Jersey")  // 打印 Creative Computing  Morristown, New Jersey
    fmt.Println()  // 打印空行
}

func printRules() {  // 定义函数 printRules，用于打印游戏规则
    fmt.Println()  // 打印空行
    fmt.Println("I am thinking of a three-digit number.  Try to guess")  // 打印游戏规则说明
    fmt.Println("my number and I will give you clues as follows:")  // 打印游戏规则说明
    fmt.Println("   PICO   - One digit correct but in the wrong position")  // 打印游戏规则说明
    fmt.Println("   FERMI  - One digit correct and in the right position")  // 打印游戏规则说明
    fmt.Println("   BAGELS - No digits correct")  // 打印游戏规则说明
}

func getNumber() []string {  // 定义函数 getNumber，用于生成一个三位数的随机数字
    numbers := []string{"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}  // 创建包含数字字符的切片
    rand.Shuffle(len(numbers), func(i, j int) { numbers[i], numbers[j] = numbers[j], numbers[i] })  // 对数字字符切片进行随机打乱

    return numbers[:3]  // 返回打乱后的前三个数字字符
}

func getValidGuess(guessNumber int) string {  // 定义函数 getValidGuess，用于获取有效的猜测数字
    var guess string  // 声明变量 guess，用于存储猜测数字
    scanner := bufio.NewScanner(os.Stdin)  // 创建用于读取输入的 Scanner 对象
    valid := false  // 声明变量 valid，表示猜测数字是否有效
    for !valid {  // 循环直到猜测数字有效
        fmt.Printf("Guess # %d?\n", guessNumber)  // 打印猜测次数提示
        scanner.Scan()  // 读取用户输入
        guess = strings.TrimSpace(scanner.Text())  // 去除用户输入的首尾空格，并存储到 guess 变量中

        // 确保猜测数字为三位数
        if len(guess) == 3 {
            // 确保猜测数字为数字字符
            _, err := strconv.Atoi(guess)
            if err != nil {
                fmt.Println("What?")  // 打印提示信息
            } else {
                // 确保猜测数字的每个数字字符都不相同
# 检查猜测的数字是否有重复的数字，如果没有则设置valid为true
if (guess[0:1] != guess[1:2]) && (guess[0:1] != guess[2:3]) && (guess[1:2] != guess[2:3]) {
    valid = true
} else {
    # 如果猜测的数字有重复的数字，则输出提示信息
    fmt.Println("Oh, I forgot to tell you that the number I have in mind")
    fmt.Println("has no two digits the same.")
}

# 如果猜测的数字不是三位数，则输出提示信息
} else {
    fmt.Println("Try guessing a three-digit number.")
}

# 返回猜测的数字
return guess
}

# 构建结果字符串
func buildResultString(num []string, guess string) string {
    result := ""

    # 检查数字中正确但位置错误的数字
    for i := 0; i < 2; i++ {
        if num[i] == guess[i+1:i+2] {
            result += "PICO "
        }
        if num[i+1] == guess[i:i+1] {
            result += "PICO "
        }
    }
    if num[0] == guess[2:3] {
        result += "PICO "
    }
    if num[2] == guess[0:1] {
        result += "PICO "
    }

    # 检查数字中正确且位置正确的数字
    for i := 0; i < 3; i++ {
        if num[i] == guess[i:i+1] {
            result += "FERMI "
        }
    }

    # 如果没有任何数字正确，则设置结果为"BAGELS"
    if result == "" {
        result = "BAGELS"
    }

    return result
}

# 主函数
func main() {
    rand.Seed(time.Now().UnixNano())
    scanner := bufio.NewScanner(os.Stdin)

    printWelcome()

    fmt.Println("Would you like the rules (Yes or No)? ")
}
# 扫描输入
scanner.Scan()
# 获取输入的文本
response := scanner.Text()
# 如果输入的文本长度大于0
if len(response) > 0 {
    # 如果输入的文本的第一个字符不是"N"，则打印规则
    if strings.ToUpper(response[0:1]) != "N" {
        printRules()
    }
} else {
    # 如果输入的文本长度为0，则打印规则
    printRules()
}

# 初始化变量gamesWon为0
gamesWon := 0
# 初始化变量stillRunning为true
stillRunning := true

# 当stillRunning为true时执行循环
for stillRunning {
    # 获取一个数字
    num := getNumber()
    # 将数字转换为字符串
    numStr := strings.Join(num, "")
    # 初始化变量guesses为1
    guesses := 1

    # 打印消息
    fmt.Println("\nO.K.  I have a number in mind.")
    # 初始化变量guessing为true
    guessing := true
    # 当guessing为true时执行循环
    for guessing {
        # 获取有效的猜测
        guess := getValidGuess(guesses)

        # 如果猜测等于数字字符串
        if guess == numStr {
            # 打印消息
            fmt.Println("You got it!!")
            # 增加游戏获胜次数
            gamesWon++
            # 将guessing设置为false
            guessing = false
        } else {
            # 打印猜测结果
            fmt.Println(buildResultString(num, guess))
            # 增加猜测次数
            guesses++
            # 如果猜测次数大于最大猜测次数
            if guesses > MAXGUESSES {
                # 打印消息
                fmt.Println("Oh well")
                fmt.Printf("That's %d guesses. My number was %s\n", MAXGUESSES, numStr)
                # 将guessing设置为false
                guessing = false
            }
        }
    }

    # 初始化变量validRespone为false
    validRespone := false
    # 当validRespone为false时执行循环
    for !validRespone {
        # 打印消息
        fmt.Println("Play again (Yes or No)?")
        # 扫描输入
        scanner.Scan()
        # 获取输入的文本
        response := scanner.Text()
# 如果响应的长度大于0
if len(response) > 0 {
    # 设置validRespone为true
    validRespone = true
    # 如果响应的第一个字符不是大写的Y
    if strings.ToUpper(response[0:1]) != "Y" {
        # 设置stillRunning为false
        stillRunning = false
    }
}

# 如果gamesWon大于0
if gamesWon > 0 {
    # 打印获得的分数
    fmt.Printf("\nA %d point Bagels buff!!\n", gamesWon)
}

# 打印结束语
fmt.Println("Hope you had fun.  Bye")
}
```