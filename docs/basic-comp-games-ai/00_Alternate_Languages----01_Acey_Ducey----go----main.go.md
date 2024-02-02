# `basic-computer-games\00_Alternate_Languages\01_Acey_Ducey\go\main.go`

```py
package main

import (
    "bufio"  // 导入 bufio 包，提供读取输入的功能
    "fmt"    // 导入 fmt 包，提供格式化输入输出的功能
    "math/rand"  // 导入 math/rand 包，提供生成随机数的功能
    "os"     // 导入 os 包，提供操作系统功能
    "sort"   // 导入 sort 包，提供排序功能
    "strconv"    // 导入 strconv 包，提供字符串转换功能
    "strings"    // 导入 strings 包，提供字符串处理功能
    "time"   // 导入 time 包，提供时间相关功能
)

var welcome = `
Acey-Ducey is played in the following manner
The dealer (computer) deals two cards face up
You have an option to bet or not bet depending
on whether or not you feel the card will have
a value between the first two.
If you do not want to bet, input a 0
  `  // 定义欢迎信息

func main() {
    rand.Seed(time.Now().UnixNano())  // 使用当前时间作为随机数种子
    scanner := bufio.NewScanner(os.Stdin)  // 创建一个从标准输入读取数据的扫描器

    fmt.Println(welcome)  // 打印欢迎信息

    for {  // 无限循环
        play(100)  // 调用 play 函数，传入初始金额
        fmt.Println("TRY AGAIN (YES OR NO)")  // 打印提示信息
        scanner.Scan()  // 读取用户输入
        response := scanner.Text()  // 获取用户输入的文本
        if strings.ToUpper(response) != "YES" {  // 判断用户输入是否为 YES
            break  // 如果不是 YES，则跳出循环
        }
    }

    fmt.Println("O.K., HOPE YOU HAD FUN!")  // 打印结束信息
}

func play(money int) {
    scanner := bufio.NewScanner(os.Stdin)  // 创建一个从标准输入读取数据的扫描器
    var bet int  // 定义赌注变量

    for {  // 无限循环
        // Shuffle the cards
        cards := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}  // 创建一个包含 14 张牌的切片
        rand.Shuffle(len(cards), func(i, j int) { cards[i], cards[j] = cards[j], cards[i] })  // 洗牌

        // Take the first two for the dealer and sort
        dealerCards := cards[0:2]  // 获取前两张牌
        sort.Ints(dealerCards)  // 对牌进行排序

        fmt.Printf("YOU NOW HAVE %d DOLLARS.\n\n", money)  // 打印当前金额
        fmt.Printf("HERE ARE YOUR NEXT TWO CARDS:\n%s\n%s", getCardName(dealerCards[0]), getCardName(dealerCards[1]))  // 打印玩家的两张牌
        fmt.Printf("\n\n")  // 打印空行

        //Check if Bet is Valid
        for {  // 无限循环
            fmt.Println("WHAT IS YOUR BET:")  // 打印提示信息
            scanner.Scan()  // 读取用户输入
// 将文本转换为整数，并将结果赋值给变量b，同时检查是否有错误发生
b, err := strconv.Atoi(scanner.Text())

// 如果有错误发生，则打印提示信息并继续循环
if err != nil {
    fmt.Println("PLEASE ENTER A POSITIVE NUMBER")
    continue
}

// 将变量b的值赋给变量bet
bet = b

// 如果bet的值为0，则打印提示信息并跳转到标签there
if bet == 0 {
    fmt.Printf("CHICKEN!\n\n")
    goto there
}

// 如果bet的值大于0且不超过money的值，则跳出循环
if (bet > 0) && (bet <= money) {
    break
}

// 打印玩家的卡片信息
fmt.Printf("YOUR CARD: %s\n", getCardName(cards[2]))

// 如果玩家的卡片大于庄家的第一张卡片且小于庄家的第二张卡片，则玩家获胜，否则玩家输掉赌注
if (cards[2] > dealerCards[0]) && (cards[2] < dealerCards[1]) {
    fmt.Println("YOU WIN!!!")
    money = money + bet
} else {
    fmt.Println("SORRY, YOU LOSE")
    money = money - bet
}

// 打印空行
fmt.Println()

// 如果money小于等于0，则打印提示信息并返回
if money <= 0 {
    fmt.Printf("%s\n", "SORRY, FRIEND, BUT YOU BLEW YOUR WAD.")
    return
}

// 标签there
there:
}

// 根据卡片的值返回对应的名称
func getCardName(c int) string {
    switch c {
    case 11:
        return "JACK"
    case 12:
        return "QUEEN"
    case 13:
        return "KING"
    case 14:
        return "ACE"
    default:
        return strconv.Itoa(c)
    }
}
```