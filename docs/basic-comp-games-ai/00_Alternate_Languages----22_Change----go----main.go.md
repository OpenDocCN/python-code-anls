# `basic-computer-games\00_Alternate_Languages\22_Change\go\main.go`

```py
package main
// 导入所需的包
import (
    "bufio"
    "fmt"
    "math"
    "os"
    "strconv"
)

// 打印欢迎信息
func printWelcome() {
    fmt.Println("                 CHANGE")
    fmt.Println("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
    fmt.Println()
    fmt.Println()
    fmt.Println()
    fmt.Println("I, YOUR FRIENDLY MICROCOMPUTER, WILL DETERMINE")
    fmt.Println("THE CORRECT CHANGE FOR ITEMS COSTING UP TO $100.")
    fmt.Println()
}

// 计算找零
func computeChange(cost, payment float64) {
    // 计算找零金额
    change := int(math.Round((payment - cost) * 100))

    // 如果找零金额为0，打印正确金额信息并返回
    if change == 0 {
        fmt.Println("\nCORRECT AMOUNT, THANK YOU.")
        return
    }

    // 如果找零金额小于0，打印找零不足信息并返回
    if change < 0 {
        fmt.Printf("\nSORRY, YOU HAVE SHORT-CHANGED ME $%0.2f\n", float64(change)/-100.0)
        print()
        return
    }

    // 打印找零金额
    fmt.Printf("\nYOUR CHANGE, $%0.2f:\n", float64(change)/100.0)

    // 计算各面额的钞票/硬币数量并打印
    d := change / 1000
    if d > 0 {
        fmt.Printf("  %d TEN DOLLAR BILL(S)\n", d)
        change -= d * 1000
    }

    d = change / 500
    if d > 0 {
        fmt.Printf("  %d FIVE DOLLAR BILL(S)\n", d)
        change -= d * 500
    }

    d = change / 100
    if d > 0 {
        fmt.Printf("  %d ONE DOLLAR BILL(S)\n", d)
        change -= d * 100
    }

    d = change / 50
    if d > 0 {
        fmt.Println("  1 HALF DOLLAR")
        change -= d * 50
    }

    d = change / 25
    if d > 0 {
        fmt.Printf("  %d QUARTER(S)\n", d)
        change -= d * 25
    }

    d = change / 10
    if d > 0 {
        fmt.Printf("  %d DIME(S)\n", d)
        change -= d * 10
    }

    d = change / 5
    if d > 0 {
        fmt.Printf("  %d NICKEL(S)\n", d)
        change -= d * 5
    }

    if change > 0 {
        fmt.Printf("  %d PENNY(S)\n", change)
    }
}

func main() {
    // 创建用于读取输入的扫描器
    scanner := bufio.NewScanner(os.Stdin)

    // 打印欢迎信息
    printWelcome()

    var cost, payment float64
    var err error
    # 无限循环，直到输入正确的商品价格
    for {
        # 提示用户输入商品价格
        fmt.Println("COST OF ITEM?")
        # 读取用户输入
        scanner.Scan()
        # 将输入的字符串转换为浮点数，存储在 cost 变量中
        cost, err = strconv.ParseFloat(scanner.Text(), 64)
        # 如果转换出错或者输入的价格小于0，则提示用户重新输入
        if err != nil || cost < 0.0 {
            fmt.Println("INVALID INPUT. TRY AGAIN.")
            # 继续下一次循环
            continue
        }
        # 输入正确，跳出循环
        break
    }
    # 无限循环，直到输入正确的付款金额
    for {
        # 提示用户输入付款金额
        fmt.Println("\nAMOUNT OF PAYMENT?")
        # 读取用户输入
        scanner.Scan()
        # 将输入的字符串转换为浮点数，存储在 payment 变量中
        payment, err = strconv.ParseFloat(scanner.Text(), 64)
        # 如果转换出错，则提示用户重新输入
        if err != nil {
            fmt.Println("INVALID INPUT. TRY AGAIN.")
            # 继续下一次循环
            continue
        }
        # 输入正确，跳出循环
        break
    }

    # 调用 computeChange 函数计算找零金额
    computeChange(cost, payment)
    # 打印空行
    fmt.Println()
# 闭合前面的函数定义
```