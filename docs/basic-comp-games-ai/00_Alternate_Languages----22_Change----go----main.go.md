# `basic-computer-games\00_Alternate_Languages\22_Change\go\main.go`

```

package main

import (
	"bufio" // 导入 bufio 包，用于读取输入
	"fmt" // 导入 fmt 包，用于格式化输出
	"math" // 导入 math 包，用于数学计算
	"os" // 导入 os 包，用于操作系统功能
	"strconv" // 导入 strconv 包，用于字符串和基本数据类型之间的转换
)

func printWelcome() {
	// 打印欢迎词
	fmt.Println("                 CHANGE")
	fmt.Println("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
	fmt.Println()
	fmt.Println()
	fmt.Println()
	fmt.Println("I, YOUR FRIENDLY MICROCOMPUTER, WILL DETERMINE")
	fmt.Println("THE CORRECT CHANGE FOR ITEMS COSTING UP TO $100.")
	fmt.Println()
}

func computeChange(cost, payment float64) {
	// 计算找零
	change := int(math.Round((payment - cost) * 100))

	if change == 0 {
		// 如果找零为0，打印正确金额
		fmt.Println("\nCORRECT AMOUNT, THANK YOU.")
		return
	}

	if change < 0 {
		// 如果找零小于0，打印找零不足的信息
		fmt.Printf("\nSORRY, YOU HAVE SHORT-CHANGED ME $%0.2f\n", float64(change)/-100.0)
		print()
		return
	}

	fmt.Printf("\nYOUR CHANGE, $%0.2f:\n", float64(change)/100.0)

	// 计算各种面额的钞票和硬币数量
	d := change / 1000
	if d > 0 {
		fmt.Printf("  %d TEN DOLLAR BILL(S)\n", d)
		change -= d * 1000
	}

	// ... 其他面额的钞票和硬币计算类似

	if change > 0 {
		fmt.Printf("  %d PENNY(S)\n", change)
	}
}

func main() {
	scanner := bufio.NewScanner(os.Stdin)

	printWelcome()

	var cost, payment float64
	var err error
	for {
		fmt.Println("COST OF ITEM?")
		scanner.Scan()
		cost, err = strconv.ParseFloat(scanner.Text(), 64)
		if err != nil || cost < 0.0 {
			fmt.Println("INVALID INPUT. TRY AGAIN.")
			continue
		}
		break
	}
	for {
		fmt.Println("\nAMOUNT OF PAYMENT?")
		scanner.Scan()
		payment, err = strconv.ParseFloat(scanner.Text(), 64)
		if err != nil {
			fmt.Println("INVALID INPUT. TRY AGAIN.")
			continue
		}
		break
	}

	computeChange(cost, payment)
	fmt.Println()
}

```