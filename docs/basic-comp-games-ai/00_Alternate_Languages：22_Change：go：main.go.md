# `00_Alternate_Languages\22_Change\go\main.go`

```
package main  // 声明当前文件所属的包为 main

import (  // 导入需要使用的包
	"bufio"  // 用于读取输入
	"fmt"  // 用于格式化输出
	"math"  // 提供数学函数
	"os"  // 提供对操作系统功能的访问
	"strconv"  // 提供字符串和数字之间的转换
)

func printWelcome() {  // 定义一个名为 printWelcome 的函数
	fmt.Println("                 CHANGE")  // 输出 CHANGE
	fmt.Println("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")  // 输出 CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY
	fmt.Println()  // 输出空行
	fmt.Println()  // 输出空行
	fmt.Println()  // 输出空行
	fmt.Println("I, YOUR FRIENDLY MICROCOMPUTER, WILL DETERMINE")  // 输出 I, YOUR FRIENDLY MICROCOMPUTER, WILL DETERMINE
	fmt.Println("THE CORRECT CHANGE FOR ITEMS COSTING UP TO $100.")  // 输出 THE CORRECT CHANGE FOR ITEMS COSTING UP TO $100.
	fmt.Println()  // 输出空行
}
func computeChange(cost, payment float64) {  // 定义一个名为 computeChange 的函数，接受两个 float64 类型的参数 cost 和 payment
	change := int(math.Round((payment - cost) * 100))  // 计算找零金额，将结果四舍五入为最接近的整数

	if change == 0 {  // 如果找零金额为 0
		fmt.Println("\nCORRECT AMOUNT, THANK YOU.")  // 打印正确的金额，谢谢
		return  // 返回
	}

	if change < 0 {  // 如果找零金额小于 0
		fmt.Printf("\nSORRY, YOU HAVE SHORT-CHANGED ME $%0.2f\n", float64(change)/-100.0)  // 打印抱歉，你找错了钱
		print()  // 打印
		return  // 返回
	}

	fmt.Printf("\nYOUR CHANGE, $%0.2f:\n", float64(change)/100.0)  // 打印你的找零金额

	d := change / 1000  // 计算找零金额中 10 美元纸币的数量
	if d > 0 {  // 如果数量大于 0
		fmt.Printf("  %d TEN DOLLAR BILL(S)\n", d)  // 打印 10 美元纸币的数量
		change -= d * 1000  # 减去尽可能多的1000美元纸币，更新找零金额
	}

	d = change / 500  # 计算500美元纸币的数量
	if d > 0:  # 如果数量大于0
		fmt.Printf("  %d FIVE DOLLAR BILL(S)\n", d)  # 打印500美元纸币的数量
		change -= d * 500  # 减去尽可能多的500美元纸币，更新找零金额

	d = change / 100  # 计算100美元纸币的数量
	if d > 0:  # 如果数量大于0
		fmt.Printf("  %d ONE DOLLAR BILL(S)\n", d)  # 打印100美元纸币的数量
		change -= d * 100  # 减去尽可能多的100美元纸币，更新找零金额

	d = change / 50  # 计算50美分硬币的数量
	if d > 0:  # 如果数量大于0
		fmt.Println("  1 HALF DOLLAR")  # 打印半美元硬币的数量
		change -= d * 50  # 减去尽可能多的50美分硬币，更新找零金额
# 将找零金额除以25，得到25美分硬币的数量
d = change / 25
# 如果25美分硬币的数量大于0，则打印出数量和硬币类型，并更新找零金额
if d > 0:
    print("  %d QUARTER(S)\n" % d)
    change -= d * 25

# 将找零金额除以10，得到10美分硬币的数量
d = change / 10
# 如果10美分硬币的数量大于0，则打印出数量和硬币类型，并更新找零金额
if d > 0:
    print("  %d DIME(S)\n" % d)
    change -= d * 10

# 将找零金额除以5，得到5美分硬币的数量
d = change / 5
# 如果5美分硬币的数量大于0，则打印出数量和硬币类型，并更新找零金额
if d > 0:
    print("  %d NICKEL(S)\n" % d)
    change -= d * 5

# 如果还有剩余的找零金额大于0，则执行下面的代码
if change > 0:
		fmt.Printf("  %d PENNY(S)\n", change)  // 打印找零的数量
	}
}

func main() {
	scanner := bufio.NewScanner(os.Stdin)  // 创建一个从标准输入读取数据的扫描器

	printWelcome()  // 调用打印欢迎信息的函数

	var cost, payment float64  // 声明变量cost和payment为浮点数
	var err error  // 声明变量err为错误类型
	for {
		fmt.Println("COST OF ITEM?")  // 打印提示信息
		scanner.Scan()  // 从标准输入读取数据
		cost, err = strconv.ParseFloat(scanner.Text(), 64)  // 将输入的数据转换为浮点数并赋值给cost，同时检查是否有错误
		if err != nil || cost < 0.0 {  // 如果有错误或者cost小于0
			fmt.Println("INVALID INPUT. TRY AGAIN.")  // 打印错误信息
			continue  // 继续下一次循环
		}
		break  // 结束循环
	}
	// 循环直到用户输入有效的支付金额
	for {
		// 提示用户输入支付金额
		fmt.Println("\nAMOUNT OF PAYMENT?")
		// 读取用户输入
		scanner.Scan()
		// 将用户输入的字符串转换为浮点数
		payment, err = strconv.ParseFloat(scanner.Text(), 64)
		// 如果转换出错，提示用户重新输入
		if err != nil {
			fmt.Println("INVALID INPUT. TRY AGAIN.")
			continue
		}
		// 如果转换成功，跳出循环
		break
	}

	// 计算找零金额
	computeChange(cost, payment)
	// 打印空行
	fmt.Println()
}
```