# `00_Alternate_Languages\01_Acey_Ducey\go\main.go`

```
package main  # 导入 main 包

import (  # 导入所需的包
	"bufio"  # 用于读取输入
	"fmt"  # 用于格式化输出
	"math/rand"  # 用于生成随机数
	"os"  # 用于操作文件和目录
	"sort"  # 用于对切片进行排序
	"strconv"  # 用于字符串和数字之间的转换
	"strings"  # 用于处理字符串
	"time"  # 用于处理时间
)

var welcome = `  # 定义字符串变量 welcome
Acey-Ducey is played in the following manner
The dealer (computer) deals two cards face up
You have an option to bet or not bet depending
on whether or not you feel the card will have
a value between the first two.
If you do not want to bet, input a 0
func main() {
	// 设置随机数种子
	rand.Seed(time.Now().UnixNano())
	// 创建一个从标准输入读取数据的扫描器
	scanner := bufio.NewScanner(os.Stdin)

	// 打印欢迎消息
	fmt.Println(welcome)

	// 循环进行游戏
	for {
		// 进行游戏，初始分数为100
		play(100)
		// 询问是否再次进行游戏
		fmt.Println("TRY AGAIN (YES OR NO)")
		// 读取用户输入的响应
		scanner.Scan()
		response := scanner.Text()
		// 如果响应不是"YES"（不区分大小写），则跳出循环
		if strings.ToUpper(response) != "YES" {
			break
		}
	}

	// 打印结束游戏的消息
	fmt.Println("O.K., HOPE YOU HAD FUN!")
}
func play(money int) {  // 定义一个名为 play 的函数，接受一个整数参数 money
	scanner := bufio.NewScanner(os.Stdin)  // 创建一个用于读取标准输入的 Scanner 对象
	var bet int  // 声明一个整数变量 bet

	for {  // 进入无限循环
		// Shuffle the cards  // 洗牌
		cards := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}  // 创建一个整数切片 cards，包含 1 到 14 的数字
		rand.Shuffle(len(cards), func(i, j int) { cards[i], cards[j] = cards[j], cards[i] })  // 使用随机函数对 cards 切片进行洗牌

		// Take the first two for the dealer and sort  // 取出前两张牌给庄家并排序
		dealerCards := cards[0:2]  // 创建一个名为 dealerCards 的切片，包含 cards 切片的前两个元素
		sort.Ints(dealerCards)  // 对 dealerCards 切片进行排序

		fmt.Printf("YOU NOW HAVE %d DOLLARS.\n\n", money)  // 打印当前玩家的金额
		fmt.Printf("HERE ARE YOUR NEXT TWO CARDS:\n%s\n%s", getCardName(dealerCards[0]), getCardName(dealerCards[1]))  // 打印玩家的下两张牌
		fmt.Printf("\n\n")  // 打印空行

		//Check if Bet is Valid  // 检查下注是否有效
		for {  // 进入内部无限循环
			// 打印提示信息，要求用户输入赌注
			fmt.Println("WHAT IS YOUR BET:")
			// 从标准输入中读取用户输入的赌注，并转换为整数
			scanner.Scan()
			b, err := strconv.Atoi(scanner.Text())
			// 如果转换出错，打印提示信息并继续循环
			if err != nil {
				fmt.Println("PLEASE ENTER A POSITIVE NUMBER")
				continue
			}
			// 将用户输入的赌注赋值给变量bet
			bet = b

			// 如果赌注为0，打印提示信息并跳转到标签there处
			if bet == 0 {
				fmt.Printf("CHICKEN!\n\n")
				goto there
			}

			// 如果赌注大于0且不超过玩家拥有的金额，跳出循环
			if (bet > 0) && (bet <= money) {
				break
			}
		}

		// 绘制玩家的牌
		// 打印玩家的第三张牌的名称
		fmt.Printf("YOUR CARD: %s\n", getCardName(cards[2]))
		// 如果玩家的第三张牌大于庄家的第一张牌并且小于庄家的第二张牌，则玩家赢得本局游戏
		if (cards[2] > dealerCards[0]) && (cards[2] < dealerCards[1]) {
			fmt.Println("YOU WIN!!!")
			// 玩家赢得赌注金额
			money = money + bet
		} else {
			// 否则玩家输掉赌注金额
			fmt.Println("SORRY, YOU LOSE")
			money = money - bet
		}
		// 打印空行
		fmt.Println()

		// 如果玩家的金额小于等于0，则游戏结束
		if money <= 0 {
			fmt.Printf("%s\n", "SORRY, FRIEND, BUT YOU BLEW YOUR WAD.")
			return
		}
		// 标签，用于跳转到游戏继续进行的地方
	there:
	}
}

// 根据牌的数字返回牌的名称
func getCardName(c int) string {
	switch c {
    case 11:  # 如果输入的值为11
        return "JACK"  # 返回字符串"JACK"
    case 12:  # 如果输入的值为12
        return "QUEEN"  # 返回字符串"QUEEN"
    case 13:  # 如果输入的值为13
        return "KING"  # 返回字符串"KING"
    case 14:  # 如果输入的值为14
        return "ACE"  # 返回字符串"ACE"
    default:  # 如果输入的值不是11、12、13、14中的任何一个
        return strconv.Itoa(c)  # 返回输入值的字符串形式
```