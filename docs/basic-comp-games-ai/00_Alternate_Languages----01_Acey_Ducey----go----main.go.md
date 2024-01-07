# `basic-computer-games\00_Alternate_Languages\01_Acey_Ducey\go\main.go`

```

package main

import (
	"bufio" // 导入用于读取输入的包
	"fmt" // 导入用于格式化输出的包
	"math/rand" // 导入用于生成随机数的包
	"os" // 导入用于操作文件和目录的包
	"sort" // 导入用于排序的包
	"strconv" // 导入用于字符串和数字转换的包
	"strings" // 导入用于处理字符串的包
	"time" // 导入用于处理时间的包
)

var welcome = `
Acey-Ducey is played in the following manner
The dealer (computer) deals two cards face up
You have an option to bet or not bet depending
on whether or not you feel the card will have
a value between the first two.
If you do not want to bet, input a 0
  ` // 定义欢迎信息

func main() {
	rand.Seed(time.Now().UnixNano()) // 设置随机数种子
	scanner := bufio.NewScanner(os.Stdin) // 创建用于读取输入的扫描器

	fmt.Println(welcome) // 输出欢迎信息

	for {
		play(100) // 进行游戏
		fmt.Println("TRY AGAIN (YES OR NO)") // 提示是否再玩一次
		scanner.Scan() // 读取输入
		response := scanner.Text() // 获取输入的文本
		if strings.ToUpper(response) != "YES" { // 判断是否继续游戏
			break
		}
	}

	fmt.Println("O.K., HOPE YOU HAD FUN!") // 输出结束信息
}

func play(money int) {
	scanner := bufio.NewScanner(os.Stdin) // 创建用于读取输入的扫描器
	var bet int // 定义赌注变量

	for {
		// 洗牌
		cards := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14} // 创建牌组
		rand.Shuffle(len(cards), func(i, j int) { cards[i], cards[j] = cards[j], cards[i] }) // 洗牌

		// 发两张牌给庄家并排序
		dealerCards := cards[0:2] // 获取庄家的前两张牌
		sort.Ints(dealerCards) // 对庄家的牌进行排序

		fmt.Printf("YOU NOW HAVE %d DOLLARS.\n\n", money) // 输出玩家当前的金额
		fmt.Printf("HERE ARE YOUR NEXT TWO CARDS:\n%s\n%s", getCardName(dealerCards[0]), getCardName(dealerCards[1])) // 输出玩家的两张牌
		fmt.Printf("\n\n")

		// 检查赌注是否有效
		for {
			fmt.Println("WHAT IS YOUR BET:") // 提示输入赌注
			scanner.Scan() // 读取输入
			b, err := strconv.Atoi(scanner.Text()) // 将输入的文本转换为整数
			if err != nil {
				fmt.Println("PLEASE ENTER A POSITIVE NUMBER") // 提示输入正数
				continue
			}
			bet = b // 设置赌注

			if bet == 0 {
				fmt.Printf("CHICKEN!\n\n") // 输出放弃赌注的信息
				goto there // 跳转到标签 there
			}

			if (bet > 0) && (bet <= money) { // 判断赌注是否在有效范围内
				break
			}
		}

		// 抽取玩家的牌
		fmt.Printf("YOUR CARD: %s\n", getCardName(cards[2])) // 输出玩家的第三张牌
		if (cards[2] > dealerCards[0]) && (cards[2] < dealerCards[1]) { // 判断玩家是否赢得比赛
			fmt.Println("YOU WIN!!!") // 输出赢得比赛的信息
			money = money + bet // 更新玩家的金额
		} else {
			fmt.Println("SORRY, YOU LOSE") // 输出输掉比赛的信息
			money = money - bet // 更新玩家的金额
		}
		fmt.Println()

		if money <= 0 { // 判断玩家的金额是否小于等于0
			fmt.Printf("%s\n", "SORRY, FRIEND, BUT YOU BLEW YOUR WAD.") // 输出玩家破产的信息
			return
		}
	there:
	}
}

func getCardName(c int) string {
	switch c {
	case 11:
		return "JACK" // 返回JACK
	case 12:
		return "QUEEN" // 返回QUEEN
	case 13:
		return "KING" // 返回KING
	case 14:
		return "ACE" // 返回ACE
	default:
		return strconv.Itoa(c) // 返回数字对应的字符串
	}
}

```