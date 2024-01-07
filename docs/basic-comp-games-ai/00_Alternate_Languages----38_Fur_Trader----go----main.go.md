# `basic-computer-games\00_Alternate_Languages\38_Fur_Trader\go\main.go`

```

package main

import (
	"bufio" // 导入 bufio 包，用于读取输入
	"fmt" // 导入 fmt 包，用于格式化输出
	"log" // 导入 log 包，用于记录日志
	"math/rand" // 导入 math/rand 包，用于生成随机数
	"os" // 导入 os 包，用于操作系统功能
	"strconv" // 导入 strconv 包，用于字符串和基本数据类型之间的转换
	"strings" // 导入 strings 包，用于处理字符串
	"time" // 导入 time 包，用于时间相关操作
)

const (
	MAXFURS    = 190 // 定义最大毛皮数量
	STARTFUNDS = 600 // 定义初始资金
)

type Fur int8 // 定义毛皮类型

const (
	FUR_MINK Fur = iota // 定义不同种类的毛皮
	FUR_BEAVER
	FUR_ERMINE
	FUR_FOX
)

type Fort int8 // 定义堡垒类型

const (
	FORT_MONTREAL Fort = iota // 定义不同的堡垒
	FORT_QUEBEC
	FORT_NEWYORK
)

type GameState int8 // 定义游戏状态类型

const (
	STARTING GameState = iota // 定义游戏的不同状态
	TRADING
	CHOOSINGFORT
	TRAVELLING
)

func FURS() []string { // 定义函数返回毛皮类型的字符串切片
	return []string{"MINK", "BEAVER", "ERMINE", "FOX"}
}

func FORTS() []string { // 定义函数返回堡垒名称的字符串切片
	return []string{"HOCHELAGA (MONTREAL)", "STADACONA (QUEBEC)", "NEW YORK"}
}

type Player struct { // 定义玩家结构体
	funds float32 // 玩家资金
	furs  []int // 玩家拥有的不同种类毛皮数量
}

func NewPlayer() Player { // 定义创建新玩家的函数
	p := Player{} // 创建新玩家
	p.funds = STARTFUNDS // 设置玩家初始资金
	p.furs = make([]int, 4) // 初始化玩家拥有的不同种类毛皮数量
	return p // 返回新玩家
}

func (p *Player) totalFurs() int { // 定义计算玩家总毛皮数量的方法
	f := 0 // 初始化总毛皮数量
	for _, v := range p.furs { // 遍历玩家拥有的不同种类毛皮数量
		f += v // 累加毛皮数量
	}
	return f // 返回总毛皮数量
}

func (p *Player) lostFurs() { // 定义清空玩家毛皮数量的方法
	for f := 0; f < len(p.furs); f++ { // 遍历玩家拥有的不同种类毛皮数量
		p.furs[f] = 0 // 清空毛皮数量
	}
}

func printTitle() { // 定义打印游戏标题的函数
	fmt.Println("                               FUR TRADER") // 打印游戏标题
	fmt.Println("               CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY") // 打印游戏信息
	fmt.Println() // 打印空行
	fmt.Println() // 打印空行
	fmt.Println() // 打印空行
}

func printIntro() { // 定义打印游戏介绍的函数
	fmt.Println("YOU ARE THE LEADER OF A FRENCH FUR TRADING EXPEDITION IN ") // 打印游戏介绍
	// 其他打印类似
}

func getFortChoice() Fort { // 定义获取玩家选择堡垒的函数
	// 实现获取玩家输入并返回选择的堡垒
}

func printFortComment(f Fort) { // 定义根据选择的堡垒打印评论的函数
	// 根据选择的堡垒打印不同的评论
}

func getYesOrNo() string { // 定义获取玩家输入是或否的函数
	// 实现获取玩家输入并返回是或否
}

func getFursPurchase() []int { // 定义获取玩家购买毛皮数量的函数
	// 实现获取玩家输入并返回购买的毛皮数量
}

```