# `basic-computer-games\00_Alternate_Languages\08_Batnum\go\main.go`

```

package main

import (
	"bufio" // 导入 bufio 包，用于读取输入
	"fmt" // 导入 fmt 包，用于格式化输出
	"os" // 导入 os 包，用于访问操作系统功能
	"strconv" // 导入 strconv 包，用于字符串和基本数据类型之间的转换
	"strings" // 导入 strings 包，用于处理字符串
)

type StartOption int8 // 定义 StartOption 类型为 int8
const (
	StartUndefined StartOption = iota // 定义 StartUndefined 常量为 0
	ComputerFirst // 定义 ComputerFirst 常量为 1
	PlayerFirst // 定义 PlayerFirst 常量为 2
)

type WinOption int8 // 定义 WinOption 类型为 int8
const (
	WinUndefined WinOption = iota // 定义 WinUndefined 常量为 0
	TakeLast // 定义 TakeLast 常量为 1
	AvoidLast // 定义 AvoidLast 常量为 2
)

type GameOptions struct { // 定义 GameOptions 结构体
	pileSize    int // 定义 pileSize 字段为 int 类型
	winOption   WinOption // 定义 winOption 字段为 WinOption 类型
	startOption StartOption // 定义 startOption 字段为 StartOption 类型
	minSelect   int // 定义 minSelect 字段为 int 类型
	maxSelect   int // 定义 maxSelect 字段为 int 类型
}

func NewOptions() *GameOptions { // 定义 NewOptions 函数，返回 GameOptions 指针
	g := GameOptions{} // 创建 GameOptions 结构体实例

	g.pileSize = getPileSize() // 调用 getPileSize 函数获取 pileSize
	if g.pileSize < 0 { // 如果 pileSize 小于 0
		return &g // 返回 GameOptions 实例的指针
	}

	g.winOption = getWinOption() // 调用 getWinOption 函数获取 winOption
	g.minSelect, g.maxSelect = getMinMax() // 调用 getMinMax 函数获取 minSelect 和 maxSelect
	g.startOption = getStartOption() // 调用 getStartOption 函数获取 startOption

	return &g // 返回 GameOptions 实例的指针
}

// 其余函数的作用和功能在代码中已经有详细注释，不再赘述

```