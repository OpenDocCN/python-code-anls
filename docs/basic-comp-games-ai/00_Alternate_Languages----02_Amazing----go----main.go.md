# `basic-computer-games\00_Alternate_Languages\02_Amazing\go\main.go`

```

package main

import (
	"bufio" // 导入 bufio 包，用于读取输入
	"fmt" // 导入 fmt 包，用于格式化输出
	"log" // 导入 log 包，用于记录错误日志
	"math/rand" // 导入 math/rand 包，用于生成随机数
	"os" // 导入 os 包，用于操作文件和目录
	"strconv" // 导入 strconv 包，用于字符串和数字之间的转换
	"time" // 导入 time 包，用于处理时间
)

func main() {
	rand.Seed(time.Now().UnixNano()) // 设置随机数种子为当前时间的纳秒数

	printWelcome() // 调用打印欢迎信息的函数

	h, w := getDimensions() // 调用获取迷宫高度和宽度的函数，并将返回值赋给变量 h 和 w
	m := NewMaze(h, w) // 创建一个新的迷宫对象
	m.draw() // 绘制迷宫
}

type direction int64 // 定义一个表示方向的类型

const (
	LEFT direction = iota // 定义四个方向常量
	UP
	RIGHT
	DOWN
)

const (
	EXIT_DOWN  = 1 // 定义迷宫出口向下的标识
	EXIT_RIGHT = 2 // 定义迷宫出口向右的标识
)

type maze struct { // 定义迷宫结构体
	width    int // 迷宫宽度
	length   int // 迷宫长度
	used     [][]int // 记录迷宫中已经使用的位置
	walls    [][]int // 记录迷宫中的墙
	enterCol int // 迷宫入口所在列
}

func NewMaze(w, l int) maze { // 创建新迷宫的函数
	if (w < 2) || (l < 2) { // 如果宽度或长度小于2，则输出错误信息并退出程序
		log.Fatal("invalid dimensions supplied")
	}

	m := maze{width: w, length: l} // 创建迷宫对象并初始化宽度和长度

	m.used = make([][]int, l) // 初始化 used 切片
	for i := range m.used {
		m.used[i] = make([]int, w)
	}

	m.walls = make([][]int, l) // 初始化 walls 切片
	for i := range m.walls {
		m.walls[i] = make([]int, w)
	}

	m.enterCol = rand.Intn(w) // 随机确定迷宫入口所在列

	m.build() // 构建迷宫
	col := rand.Intn(m.width - 1) // 随机确定迷宫出口所在列
	row := m.length - 1 // 出口所在行为迷宫长度减一
	m.walls[row][col] = m.walls[row][col] + 1 // 在出口位置设置出口标识

	return m // 返回创建的迷宫对象
}

func (m *maze) build() { // 构建迷宫的方法
	// 省略部分代码
}

func (m *maze) getPossibleDirections(row, col int) []direction { // 获取可能的方向的方法
	// 省略部分代码
}

func (m *maze) makeOpening(dirs []direction, row, col, count int) (int, int, int) { // 创建迷宫开口的方法
	// 省略部分代码
}

func (m *maze) draw() { // 绘制迷宫的方法
	// 省略部分代码
}

func printWelcome() { // 打印欢迎信息的函数
	// 省略部分代码
}

func getDimensions() (int, int) { // 获取迷宫高度和宽度的函数
	// 省略部分代码
}

```