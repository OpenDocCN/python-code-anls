# `00_Alternate_Languages\02_Amazing\go\main.go`

```
package main  // 声明当前文件所属的包为 main

import (
	"bufio"  // 导入 bufio 包，用于提供缓冲 I/O
	"fmt"  // 导入 fmt 包，用于格式化输入输出
	"log"  // 导入 log 包，用于记录日志
	"math/rand"  // 导入 math/rand 包，用于生成随机数
	"os"  // 导入 os 包，用于提供操作系统功能
	"strconv"  // 导入 strconv 包，用于字符串和基本数据类型之间的转换
	"time"  // 导入 time 包，用于处理时间
)

func main() {  // 主函数
	rand.Seed(time.Now().UnixNano())  // 使用当前时间的纳秒数作为随机数种子

	printWelcome()  // 调用 printWelcome 函数，打印欢迎信息

	h, w := getDimensions()  // 调用 getDimensions 函数，获取迷宫的高度和宽度，并赋值给 h 和 w

	m := NewMaze(h, w)  // 调用 NewMaze 函数，创建一个新的迷宫对象，赋值给 m
	m.draw()  // 调用迷宫对象的 draw 方法，绘制迷宫
}
}

type direction int64  # 定义一个新的类型 direction，用于表示方向

const (
	LEFT direction = iota  # 定义常量 LEFT，值为 0
	UP  # 定义常量 UP，值为 1
	RIGHT  # 定义常量 RIGHT，值为 2
	DOWN  # 定义常量 DOWN，值为 3
)

const (
	EXIT_DOWN  = 1  # 定义常量 EXIT_DOWN，值为 1
	EXIT_RIGHT = 2  # 定义常量 EXIT_RIGHT，值为 2
)

type maze struct {  # 定义结构体 maze
	width    int  # 定义结构体字段 width，表示迷宫的宽度
	length   int  # 定义结构体字段 length，表示迷宫的长度
	used     [][]int  # 定义结构体字段 used，表示迷宫中已使用的位置
	walls    [][]int  // 二维切片，用于表示迷宫中的墙壁
	enterCol int     // 表示迷宫的入口列号
}

func NewMaze(w, l int) maze {
	if (w < 2) || (l < 2) {
		log.Fatal("invalid dimensions supplied")  // 如果提供的迷宫宽度或长度小于2，则输出错误信息并终止程序
	}

	m := maze{width: w, length: l}  // 创建迷宫对象并初始化宽度和长度

	m.used = make([][]int, l)  // 初始化迷宫的使用情况切片
	for i := range m.used {
		m.used[i] = make([]int, w)
	}

	m.walls = make([][]int, l)  // 初始化迷宫的墙壁切片
	for i := range m.walls {
		m.walls[i] = make([]int, w)
	}
	// 随机确定入口列
	m.enterCol = rand.Intn(w)

	// 确定墙壁的布局
	m.build()

	// 添加一个出口
	col := rand.Intn(m.width - 1)
	row := m.length - 1
	m.walls[row][col] = m.walls[row][col] + 1

	return m
}

func (m *maze) build() {
	row := 0
	col := 0
	count := 2
``` 

在这段代码中，注释解释了每个语句的作用。例如，`m.enterCol = rand.Intn(w)` 语句用于随机确定迷宫的入口列，`m.build()` 语句用于确定墙壁的布局，`col := rand.Intn(m.width - 1)` 和 `row := m.length - 1` 语句用于添加一个出口。
	for {
		# 获取当前位置可行的方向
		possibleDirs := m.getPossibleDirections(row, col)

		# 如果存在可行的方向
		if len(possibleDirs) != 0 {
			# 根据可行的方向创建开口，并更新行、列和计数
			row, col, count = m.makeOpening(possibleDirs, row, col, count)
		} else {
			# 如果不存在可行的方向
			for {
				# 如果列不是迷宫的宽度减一
				if col != m.width-1 {
					col = col + 1
				} else if row != m.length-1 {
					# 如果行不是迷宫的长度减一
					row = row + 1
					col = 0
				} else {
					# 否则，重置行和列为0
					row = 0
					col = 0
				}

				# 如果当前位置已经被使用过
				if m.used[row][col] != 0 {
					# 退出循环
					break
				}
		}
	}

	// 如果计数等于迷宫的宽度乘以长度加一，则跳出循环
	if count == (m.width*m.length)+1 {
		break
	}
}

// 获取给定位置的可能方向
func (m *maze) getPossibleDirections(row, col int) []direction {
	// 创建一个包含四个方向的可能方向的映射
	possible_dirs := make(map[direction]bool, 4)
	possible_dirs[LEFT] = true
	possible_dirs[UP] = true
	possible_dirs[RIGHT] = true
	possible_dirs[DOWN] = true

	// 如果当前位置在最左边或者左边的位置已经被使用，则将左方向设为不可行
	if (col == 0) || (m.used[row][col-1] != 0) {
		possible_dirs[LEFT] = false
	}
	if (row == 0) || (m.used[row-1][col] != 0) {
		// 如果当前位置在迷宫的最顶部，或者上方的位置已经被占用，则将向上的方向标记为不可行
		possible_dirs[UP] = false
	}
	if (col == m.width-1) || (m.used[row][col+1] != 0) {
		// 如果当前位置在迷宫的最右侧，或者右侧的位置已经被占用，则将向右的方向标记为不可行
		possible_dirs[RIGHT] = false
	}
	if (row == m.length-1) || (m.used[row+1][col] != 0) {
		// 如果当前位置在迷宫的最底部，或者下方的位置已经被占用，则将向下的方向标记为不可行
		possible_dirs[DOWN] = false
	}

	ret := make([]direction, 0)
	for d, v := range possible_dirs {
		if v {
			ret = append(ret, d)
		}
	}
	return ret
}

func (m *maze) makeOpening(dirs []direction, row, col, count int) (int, int, int) {
# 从 dirs 数组中随机选择一个索引
dir := rand.Intn(len(dirs))

# 如果选中的方向是 LEFT，则将列数减一，并在当前位置的墙上标记出口朝右
if dirs[dir] == LEFT {
    col = col - 1
    m.walls[row][col] = int(EXIT_RIGHT)
} 
# 如果选中的方向是 UP，则将行数减一，并在当前位置的墙上标记出口朝下
else if dirs[dir] == UP {
    row = row - 1
    m.walls[row][col] = int(EXIT_DOWN)
} 
# 如果选中的方向是 RIGHT，则在当前位置的墙上标记出口朝右，并将列数加一
else if dirs[dir] == RIGHT {
    m.walls[row][col] = m.walls[row][col] + EXIT_RIGHT
    col = col + 1
} 
# 如果选中的方向是 DOWN，则在当前位置的墙上标记出口朝下，并将行数加一
else if dirs[dir] == DOWN {
    m.walls[row][col] = m.walls[row][col] + EXIT_DOWN
    row = row + 1
}

# 在迷宫的 used 数组中标记当前位置已经使用过
m.used[row][col] = count
# 更新计数器
count = count + 1
# 返回更新后的行数、列数和计数器值
return row, col, count
// draw the maze
func (m *maze) draw() {
	// 遍历迷宫的列
	for col := 0; col < m.width; col++ {
		// 如果当前列是入口列，则打印".  "，否则打印".--"
		if col == m.enterCol {
			fmt.Print(".  ")
		} else {
			fmt.Print(".--")
		}
	}
	// 换行
	fmt.Println(".")

	// 遍历迷宫的行
	for row := 0; row < m.length; row++ {
		// 打印"|"
		fmt.Print("|")
		// 遍历迷宫的列
		for col := 0; col < m.width; col++ {
			// 如果当前位置的墙小于2，则打印"  |"，否则打印"   "
			if m.walls[row][col] < 2 {
				fmt.Print("  |")
			} else {
				fmt.Print("   ")
			}
		}
		// 换行
		fmt.Println("|")
	}
}
		}  # 结束内层循环
		fmt.Println()  # 打印空行
		for col := 0; col < m.width; col++ {  # 遍历列
			if (m.walls[row][col] == 0) || (m.walls[row][col] == 2) {  # 如果当前位置是0或2
				fmt.Print(":--")  # 打印":--"
			} else {
				fmt.Print(":  ")  # 否则打印":  "
			}
		}
		fmt.Println(".")  # 打印"."
	}  # 结束外层循环
}

func printWelcome() {  # 定义打印欢迎信息的函数
	fmt.Println("                            AMAZING PROGRAM")  # 打印欢迎信息
	fmt.Print("               CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")  # 打印创意计算的信息
}

func getDimensions() (int, int) {  # 定义获取尺寸的函数，返回两个整数
	scanner := bufio.NewScanner(os.Stdin)  # 创建一个从标准输入读取数据的扫描器
	// 打印提示信息，要求用户输入宽度（大于1）
	fmt.Println("Enter a width ( > 1 ):")
	// 从标准输入中读取用户输入的宽度
	scanner.Scan()
	// 将用户输入的宽度转换为整数类型
	w, err := strconv.Atoi(scanner.Text())
	// 如果转换过程中出现错误，打印错误信息并退出程序
	if err != nil {
		log.Fatal("invalid dimension")
	}

	// 打印提示信息，要求用户输入高度（大于1）
	fmt.Println("Enter a height ( > 1 ):")
	// 从标准输入中读取用户输入的高度
	scanner.Scan()
	// 将用户输入的高度转换为整数类型
	h, err := strconv.Atoi(scanner.Text())
	// 如果转换过程中出现错误，打印错误信息并退出程序
	if err != nil {
		log.Fatal("invalid dimension")
	}

	// 返回用户输入的宽度和高度
	return w, h
}
```