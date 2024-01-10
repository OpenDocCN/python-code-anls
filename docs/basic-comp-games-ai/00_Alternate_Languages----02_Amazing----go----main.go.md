# `basic-computer-games\00_Alternate_Languages\02_Amazing\go\main.go`

```
package main

import (
    "bufio"  // 导入 bufio 包，提供读写数据的缓冲区
    "fmt"    // 导入 fmt 包，提供格式化 I/O 的函数
    "log"    // 导入 log 包，提供简单的日志服务
    "math/rand"  // 导入 math/rand 包，提供伪随机数生成器
    "os"     // 导入 os 包，提供操作系统函数
    "strconv"    // 导入 strconv 包，提供字符串与基本数据类型之间的转换
    "time"   // 导入 time 包，提供时间的显示和测量用的函数
)

func main() {
    rand.Seed(time.Now().UnixNano())  // 使用当前时间的纳秒部分作为随机数种子

    printWelcome()  // 调用 printWelcome 函数

    h, w := getDimensions()  // 调用 getDimensions 函数，获取迷宫的高度和宽度
    m := NewMaze(h, w)  // 调用 NewMaze 函数，创建一个新的迷宫对象
    m.draw()  // 调用迷宫对象的 draw 方法，绘制迷宫
}

type direction int64  // 定义一个名为 direction 的自定义类型

const (
    LEFT direction = iota  // 定义 LEFT 常量，并初始化为 0
    UP  // 定义 UP 常量
    RIGHT  // 定义 RIGHT 常量
    DOWN  // 定义 DOWN 常量
)

const (
    EXIT_DOWN  = 1  // 定义 EXIT_DOWN 常量，并初始化为 1
    EXIT_RIGHT = 2  // 定义 EXIT_RIGHT 常量，并初始化为 2
)

type maze struct {  // 定义迷宫结构体
    width    int  // 迷宫的宽度
    length   int  // 迷宫的长度
    used     [][]int  // 二维数组，表示迷宫中的位置是否被使用
    walls    [][]int  // 二维数组，表示迷宫中的墙
    enterCol int  // 迷宫的入口列
}

func NewMaze(w, l int) maze {  // 定义 NewMaze 函数，用于创建一个新的迷宫对象
    if (w < 2) || (l < 2) {  // 如果宽度或长度小于 2，则输出错误信息并退出程序
        log.Fatal("invalid dimensions supplied")
    }

    m := maze{width: w, length: l}  // 创建一个迷宫对象，并初始化宽度和长度

    m.used = make([][]int, l)  // 初始化 used 二维数组
    for i := range m.used {
        m.used[i] = make([]int, w)
    }

    m.walls = make([][]int, l)  // 初始化 walls 二维数组
    for i := range m.walls {
        m.walls[i] = make([]int, w)
    }

    m.enterCol = rand.Intn(w)  // 随机确定迷宫的入口列

    m.build()  // 调用迷宫对象的 build 方法，确定墙的布局

    col := rand.Intn(m.width - 1)  // 随机确定迷宫的出口列
    row := m.length - 1
    m.walls[row][col] = m.walls[row][col] + 1  // 在出口位置添加出口标记

    return m  // 返回创建的迷宫对象
}

func (m *maze) build() {  // 定义迷宫对象的 build 方法
    row := 0  // 初始化行号
    col := 0  // 初始化列号
    count := 2  // 初始化计数器

    for {  // 进入无限循环
        possibleDirs := m.getPossibleDirections(row, col)  // 获取当前位置的可能移动方向

        if len(possibleDirs) != 0 {  // 如果存在可能的移动方向
            row, col, count = m.makeOpening(possibleDirs, row, col, count)  // 在可能的方向中选择一个方向并打开墙
        } else {  // 如果不存在可能的移动方向
            for {  // 进入无限循环
                if col != m.width-1 {  // 如果当前列不是最右侧列
# 如果条件成立，将 col 加 1
                    col = col + 1
                # 否则如果 row 不等于 m.length-1
                } else if row != m.length-1 {
                    # 将 row 加 1
                    row = row + 1
                    # 将 col 设为 0
                    col = 0
                # 否则
                } else {
                    # 将 row 设为 0
                    row = 0
                    # 将 col 设为 0
                    col = 0
                }
                # 如果 m.used[row][col] 不等于 0
                if m.used[row][col] != 0 {
                    # 跳出循环
                    break
                }
            }
        }
        # 如果 count 等于 (m.width*m.length)+1
        if count == (m.width*m.length)+1 {
            # 跳出循环
            break
        }
    }
}

# 获取可能的方向
func (m *maze) getPossibleDirections(row, col int) []direction {
    # 创建一个包含四个方向的 map
    possible_dirs := make(map[direction]bool, 4)
    possible_dirs[LEFT] = true
    possible_dirs[UP] = true
    possible_dirs[RIGHT] = true
    possible_dirs[DOWN] = true

    # 如果 col 等于 0 或者 m.used[row][col-1] 不等于 0
    if (col == 0) || (m.used[row][col-1] != 0) {
        # 将 LEFT 设为 false
        possible_dirs[LEFT] = false
    }
    # 如果 row 等于 0 或者 m.used[row-1][col] 不等于 0
    if (row == 0) || (m.used[row-1][col] != 0) {
        # 将 UP 设为 false
        possible_dirs[UP] = false
    }
    # 如果 col 等于 m.width-1 或者 m.used[row][col+1] 不等于 0
    if (col == m.width-1) || (m.used[row][col+1] != 0) {
        # 将 RIGHT 设为 false
        possible_dirs[RIGHT] = false
    }
    # 如果 row 等于 m.length-1 或者 m.used[row+1][col] 不等于 0
    if (row == m.length-1) || (m.used[row+1][col] != 0) {
        # 将 DOWN 设为 false
        possible_dirs[DOWN] = false
    }

    # 创建一个空的方向数组
    ret := make([]direction, 0)
    # 遍历可能的方向
    for d, v := range possible_dirs {
        # 如果方向可行
        if v {
            # 将方向添加到数组中
            ret = append(ret, d)
        }
    }
    # 返回结果数组
    return ret
}
// 创建迷宫的开口，接受方向数组、行、列和计数作为参数，返回更新后的行、列和计数
func (m *maze) makeOpening(dirs []direction, row, col, count int) (int, int, int) {
    // 从方向数组中随机选择一个方向
    dir := rand.Intn(len(dirs))

    // 根据选择的方向更新行和列，并设置对应的墙壁状态
    if dirs[dir] == LEFT {
        col = col - 1
        m.walls[row][col] = int(EXIT_RIGHT)
    } else if dirs[dir] == UP {
        row = row - 1
        m.walls[row][col] = int(EXIT_DOWN)
    } else if dirs[dir] == RIGHT {
        m.walls[row][col] = m.walls[row][col] + EXIT_RIGHT
        col = col + 1
    } else if dirs[dir] == DOWN {
        m.walls[row][col] = m.walls[row][col] + EXIT_DOWN
        row = row + 1
    }

    // 标记当前位置已经使用过，并更新计数
    m.used[row][col] = count
    count = count + 1
    return row, col, count
}

// 绘制迷宫
func (m *maze) draw() {
    // 绘制迷宫的顶部边界
    for col := 0; col < m.width; col++ {
        if col == m.enterCol {
            fmt.Print(".  ")
        } else {
            fmt.Print(".--")
        }
    }
    fmt.Println(".")

    // 绘制迷宫的内部结构
    for row := 0; row < m.length; row++ {
        fmt.Print("|")
        for col := 0; col < m.width; col++ {
            // 根据墙壁状态绘制不同的内部结构
            if m.walls[row][col] < 2 {
                fmt.Print("  |")
            } else {
                fmt.Print("   ")
            }
        }
        fmt.Println()
        // 绘制迷宫的横向墙壁
        for col := 0; col < m.width; col++ {
            if (m.walls[row][col] == 0) || (m.walls[row][col] == 2) {
                fmt.Print(":--")
            } else {
                fmt.Print(":  ")
            }
        }
    }
}
func printWelcome() {
    // 打印欢迎信息
    fmt.Println("                            AMAZING PROGRAM")
    fmt.Print("               CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")
}

func getDimensions() (int, int) {
    // 创建一个从标准输入读取数据的扫描器
    scanner := bufio.NewScanner(os.Stdin)

    // 提示用户输入宽度（大于1）
    fmt.Println("Enter a width ( > 1 ):")
    // 扫描用户输入的宽度
    scanner.Scan()
    // 将用户输入的宽度转换为整数
    w, err := strconv.Atoi(scanner.Text())
    // 如果转换出错，则输出错误信息并退出程序
    if err != nil {
        log.Fatal("invalid dimension")
    }

    // 提示用户输入高度（大于1）
    fmt.Println("Enter a height ( > 1 ):")
    // 扫描用户输入的高度
    scanner.Scan()
    // 将用户输入的高度转换为整数
    h, err := strconv.Atoi(scanner.Text())
    // 如果转换出错，则输出错误信息并退出程序
    if err != nil {
        log.Fatal("invalid dimension")
    }

    // 返回用户输入的宽度和高度
    return w, h
}
```