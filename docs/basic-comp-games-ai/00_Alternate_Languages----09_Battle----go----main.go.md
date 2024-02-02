# `basic-computer-games\00_Alternate_Languages\09_Battle\go\main.go`

```py
package main

import (
    "bufio"  // 导入 bufio 包，提供读取输入的功能
    "fmt"    // 导入 fmt 包，提供格式化输入输出的功能
    "math/rand"  // 导入 math/rand 包，提供生成随机数的功能
    "os"     // 导入 os 包，提供操作系统功能
    "strconv"    // 导入 strconv 包，提供字符串转换功能
    "strings"    // 导入 strings 包，提供字符串处理功能
    "time"   // 导入 time 包，提供时间相关功能
)

const (
    SEA_WIDTH        = 6  // 定义海域宽度为 6
    DESTROYER_LENGTH = 2  // 定义驱逐舰长度为 2
    CRUISER_LENGTH   = 3  // 定义巡洋舰长度为 3
    CARRIER_LENGTH   = 4  // 定义航空母舰长度为 4
)

type Point [2]int  // 定义 Point 类型为包含两个整数的数组
type Vector Point   // 定义 Vector 类型为 Point 类型

type Sea [][]int  // 定义 Sea 类型为整数的二维数组

func NewSea() Sea {
    s := make(Sea, 6)  // 创建一个包含 6 个元素的 Sea 类型的数组
    for r := 0; r < SEA_WIDTH; r++ {  // 遍历海域的宽度
        c := make([]int, 6)  // 创建一个包含 6 个整数的数组
        s[r] = c  // 将数组 c 赋值给海域数组的第 r 行
    }
    return s  // 返回创建的海域数组
}

func getRandomVector() Vector {
    v := Vector{}  // 创建一个空的 Vector 类型
    for {  // 无限循环
        v[0] = rand.Intn(3) - 1  // 生成 -1 到 1 之间的随机整数，并赋值给 v 的第一个元素
        v[1] = rand.Intn(3) - 1  // 生成 -1 到 1 之间的随机整数，并赋值给 v 的第二个元素
        if !(v[0] == 0 && v[1] == 0) {  // 如果 v 不是 (0, 0)
            break  // 跳出循环
        }
    }
    return v  // 返回生成的随机向量
}

func addVector(p Point, v Vector) Point {
    newPoint := Point{}  // 创建一个空的 Point 类型
    newPoint[0] = p[0] + v[0]  // 计算新点的横坐标
    newPoint[1] = p[1] + v[1]  // 计算新点的纵坐标
    return newPoint  // 返回新点
}

func isWithinSea(p Point, s Sea) bool {
    return (1 <= p[0] && p[0] <= len(s)) && (1 <= p[1] && p[1] <= len(s))  // 判断点是否在海域范围内
}

func valueAt(p Point, s Sea) int {
    return s[p[1]-1][p[0]-1]  // 返回海域中指定点的值
}

func reportInputError() {
    fmt.Printf("INVALID. SPECIFY TWO NUMBERS FROM 1 TO %d, SEPARATED BY A COMMA.\n", SEA_WIDTH)  // 输出输入错误提示信息
}

func getNextTarget(s Sea) Point {
    scanner := bufio.NewScanner(os.Stdin)  // 创建一个从标准输入读取数据的 Scanner 对象
    for {  // 无限循环
        fmt.Println("\n?")  // 输出提示信息
        scanner.Scan()  // 读取输入
        vals := strings.Split(scanner.Text(), ",")  // 将输入按逗号分割成字符串数组
        if len(vals) != 2 {  // 如果输入不是两个数字
            reportInputError()  // 输出输入错误提示信息
            continue  // 继续下一次循环
        }
        x, xErr := strconv.Atoi(strings.TrimSpace(vals[0]))  // 将第一个字符串转换为整数
# 将字符串数组中的第二个元素转换为整数，并去除首尾空格
x20x20x20x20x20x20x20x20y, yErr := strconv.Atoi(strings.TrimSpace(vals[1]))

# 如果字符串数组长度不为2，或者 xErr 或 yErr 不为空，则执行以下操作
x20x20x20x20x20x20x20x20if (len(vals) != 2) || (xErr != nil) || (yErr != nil) {
x20x20x20x20x20x20x20x20x20x20x20x20报告输入错误
x20x20x20x20x20x20x20x20x20x20x20x20继续
x20x20x20x20x20x20x20x20}

# 创建一个 Point 结构体
x20x20x20x20x20x20x20x20p := Point{}
x20x20x20x20x20x20x20x20p[0] = x
x20x20x20x20x20x20x20x20p[1] = y

# 如果点 p 在海域 s 内，则执行以下操作
x20x20x20x20x20x20x20x20if isWithinSea(p, s) {
x20x20x20x20x20x20x20x20x20x20x20x20返回 p
x20x20x20x20x20x20x20x20}
x20x20x20x20}
}

# 设置指定位置的值
func setValueAt(value int, p Point, s Sea) {
x20x20x20x20s[p[1]-1][p[0]-1] = value
}

# 检查海域 s 中是否存在指定的船只
func hasShip(s Sea, code int) bool {
x20x20x20x20hasShip := false
x20x20x20x20for r := 0; r < SEA_WIDTH; r++ {
x20x20x20x20x20x20x
# 如果位置 p 不在海域 s 内部
        if !isWithinSea(p, s) {
            # 将清除位置标记设置为 false
            clearPosition = false
            # 跳出循环
            break
        }
        # 如果位置 p 的值大于 0
        if valueAt(p, s) > 0 {
            # 将清除位置标记设置为 false
            clearPosition = false
            # 跳出循环
            break
        }
    }
    # 如果清除位置标记为 false
    if !clearPosition {
        # 继续下一次循环
        continue
    }
    # 遍历点集合中的每个位置 p
    for _, p := range points {
        # 在海域 s 中设置位置 p 的值
        setValueAt(code, p, s)
    }
    # 跳出循环
    break
}

# 设置船只的初始位置
func setupShips(s Sea) {
    # 在海域 s 中放置长度为 DESTROYER_LENGTH 的船只
    placeShip(s, DESTROYER_LENGTH, 1)
    # 在海域 s 中放置长度为 DESTROYER_LENGTH 的船只
    placeShip(s, DESTROYER_LENGTH, 2)
    # 在海域 s 中放置长度为 CRUISER_LENGTH 的船只
    placeShip(s, CRUISER_LENGTH, 3)
    # 在海域 s 中放置长度为 CRUISER_LENGTH 的船只
    placeShip(s, CRUISER_LENGTH, 4)
    # 在海域 s 中放置长度为 CARRIER_LENGTH 的船只
    placeShip(s, CARRIER_LENGTH, 5)
    # 在海域 s 中放置长度为 CARRIER_LENGTH 的船只
    placeShip(s, CARRIER_LENGTH, 6)
}

# 打印游戏介绍
func printIntro() {
    fmt.Println("                BATTLE")
    fmt.Println("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
    fmt.Println()
    fmt.Println("THE FOLLOWING CODE OF THE BAD GUYS' FLEET DISPOSITION")
    fmt.Println("HAS BEEN CAPTURED BUT NOT DECODED: ")
    fmt.Println()
}

# 打印游戏指令
func printInstructions() {
    fmt.Println()
    fmt.Println()
    fmt.Println("DE-CODE IT AND USE IT IF YOU CAN")
    fmt.Println("BUT KEEP THE DE-CODING METHOD A SECRET.")
    fmt.Println()
    fmt.Println("START GAME")
}

# 打印加密后的海域
func printEncodedSea(s Sea) {
    # 遍历海域 s 的每一行
    for x := 0; x < SEA_WIDTH; x++ {
        fmt.Println()
        # 遍历海域 s 的每一列
        for y := SEA_WIDTH - 1; y > -1; y-- {
            # 打印海域 s 中位置 (x, y) 的值
            fmt.Printf(" %d", s[y][x])
        }
    }
    fmt.Println()
}
func wipeout(s Sea) bool {
    // 检查海域是否完全清空，即没有剩余的船只
    for c := 1; c <= 7; c++ {
        // 如果海域中还有船只存在，则返回 false
        if hasShip(s, c) {
            return false
        }
    }
    // 如果海域中没有船只存在，则返回 true
    return true
}

func main() {
    // 使用当前时间的纳秒数作为随机数种子
    rand.Seed(time.Now().UnixNano())

    // 创建一个新的海域对象
    s := NewSea()

    // 在海域中设置船只的位置
    setupShips(s)

    // 打印游戏介绍
    printIntro()

    // 打印加密后的海域信息
    printEncodedSea(s)

    // 打印游戏指令
    printInstructions()

    // 初始化击中和未击中的次数
    splashes := 0
    hits := 0

    // 无限循环，直到游戏结束
    for {
        // 获取下一个目标位置
        target := getNextTarget(s)
        // 获取目标位置的值
        targetValue := valueAt(target, s)

        // 如果目标位置的值小于 0，表示已经击中过该位置的船只
        if targetValue < 0 {
            fmt.Printf("YOU ALREADY PUT A HOLE IN SHIP NUMBER %d AT THAT POINT.\n", targetValue)
        }

        // 如果目标位置的值小于等于 0，表示未击中船只
        if targetValue <= 0 {
            fmt.Println("SPLASH! TRY AGAIN.")
            splashes += 1
            continue
        }

        // 如果目标位置的值大于 0，表示击中船只
        fmt.Printf("A DIRECT HIT ON SHIP NUMBER %d\n", targetValue)
        hits += 1
        // 在目标位置设置船只值的相反数，表示击中
        setValueAt(targetValue*-1, target, s)

        // 如果目标位置的船只已经被全部击沉
        if !hasShip(s, targetValue) {
            fmt.Println("AND YOU SUNK IT. HURRAH FOR THE GOOD GUYS.")
            fmt.Println("SO FAR, THE BAD GUYS HAVE LOST")
            fmt.Printf("%d DESTROYER(S), %d CRUISER(S), AND %d AIRCRAFT CARRIER(S).\n", countSunk(s, []int{1, 2}), countSunk(s, []int{3, 4}), countSunk(s, []int{5, 6}))
        }

        // 如果海域中还有未击沉的船只
        if !wipeout(s) {
            fmt.Printf("YOUR CURRENT SPLASH/HIT RATIO IS %2f\n", float32(splashes)/float32(hits))
            continue
        }
    }
}
# 使用格式化输出打印击中率的信息
fmt.Printf("YOU HAVE TOTALLY WIPED OUT THE BAD GUYS' FLEET WITH A FINAL SPLASH/HIT RATIO OF %2f\n", float32(splashes)/float32(hits))

# 如果未击中任何目标，则打印祝贺信息
if splashes == 0:
    fmt.Println("CONGRATULATIONS -- A DIRECT HIT EVERY TIME.")

# 打印分隔线
fmt.Println("\n****************************")

# 退出循环
break
```