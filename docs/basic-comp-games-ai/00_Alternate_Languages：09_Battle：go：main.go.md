# `00_Alternate_Languages\09_Battle\go\main.go`

```
package main  // 声明当前文件所属的包

import (
	"bufio"  // 导入 bufio 包，用于提供带缓冲的 I/O
	"fmt"  // 导入 fmt 包，用于格式化 I/O
	"math/rand"  // 导入 math/rand 包，用于生成随机数
	"os"  // 导入 os 包，提供了对操作系统功能的访问
	"strconv"  // 导入 strconv 包，用于字符串和基本数据类型之间的转换
	"strings"  // 导入 strings 包，提供了对字符串的操作函数
	"time"  // 导入 time 包，提供了时间的显示和测量的函数
)

const (
	SEA_WIDTH        = 6  // 定义常量 SEA_WIDTH，表示海域的宽度为 6
	DESTROYER_LENGTH = 2  // 定义常量 DESTROYER_LENGTH，表示驱逐舰的长度为 2
	CRUISER_LENGTH   = 3  // 定义常量 CRUISER_LENGTH，表示巡洋舰的长度为 3
	CARRIER_LENGTH   = 4  // 定义常量 CARRIER_LENGTH，表示航空母舰的长度为 4
)

type Point [2]int  // 定义类型 Point，表示二维坐标
type Vector Point  # 定义了一个名为Vector的自定义类型，其类型为Point

type Sea [][]int  # 定义了一个名为Sea的自定义类型，其类型为二维int数组

func NewSea() Sea {  # 定义了一个名为NewSea的函数，返回类型为Sea
	s := make(Sea, 6)  # 创建一个长度为6的Sea类型的切片
	for r := 0; r < SEA_WIDTH; r++ {  # 循环遍历SEA_WIDTH次
		c := make([]int, 6)  # 创建一个长度为6的int类型的切片
		s[r] = c  # 将切片c赋值给s的第r个元素
	}

	return s  # 返回s
}

func getRandomVector() Vector {  # 定义了一个名为getRandomVector的函数，返回类型为Vector
	v := Vector{}  # 创建一个空的Vector类型变量v

	for {  # 无限循环
		v[0] = rand.Intn(3) - 1  # 给v的第一个元素赋值为0到2之间的随机数减1
		v[1] = rand.Intn(3) - 1  # 给v的第二个元素赋值为0到2之间的随机数减1
		# 如果向量 v 的第一个和第二个元素都不等于 0，则跳出循环
		if !(v[0] == 0 && v[1] == 0) {
			break
		}
	}
	# 返回向量 v
	return v
}

# 添加向量 v 到点 p，返回新的点
func addVector(p Point, v Vector) Point {
	# 创建一个新的点 newPoint
	newPoint := Point{}

	# 将新点的第一个元素设置为点 p 的第一个元素加上向量 v 的第一个元素
	newPoint[0] = p[0] + v[0]
	# 将新点的第二个元素设置为点 p 的第二个元素加上向量 v 的第二个元素
	newPoint[1] = p[1] + v[1]

	# 返回新的点 newPoint
	return newPoint
}

# 判断点 p 是否在海域 s 内，返回布尔值
func isWithinSea(p Point, s Sea) bool {
	# 返回点 p 的第一个元素是否在 1 到海域 s 的长度之间，并且点 p 的第二个元素是否在 1 到海域 s 的长度之间
	return (1 <= p[0] && p[0] <= len(s)) && (1 <= p[1] && p[1] <= len(s))
}
func valueAt(p Point, s Sea) int {
	// 返回海域中指定坐标的值
	return s[p[1]-1][p[0]-1]
}

func reportInputError() {
	// 打印输入错误的提示信息
	fmt.Printf("INVALID. SPECIFY TWO NUMBERS FROM 1 TO %d, SEPARATED BY A COMMA.\n", SEA_WIDTH)
}

func getNextTarget(s Sea) Point {
	// 创建一个从标准输入读取数据的扫描器
	scanner := bufio.NewScanner(os.Stdin)

	// 循环直到输入正确的坐标
	for {
		// 打印提示信息
		fmt.Println("\n?")
		// 读取输入
		scanner.Scan()

		// 将输入按逗号分隔成两个字符串
		vals := strings.Split(scanner.Text(), ",")

		// 如果输入不是两个值，打印错误信息并继续循环
		if len(vals) != 2 {
			reportInputError()
			continue
		} // 结束函数

		// 将字符串转换为整数，并去除首尾空格
		x, xErr := strconv.Atoi(strings.TrimSpace(vals[0]))
		y, yErr := strconv.Atoi(strings.TrimSpace(vals[1]))

		// 检查输入是否符合要求，如果不符合则报告输入错误并继续循环
		if (len(vals) != 2) || (xErr != nil) || (yErr != nil) {
			reportInputError()
			continue
		}

		// 创建一个 Point 结构体
		p := Point{}
		// 设置 Point 结构体的值
		p[0] = x
		p[1] = y
		// 如果点在海洋范围内，则返回该点
		if isWithinSea(p, s) {
			return p
		}
	}
}

// 设置海洋中指定位置的值
func setValueAt(value int, p Point, s Sea) {
	s[p[1]-1][p[0]-1] = value
}
# 将给定坐标位置上的海域中的值设置为指定的值

func hasShip(s Sea, code int) bool:
# 检查海域中是否存在指定的船只代码，返回布尔值

	hasShip := false
	for r := 0; r < SEA_WIDTH; r++:
		for c := 0; c < SEA_WIDTH; c++:
			if s[r][c] == code:
				hasShip = true
				break
# 遍历海域中的每个位置，如果找到指定的船只代码，则将 hasShip 设置为 true，并且跳出循环

	return hasShip
# 返回是否存在指定船只代码的布尔值

func countSunk(s Sea, codes []int) int:
# 计算海域中沉没的船只数量，接受海域和船只代码数组作为参数

	sunk := 0
# 初始化沉没船只数量为 0

	for _, c := range codes:
# 遍历船只代码数组中的每个代码

		if !hasShip(s, c) {  # 如果海域中没有船只
			sunk += 1  # 则表示击沉的船只数量加一
		}
	}

	return sunk  # 返回击沉的船只数量
}

func placeShip(s Sea, size, code int) {  # 在海域中放置船只
	for {  # 无限循环，直到成功放置船只
		start := Point{}  # 初始化船只的起始点
		start[0] = rand.Intn(SEA_WIDTH) + 1  # 随机生成起始点的横坐标
		start[1] = rand.Intn(SEA_WIDTH) + 1  # 随机生成起始点的纵坐标
		vector := getRandomVector()  # 获取随机方向向量

		point := start  # 将当前点设置为起始点
		points := []Point{}  # 初始化船只的点集合

		for i := 0; i < size; i++ {  # 循环船只的大小次数
			point = addVector(point, vector)  # 根据方向向量移动当前点
			points = append(points, point)  # 将 point 添加到 points 切片中

		clearPosition := true  # 初始化 clearPosition 为 true
		for _, p := range points:  # 遍历 points 切片中的每个点
			if !isWithinSea(p, s):  # 如果点不在海洋范围内
				clearPosition = false  # 将 clearPosition 设置为 false
				break  # 跳出循环
			if valueAt(p, s) > 0:  # 如果点的值大于 0
				clearPosition = false  # 将 clearPosition 设置为 false
				break  # 跳出循环
		if !clearPosition:  # 如果 clearPosition 为 false
			continue  # 继续下一次循环

		for _, p := range points:  # 遍历 points 切片中的每个点
			setValueAt(code, p, s)  # 在 code 中设置点的值为 s
		}
		break  # 结束当前循环
	}
}

func setupShips(s Sea) {
	placeShip(s, DESTROYER_LENGTH, 1)  # 在海域中放置长度为 DESTROYER_LENGTH 的船只
	placeShip(s, DESTROYER_LENGTH, 2)  # 在海域中放置长度为 DESTROYER_LENGTH 的船只
	placeShip(s, CRUISER_LENGTH, 3)  # 在海域中放置长度为 CRUISER_LENGTH 的船只
	placeShip(s, CRUISER_LENGTH, 4)  # 在海域中放置长度为 CRUISER_LENGTH 的船只
	placeShip(s, CARRIER_LENGTH, 5)  # 在海域中放置长度为 CARRIER_LENGTH 的船只
	placeShip(s, CARRIER_LENGTH, 6)  # 在海域中放置长度为 CARRIER_LENGTH 的船只
}

func printIntro() {
	fmt.Println("                BATTLE")  # 打印标题
	fmt.Println("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")  # 打印创意计算机的位置
	fmt.Println()  # 打印空行
	fmt.Println("THE FOLLOWING CODE OF THE BAD GUYS' FLEET DISPOSITION")  # 打印敌方舰队编码
	fmt.Println("HAS BEEN CAPTURED BUT NOT DECODED:	")  # 打印敌方舰队编码未被解码
# 打印空行
fmt.Println()

# 打印游戏指令
func printInstructions() {
    fmt.Println()
    fmt.Println()
    fmt.Println("DE-CODE IT AND USE IT IF YOU CAN")
    fmt.Println("BUT KEEP THE DE-CODING METHOD A SECRET.")
    fmt.Println()
    fmt.Println("START GAME")
}

# 打印加密的海域
func printEncodedSea(s Sea) {
    for x := 0; x < SEA_WIDTH; x++:
        # 打印换行
        fmt.Println()
        for y := SEA_WIDTH - 1; y > -1; y--:
            # 打印海域中的数字
            fmt.Printf(" %d", s[y][x])
    # 打印换行
    fmt.Println()
}
}

# wipeout 函数用于检查海域中是否还有船只存在，如果没有则返回 true，否则返回 false
func wipeout(s Sea) bool:
    for c := 1; c <= 7; c++:
        # 调用 hasShip 函数检查海域中是否存在船只，如果存在则返回 false
        if hasShip(s, c):
            return false
    # 如果循环结束仍未返回 false，则说明海域中没有船只存在，返回 true
    return true

# main 函数是程序的入口点
func main():
    # 使用当前时间的纳秒数作为随机数种子
    rand.Seed(time.Now().UnixNano())

    # 创建一个新的海域对象
    s := NewSea()

    # 在海域中设置船只的位置
    setupShips(s)

    # 打印游戏介绍信息
    printIntro()
    # 打印加密的海域
    printEncodedSea(s)
    
    # 打印游戏指令
    printInstructions()
    
    # 初始化变量，记录击中和未击中的次数
    splashes := 0
    hits := 0
    
    # 进入游戏循环
    for {
        # 获取下一个目标位置
        target := getNextTarget(s)
        # 获取目标位置的值
        targetValue := valueAt(target, s)
        
        # 如果目标位置的值小于0，表示已经在该位置放置了一个船的部分
        if targetValue < 0:
            fmt.Printf("YOU ALREADY PUT A HOLE IN SHIP NUMBER %d AT THAT POINT.\n", targetValue)
        
        # 如果目标位置的值小于等于0，表示未击中船只
        if targetValue <= 0:
            fmt.Println("SPLASH! TRY AGAIN.")
            splashes += 1
            continue
    }
		// 打印击中目标船只的信息
		fmt.Printf("A DIRECT HIT ON SHIP NUMBER %d\n", targetValue)
		// 增加击中次数
		hits += 1
		// 在目标位置设置为击中状态
		setValueAt(targetValue*-1, target, s)

		// 如果目标船只已经被击沉
		if !hasShip(s, targetValue) {
			// 打印击沉目标船只的信息
			fmt.Println("AND YOU SUNK IT. HURRAH FOR THE GOOD GUYS.")
			fmt.Println("SO FAR, THE BAD GUYS HAVE LOST")
			// 打印击沉不同类型船只的数量
			fmt.Printf("%d DESTROYER(S), %d CRUISER(S), AND %d AIRCRAFT CARRIER(S).\n", countSunk(s, []int{1, 2}), countSunk(s, []int{3, 4}), countSunk(s, []int{5, 6}))
		}

		// 如果还有未击沉的船只
		if !wipeout(s) {
			// 打印当前的击中与未击中比率
			fmt.Printf("YOUR CURRENT SPLASH/HIT RATIO IS %2f\n", float32(splashes)/float32(hits))
			// 继续下一轮攻击
			continue
		}

		// 打印最终的击中与未击中比率
		fmt.Printf("YOU HAVE TOTALLY WIPED OUT THE BAD GUYS' FLEET WITH A FINAL SPLASH/HIT RATIO OF %2f\n", float32(splashes)/float32(hits))

		// 如果一次未击中
		if splashes == 0 {
			// 打印祝贺信息
			fmt.Println("CONGRATULATIONS -- A DIRECT HIT EVERY TIME.")
		}  # 结束当前的 for 循环

		fmt.Println("\n****************************")  # 打印分隔线
		break  # 跳出当前的 for 循环
	}
```