# `basic-computer-games\00_Alternate_Languages\09_Battle\go\main.go`

```

package main

import (
	"bufio" // 导入 bufio 包，用于读取输入
	"fmt" // 导入 fmt 包，用于格式化输出
	"math/rand" // 导入 math/rand 包，用于生成随机数
	"os" // 导入 os 包，用于访问操作系统功能
	"strconv" // 导入 strconv 包，用于字符串和数字之间的转换
	"strings" // 导入 strings 包，用于处理字符串
	"time" // 导入 time 包，用于处理时间
)

const (
	SEA_WIDTH        = 6 // 定义海域宽度为 6
	DESTROYER_LENGTH = 2 // 定义驱逐舰长度为 2
	CRUISER_LENGTH   = 3 // 定义巡洋舰长度为 3
	CARRIER_LENGTH   = 4 // 定义航空母舰长度为 4
)

type Point [2]int // 定义 Point 类型为包含两个整数的数组
type Vector Point // 定义 Vector 类型为 Point

type Sea [][]int // 定义 Sea 类型为二维整数数组

func NewSea() Sea {
	// 创建一个新的海域，初始化为 6x6 的二维数组
	s := make(Sea, 6)
	for r := 0; r < SEA_WIDTH; r++ {
		c := make([]int, 6)
		s[r] = c
	}

	return s
}

func getRandomVector() Vector {
	// 生成一个随机的方向向量
	v := Vector{}

	for {
		v[0] = rand.Intn(3) - 1
		v[1] = rand.Intn(3) - 1

		if !(v[0] == 0 && v[1] == 0) {
			break
		}
	}
	return v
}

func addVector(p Point, v Vector) Point {
	// 将向量 v 加到点 p 上，返回新的点
	newPoint := Point{}

	newPoint[0] = p[0] + v[0]
	newPoint[1] = p[1] + v[1]

	return newPoint
}

func isWithinSea(p Point, s Sea) bool {
	// 判断点 p 是否在海域 s 内
	return (1 <= p[0] && p[0] <= len(s)) && (1 <= p[1] && p[1] <= len(s))
}

func valueAt(p Point, s Sea) int {
	// 返回海域 s 中点 p 的值
	return s[p[1]-1][p[0]-1]
}

func reportInputError() {
	// 报告输入错误
	fmt.Printf("INVALID. SPECIFY TWO NUMBERS FROM 1 TO %d, SEPARATED BY A COMMA.\n", SEA_WIDTH)
}

func getNextTarget(s Sea) Point {
	// 获取下一个目标点
	scanner := bufio.NewScanner(os.Stdin)

	for {
		fmt.Println("\n?")
		scanner.Scan()

		vals := strings.Split(scanner.Text(), ",")

		if len(vals) != 2 {
			reportInputError()
			continue
		}

		x, xErr := strconv.Atoi(strings.TrimSpace(vals[0]))
		y, yErr := strconv.Atoi(strings.TrimSpace(vals[1]))

		if (len(vals) != 2) || (xErr != nil) || (yErr != nil) {
			reportInputError()
			continue
		}

		p := Point{}
		p[0] = x
		p[1] = y
		if isWithinSea(p, s) {
			return p
		}
	}
}

func setValueAt(value int, p Point, s Sea) {
	// 设置海域 s 中点 p 的值为 value
	s[p[1]-1][p[0]-1] = value
}

func hasShip(s Sea, code int) bool {
	// 判断海域 s 中是否存在指定编号的船只
	hasShip := false
	for r := 0; r < SEA_WIDTH; r++ {
		for c := 0; c < SEA_WIDTH; c++ {
			if s[r][c] == code {
				hasShip = true
				break
			}
		}
	}
	return hasShip
}

func countSunk(s Sea, codes []int) int {
	// 统计海域 s 中已沉没的船只数量
	sunk := 0

	for _, c := range codes {
		if !hasShip(s, c) {
			sunk += 1
		}
	}

	return sunk
}

func placeShip(s Sea, size, code int) {
	// 在海域 s 中放置指定长度和编号的船只
	for {
		start := Point{}
		start[0] = rand.Intn(SEA_WIDTH) + 1
		start[1] = rand.Intn(SEA_WIDTH) + 1
		vector := getRandomVector()

		point := start
		points := []Point{}

		for i := 0; i < size; i++ {
			point = addVector(point, vector)
			points = append(points, point)
		}

		clearPosition := true
		for _, p := range points {
			if !isWithinSea(p, s) {
				clearPosition = false
				break
			}
			if valueAt(p, s) > 0 {
				clearPosition = false
				break
			}
		}
		if !clearPosition {
			continue
		}

		for _, p := range points {
			setValueAt(code, p, s)
		}
		break
	}
}

func setupShips(s Sea) {
	// 设置海域 s 中的船只
	placeShip(s, DESTROYER_LENGTH, 1)
	placeShip(s, DESTROYER_LENGTH, 2)
	placeShip(s, CRUISER_LENGTH, 3)
	placeShip(s, CRUISER_LENGTH, 4)
	placeShip(s, CARRIER_LENGTH, 5)
	placeShip(s, CARRIER_LENGTH, 6)
}

func printIntro() {
	// 打印游戏介绍
	fmt.Println("                BATTLE")
	fmt.Println("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
	fmt.Println()
	fmt.Println("THE FOLLOWING CODE OF THE BAD GUYS' FLEET DISPOSITION")
	fmt.Println("HAS BEEN CAPTURED BUT NOT DECODED:	")
	fmt.Println()
}

func printInstructions() {
	// 打印游戏指令
	fmt.Println()
	fmt.Println()
	fmt.Println("DE-CODE IT AND USE IT IF YOU CAN")
	fmt.Println("BUT KEEP THE DE-CODING METHOD A SECRET.")
	fmt.Println()
	fmt.Println("START GAME")
}

func printEncodedSea(s Sea) {
	// 打印加密的海域信息
	for x := 0; x < SEA_WIDTH; x++ {
		fmt.Println()
		for y := SEA_WIDTH - 1; y > -1; y-- {
			fmt.Printf(" %d", s[y][x])
		}
	}
	fmt.Println()
}

func wipeout(s Sea) bool {
	// 判断海域 s 中的船只是否全部被击沉
	for c := 1; c <= 7; c++ {
		if hasShip(s, c) {
			return false
		}
	}
	return true
}

func main() {
	// 主函数
	rand.Seed(time.Now().UnixNano()) // 设置随机数种子

	s := NewSea() // 创建新的海域

	setupShips(s) // 设置海域中的船只

	printIntro() // 打印游戏介绍

	printEncodedSea(s) // 打印加密的海域信息

	printInstructions() // 打印游戏指令

	splashes := 0 // 初始化击中水面的次数
	hits := 0 // 初始化击中船只的次数

	for {
		target := getNextTarget(s) // 获取下一个目标点
		targetValue := valueAt(target, s) // 获取目标点的值

		if targetValue < 0 {
			fmt.Printf("YOU ALREADY PUT A HOLE IN SHIP NUMBER %d AT THAT POINT.\n", targetValue)
		}

		if targetValue <= 0 {
			fmt.Println("SPLASH! TRY AGAIN.")
			splashes += 1
			continue
		}

		fmt.Printf("A DIRECT HIT ON SHIP NUMBER %d\n", targetValue)
		hits += 1
		setValueAt(targetValue*-1, target, s)

		if !hasShip(s, targetValue) {
			fmt.Println("AND YOU SUNK IT. HURRAH FOR THE GOOD GUYS.")
			fmt.Println("SO FAR, THE BAD GUYS HAVE LOST")
			fmt.Printf("%d DESTROYER(S), %d CRUISER(S), AND %d AIRCRAFT CARRIER(S).\n", countSunk(s, []int{1, 2}), countSunk(s, []int{3, 4}), countSunk(s, []int{5, 6}))
		}

		if !wipeout(s) {
			fmt.Printf("YOUR CURRENT SPLASH/HIT RATIO IS %2f\n", float32(splashes)/float32(hits))
			continue
		}

		fmt.Printf("YOU HAVE TOTALLY WIPED OUT THE BAD GUYS' FLEET WITH A FINAL SPLASH/HIT RATIO OF %2f\n", float32(splashes)/float32(hits))

		if splashes == 0 {
			fmt.Println("CONGRATULATIONS -- A DIRECT HIT EVERY TIME.")
		}

		fmt.Println("\n****************************")
		break
	}
}

```