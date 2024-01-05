# `00_Alternate_Languages\11_Bombardment\go\main.go`

```
package main  // 声明当前文件属于 main 包

import (
	"bufio"  // 导入 bufio 包，用于提供带缓冲的 I/O
	"fmt"  // 导入 fmt 包，用于格式化输入输出
	"math/rand"  // 导入 math/rand 包，用于生成随机数
	"os"  // 导入 os 包，提供操作系统函数
	"strconv"  // 导入 strconv 包，用于字符串和基本数据类型之间的转换
	"strings"  // 导入 strings 包，提供对字符串的操作
	"time"  // 导入 time 包，提供时间的功能
)

// Messages correspond to outposts remaining (3, 2, 1, 0)
var PLAYER_PROGRESS_MESSAGES = []string{  // 声明一个字符串切片变量 PLAYER_PROGRESS_MESSAGES
	"YOU GOT ME, I'M GOING FAST. BUT I'LL GET YOU WHEN\nMY TRANSISTO&S RECUP%RA*E!",  // 第一个元素
	"THREE DOWN, ONE TO GO.\n\n",  // 第二个元素
	"TWO DOWN, TWO TO GO.\n\n",  // 第三个元素
	"ONE DOWN, THREE TO GO.\n\n",  // 第四个元素
}
var ENEMY_PROGRESS_MESSAGES = []string{  // 定义一个字符串数组，用于存储敌人进度的消息
	"YOU'RE DEAD. YOUR LAST OUTPOST WAS AT %d. HA, HA, HA.\nBETTER LUCK NEXT TIME.",  // 第一个消息，包含一个占位符
	"YOU HAVE ONLY ONE OUTPOST LEFT.\n\n",  // 第二个消息
	"YOU HAVE ONLY TWO OUTPOSTS LEFT.\n\n",  // 第三个消息
	"YOU HAVE ONLY THREE OUTPOSTS LEFT.\n\n",  // 第四个消息
}

func displayField() {  // 定义一个名为 displayField 的函数
	for r := 0; r < 5; r++ {  // 循环5次，表示行数
		initial := r*5 + 1  // 计算每行的初始值
		for c := 0; c < 5; c++ {  // 循环5次，表示列数
			//x := strconv.Itoa(initial + c)  // 将初始值和列数相加并转换为字符串
			fmt.Printf("\t%d", initial+c)  // 打印每个格子的编号
		}
		fmt.Println()  // 换行
	}
	fmt.Print("\n\n\n\n\n\n\n\n\n")  // 打印多个空行
}

func printIntro() {  // 定义一个名为 printIntro 的函数
# 打印游戏标题和说明
fmt.Println("                                BOMBARDMENT")
fmt.Println("                CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
fmt.Println()
fmt.Println()
fmt.Println("YOU ARE ON A BATTLEFIELD WITH 4 PLATOONS AND YOU")
fmt.Println("HAVE 25 OUTPOSTS AVAILABLE WHERE THEY MAY BE PLACED.")
fmt.Println("YOU CAN ONLY PLACE ONE PLATOON AT ANY ONE OUTPOST.")
fmt.Println("THE COMPUTER DOES THE SAME WITH ITS FOUR PLATOONS.")
fmt.Println()
fmt.Println("THE OBJECT OF THE GAME IS TO FIRE MISSLES AT THE")
fmt.Println("OUTPOSTS OF THE COMPUTER.  IT WILL DO THE SAME TO YOU.")
fmt.Println("THE ONE WHO DESTROYS ALL FOUR OF THE ENEMY'S PLATOONS")
fmt.Println("FIRST IS THE WINNER.")
fmt.Println()
fmt.Println("GOOD LUCK... AND TELL US WHERE YOU WANT THE BODIES SENT!")
fmt.Println()
fmt.Println("TEAR OFF MATRIX AND USE IT TO CHECK OFF THE NUMBERS.")
fmt.Print("\n\n\n\n")
```
```python
# 这部分代码是游戏的介绍和说明，用于向玩家解释游戏规则和目标
func positionList() []int {
	// 创建一个长度为25的整数切片
	positions := make([]int, 25)
	// 将切片中的元素依次赋值为1到25
	for i := 0; i < 25; i++ {
		positions[i] = i + 1
	}
	// 返回包含1到25的整数切片
	return positions
}

// 从1到25的范围中随机选择4个位置
func generateEnemyPositions() []int {
	// 调用positionList函数获取包含1到25的整数切片
	positions := positionList()
	// 使用随机算法打乱切片中的元素顺序
	rand.Shuffle(len(positions), func(i, j int) { positions[i], positions[j] = positions[j], positions[i] })
	// 返回打乱后的切片中的前4个元素
	return positions[:4]
}

// 判断位置是否在1到25的范围内
func isValidPosition(p int) bool {
	// 如果位置在1到25的范围内，返回true，否则返回false
	return p >= 1 && p <= 25
}

func promptForPlayerPositions() []int {
	// 待补充
}
	scanner := bufio.NewScanner(os.Stdin)  // 创建一个从标准输入读取数据的 Scanner 对象
	var positions []int  // 创建一个整数类型的切片 positions

	for {  // 无限循环
		fmt.Println("\nWHAT ARE YOUR FOUR POSITIONS (1-25)?")  // 打印提示信息
		scanner.Scan()  // 读取输入
		rawPositions := strings.Split(scanner.Text(), " ")  // 将输入按空格分割成字符串切片

		if len(rawPositions) != 4 {  // 如果输入的位置数量不等于 4
			fmt.Println("PLEASE ENTER FOUR UNIQUE POSITIONS")  // 打印错误信息
			goto there  // 跳转到标签 there
		}

		for _, p := range rawPositions {  // 遍历输入的位置
			pos, err := strconv.Atoi(p)  // 将字符串转换为整数
			if (err != nil) || !isValidPosition(pos) {  // 如果转换出错或者位置不合法
				fmt.Println("ALL POSITIONS MUST RANGE (1-25)")  // 打印错误信息
				goto there  // 跳转到标签 there
			}
			positions = append(positions, pos)  // 将合法的位置添加到 positions 切片中
		}
		}
		if len(positions) == 4 {  # 如果 positions 列表的长度为 4
			return positions  # 返回 positions 列表
		}

	there:  # 定义标签 "there"
	}
}

func promptPlayerForTarget() int {  # 定义函数 promptPlayerForTarget，返回整数类型
	scanner := bufio.NewScanner(os.Stdin)  # 创建一个从标准输入读取数据的 Scanner 对象

	for {  # 进入无限循环
		fmt.Println("\nWHERE DO YOU WISH TO FIRE YOUR MISSILE?")  # 打印提示信息
		scanner.Scan()  # 从标准输入读取一行数据
		target, err := strconv.Atoi(scanner.Text())  # 将读取的数据转换为整数类型，同时检查是否有错误

		if (err != nil) || !isValidPosition(target) {  # 如果发生错误或者目标位置不合法
			fmt.Println("POSITIONS MUST RANGE (1-25)")  # 打印错误提示信息
			continue  # 继续下一次循环
		}
		return target
	}
}
```
这部分代码是一个函数的结束和一个函数的开始。

```
func generateAttackSequence() []int {
	positions := positionList()
	rand.Shuffle(len(positions), func(i, j int) { positions[i], positions[j] = positions[j], positions[i] })
	return positions
}
```
这部分代码定义了一个名为`generateAttackSequence`的函数，该函数返回一个整数切片。它调用了`positionList`函数来获取位置列表，然后使用`rand.Shuffle`函数对位置列表进行随机排序，并返回随机排序后的位置列表。

```
// Performs attack procedure returning True if we are to continue.
func attack(target int, positions *[]int, hitMsg, missMsg string, progressMsg []string) bool {
	for i := 0; i < len(*positions); i++ {
		if target == (*positions)[i] {
			fmt.Print(hitMsg)

			// remove the target just hit
			(*positions)[i] = (*positions)[len((*positions))-1]
			(*positions)[len((*positions))-1] = 0
```
这部分代码定义了一个名为`attack`的函数，该函数接受一个整数`target`、一个整数切片指针`positions`、两个字符串`hitMsg`和`missMsg`，以及一个字符串切片`progressMsg`作为参数，并返回一个布尔值。函数使用`for`循环遍历`positions`切片，如果`target`等于`positions`中的某个元素，则打印`hitMsg`。然后将被击中的目标从`positions`中移除。
			(*positions) = (*positions)[:len((*positions))-1]  # 从 positions 切片中移除最后一个元素

			if len((*positions)) != 0 {  # 如果 positions 切片不为空
				fmt.Print(progressMsg[len((*positions))])  # 打印进度消息中对应长度的消息
			} else {  # 如果 positions 切片为空
				fmt.Printf(progressMsg[len((*positions))], target)  # 打印进度消息中对应长度的消息，并传入目标值
			}
			return len((*positions)) > 0  # 返回 positions 切片的长度是否大于 0
		}
	}
	fmt.Print(missMsg)  # 打印未找到目标值的消息
	return len((*positions)) > 0  # 返回 positions 切片的长度是否大于 0
}

func main() {
	rand.Seed(time.Now().UnixNano())  # 使用当前时间的纳秒数作为随机数种子

	printIntro()  # 调用打印介绍的函数
	displayField()  # 调用显示字段的函数
	enemyPositions := generateEnemyPositions()  // 生成敌人的位置信息
	enemyAttacks := generateAttackSequence()  // 生成敌人的攻击顺序
	enemyAttackCounter := 0  // 初始化敌人的攻击计数器为0

	playerPositions := promptForPlayerPositions()  // 提示玩家输入自己的位置信息

	for {
		// 玩家发起攻击
		if !attack(promptPlayerForTarget(), &enemyPositions, "YOU GOT ONE OF MY OUTPOSTS!\n\n", "HA, HA YOU MISSED. MY TURN NOW:\n\n", PLAYER_PROGRESS_MESSAGES) {
			break  // 如果攻击失败则跳出循环
		}
		// 电脑发起攻击
		hitMsg := fmt.Sprintf("I GOT YOU. IT WON'T BE LONG NOW. POST %d WAS HIT.\n", enemyAttacks[enemyAttackCounter])  // 格式化命中消息
		missMsg := fmt.Sprintf("I MISSED YOU, YOU DIRTY RAT. I PICKED %d. YOUR TURN:\n\n", enemyAttacks[enemyAttackCounter])  // 格式化未命中消息
		if !attack(enemyAttacks[enemyAttackCounter], &playerPositions, hitMsg, missMsg, ENEMY_PROGRESS_MESSAGES) {
			break  // 如果攻击失败则跳出循环
		}
		enemyAttackCounter += 1  // 敌人的攻击计数器加1
	}
# 关闭 ZIP 对象
zip.close()  # 关闭 ZIP 对象，释放资源，避免内存泄漏。
```