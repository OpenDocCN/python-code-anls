# `d:/src/tocomm/basic-computer-games\00_Alternate_Languages\42_Gunner\go\main.go`

```
package main  # 声明当前文件所属的包

import (  # 导入需要使用的包
	"bufio"  # 用于提供带缓冲的 I/O
	"fmt"  # 用于格式化输入输出
	"math"  # 提供数学函数
	"math/rand"  # 提供伪随机数生成
	"os"  # 提供对操作系统功能的访问
	"strconv"  # 提供字符串和基本数据类型之间的转换
	"strings"  # 提供对字符串的操作
	"time"  # 提供时间的功能
)

func printIntro() {  # 定义一个名为 printIntro 的函数
	fmt.Println("                                 GUNNER")  # 打印输出指定的字符串
	fmt.Println("               CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")  # 打印输出指定的字符串
	fmt.Print("\n\n\n")  # 打印输出指定的字符串并换行
	fmt.Println("YOU ARE THE OFFICER-IN-CHARGE, GIVING ORDERS TO A GUN")  # 打印输出指定的字符串
	fmt.Println("CREW, TELLING THEM THE DEGREES OF ELEVATION YOU ESTIMATE")  # 打印输出指定的字符串
	fmt.Println("WILL PLACE A PROJECTILE ON TARGET.  A HIT WITHIN 100 YARDS")  # 打印输出指定的字符串
	fmt.Println("OF THE TARGET WILL DESTROY IT.")  // 打印字符串 "OF THE TARGET WILL DESTROY IT."
	fmt.Println()  // 打印空行
}

func getFloat() float64 {
	scanner := bufio.NewScanner(os.Stdin)  // 创建一个从标准输入读取数据的 Scanner 对象
	for {
		scanner.Scan()  // 读取下一行输入
		fl, err := strconv.ParseFloat(scanner.Text(), 64)  // 将输入的字符串转换为 float64 类型

		if err != nil {  // 如果转换出错
			fmt.Println("Invalid input")  // 打印错误信息
			continue  // 继续循环，等待有效输入
		}

		return fl  // 返回转换后的 float64 值
	}
}

func play() {
	gunRange := int(40000*rand.Float64() + 20000)  // 生成一个随机的枪的射程范围，单位为码
	fmt.Printf("\nMAXIMUM RANGE OF YOUR GUN IS %d YARDS\n", gunRange)  // 打印枪的最大射程

	killedEnemies := 0  // 记录击毙的敌人数量
	S1 := 0  // 初始化 S1 变量

	for {  // 进入无限循环
		targetDistance := int(float64(gunRange) * (0.1 + 0.8*rand.Float64()))  // 生成一个随机的目标距离，单位为码
		shots := 0  // 初始化射击次数

		fmt.Printf("\nDISTANCE TO THE TARGET IS %d YARDS\n", targetDistance)  // 打印目标距离

		for {  // 进入内层无限循环
			fmt.Print("\n\nELEVATION? ")  // 打印提示信息，要求输入枪的仰角
			elevation := getFloat()  // 获取用户输入的枪的仰角

			if elevation > 89 {  // 如果仰角大于89度
				fmt.Println("MAXIMUM ELEVATION IS 89 DEGREES")  // 打印提示信息
				continue  // 继续下一次循环
			}
			if elevation < 1 {  // 如果仰角小于1度
				fmt.Println("MINIMUM ELEVATION IS 1 DEGREE")  // 打印最小仰角为1度
				continue  // 继续下一次循环
			}

			shots += 1  // 射击次数加1

			if shots < 6 {  // 如果射击次数小于6
				B2 := 2 * elevation / 57.3  // 计算 B2 角度
				shotImpact := int(float64(gunRange) * math.Sin(B2))  // 计算射击影响
				shotProximity := int(targetDistance - shotImpact)  // 计算射击接近度

				if math.Abs(float64(shotProximity)) < 100 {  // 如果射击接近度绝对值小于100
					fmt.Printf("*** TARGET DESTROYED *** %d ROUNDS OF AMMUNITION EXPENDED.\n", shots)  // 打印目标被摧毁，消耗弹药数量
					S1 += shots  // S1 增加射击次数

					if killedEnemies == 4 {  // 如果击毙敌人数量为4
						fmt.Printf("\n\nTOTAL ROUNDS EXPENDED WERE: %d\n", S1)  // 打印总共消耗的弹药数量
						if S1 > 18 {  // 如果总共消耗的弹药数量大于18
# 如果射击距离小于等于10码，则输出"DIRECT HIT !!"，然后返回
if shotProximity <= 10 {
    fmt.Println("DIRECT HIT !!")
    return
} else {
    # 如果射击距离大于10码且小于等于50码，则输出"GOOD SHOT !!"，然后返回
    if shotProximity <= 50 {
        fmt.Println("GOOD SHOT !!")
        return
    } else {
        # 如果射击距离大于50码且小于等于100码，则输出"BETTER GO BACK TO FORT SILL FOR REFRESHER TRAINING!"，然后返回
        if shotProximity <= 100 {
            fmt.Println("BETTER GO BACK TO FORT SILL FOR REFRESHER TRAINING!")
            return
        } else {
            # 如果射击距离大于100码，则输出"NICE SHOOTING !!"，然后返回
            fmt.Println("NICE SHOOTING !!")
            return
        }
    }
}
# 如果射击未命中，则根据射击距离输出相应的信息
if shotProximity > 100 {
    fmt.Printf("SHORT OF TARGET BY %d YARDS.\n", int(math.Abs(float64(shotProximity))))
} else {
    fmt.Printf("OVER TARGET BY %d YARDS.\n", int(math.Abs(float64(shotProximity))))
}
# 如果射击距离大于10码，则输出"BOOM !!!!   YOU HAVE JUST BEEN DESTROYED BY THE ENEMY."
fmt.Print("\nBOOM !!!!   YOU HAVE JUST BEEN DESTROYED BY THE ENEMY.\n\n\n")
				fmt.Println("BETTER GO BACK TO FORT SILL FOR REFRESHER TRAINING!")
```
这行代码用于在控制台打印一条警告信息。

```python
				return
```
这行代码用于结束当前函数的执行并返回到调用它的地方。

```python
			}
		}
	}
}
```
这几行代码用于结束之前的循环和函数定义。

```python
func main() {
```
这行代码定义了一个名为main的函数，是程序的入口点。

```python
	rand.Seed(time.Now().UnixNano())
```
这行代码用于设置随机数种子，以确保每次运行程序时生成的随机数不同。

```python
	scanner := bufio.NewScanner(os.Stdin)
```
这行代码用于创建一个从标准输入读取数据的扫描器。

```python
	printIntro()
```
这行代码调用了一个名为printIntro的函数，用于在控制台打印一些介绍性的信息。

```python
	for {
		play()
```
这几行代码定义了一个无限循环，每次循环调用play函数。

```python
		fmt.Print("TRY AGAIN (Y OR N)? ")
		scanner.Scan()
```
这两行代码用于在控制台打印提示信息，并从标准输入读取用户输入的内容。

```python
		if strings.ToUpper(scanner.Text())[0:1] != "Y" {
```
这行代码用于判断用户输入的内容是否以大写字母Y开头，如果不是则执行下一步操作。
# 打印提示信息，表示程序执行完成，返回基地营地
fmt.Println("\nOK. RETURN TO BASE CAMP.")
# 跳出循环，结束程序执行
break
```