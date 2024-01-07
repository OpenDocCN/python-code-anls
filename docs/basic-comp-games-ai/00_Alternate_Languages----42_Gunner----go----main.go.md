# `basic-computer-games\00_Alternate_Languages\42_Gunner\go\main.go`

```

package main

import (
	"bufio" // 导入 bufio 包，用于读取输入
	"fmt" // 导入 fmt 包，用于格式化输出
	"math" // 导入 math 包，用于数学计算
	"math/rand" // 导入 math/rand 包，用于生成随机数
	"os" // 导入 os 包，用于操作系统功能
	"strconv" // 导入 strconv 包，用于字符串转换
	"strings" // 导入 strings 包，用于字符串操作
	"time" // 导入 time 包，用于时间相关功能
)

func printIntro() {
	// 打印游戏介绍
	fmt.Println("                                 GUNNER")
	fmt.Println("               CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
	fmt.Print("\n\n\n")
	fmt.Println("YOU ARE THE OFFICER-IN-CHARGE, GIVING ORDERS TO A GUN")
	fmt.Println("CREW, TELLING THEM THE DEGREES OF ELEVATION YOU ESTIMATE")
	fmt.Println("WILL PLACE A PROJECTILE ON TARGET.  A HIT WITHIN 100 YARDS")
	fmt.Println("OF THE TARGET WILL DESTROY IT.")
	fmt.Println()
}

func getFloat() float64 {
	// 从标准输入获取浮点数
	scanner := bufio.NewScanner(os.Stdin)
	for {
		scanner.Scan()
		fl, err := strconv.ParseFloat(scanner.Text(), 64)

		if err != nil {
			fmt.Println("Invalid input")
			continue
		}

		return fl
	}
}

func play() {
	// 随机生成敌人距离和枪的射程
	gunRange := int(40000*rand.Float64() + 20000)
	fmt.Printf("\nMAXIMUM RANGE OF YOUR GUN IS %d YARDS\n", gunRange)

	killedEnemies := 0
	S1 := 0

	for {
		// 随机生成目标距离
		targetDistance := int(float64(gunRange) * (0.1 + 0.8*rand.Float64()))
		shots := 0

		fmt.Printf("\nDISTANCE TO THE TARGET IS %d YARDS\n", targetDistance)

		for {
			fmt.Print("\n\nELEVATION? ")
			elevation := getFloat()

			if elevation > 89 {
				fmt.Println("MAXIMUM ELEVATION IS 89 DEGREES")
				continue
			}

			if elevation < 1 {
				fmt.Println("MINIMUM ELEVATION IS 1 DEGREE")
				continue
			}

			shots += 1

			if shots < 6 {
				B2 := 2 * elevation / 57.3
				shotImpact := int(float64(gunRange) * math.Sin(B2))
				shotProximity := int(targetDistance - shotImpact)

				if math.Abs(float64(shotProximity)) < 100 { // 击中目标
					fmt.Printf("*** TARGET DESTROYED *** %d ROUNDS OF AMMUNITION EXPENDED.\n", shots)
					S1 += shots

					if killedEnemies == 4 {
						fmt.Printf("\n\nTOTAL ROUNDS EXPENDED WERE: %d\n", S1)
						if S1 > 18 {
							print("BETTER GO BACK TO FORT SILL FOR REFRESHER TRAINING!")
							return
						} else {
							print("NICE SHOOTING !!")
							return
						}
					} else {
						killedEnemies += 1
						fmt.Println("\nTHE FORWARD OBSERVER HAS SIGHTED MORE ENEMY ACTIVITY...")
						break
					}
				} else { // 未击中
					if shotProximity > 100 {
						fmt.Printf("SHORT OF TARGET BY %d YARDS.\n", int(math.Abs(float64(shotProximity))))
					} else {
						fmt.Printf("OVER TARGET BY %d YARDS.\n", int(math.Abs(float64(shotProximity))))
					}
				}
			} else {
				fmt.Print("\nBOOM !!!!   YOU HAVE JUST BEEN DESTROYED BY THE ENEMY.\n\n\n")
				fmt.Println("BETTER GO BACK TO FORT SILL FOR REFRESHER TRAINING!")
				return
			}
		}
	}
}

func main() {
	rand.Seed(time.Now().UnixNano()) // 使用当前时间作为随机数种子
	scanner := bufio.NewScanner(os.Stdin)

	printIntro() // 打印游戏介绍

	for {
		play() // 开始游戏

		fmt.Print("TRY AGAIN (Y OR N)? ")
		scanner.Scan()

		if strings.ToUpper(scanner.Text())[0:1] != "Y" {
			fmt.Println("\nOK. RETURN TO BASE CAMP.")
			break
		}
	}
}

```