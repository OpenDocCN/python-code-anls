# `basic-computer-games\00_Alternate_Languages\45_Hello\go\main.go`

```

package main

import (
	"bufio" // 导入 bufio 包，用于读取输入
	"fmt" // 导入 fmt 包，用于格式化输出
	"os" // 导入 os 包，用于操作系统功能
	"strings" // 导入 strings 包，用于处理字符串
	"time" // 导入 time 包，用于时间相关操作
)

type PROBLEM_TYPE int8 // 定义一个枚举类型 PROBLEM_TYPE

const (
	SEX PROBLEM_TYPE = iota // 定义枚举常量 SEX
	HEALTH // 定义枚举常量 HEALTH
	MONEY // 定义枚举常量 MONEY
	JOB // 定义枚举常量 JOB
	UKNOWN // 定义枚举常量 UKNOWN
)

func getYesOrNo() (bool, bool, string) { // 定义函数，返回两个布尔值和一个字符串
	scanner := bufio.NewScanner(os.Stdin) // 创建一个从标准输入读取数据的 Scanner

	scanner.Scan() // 读取输入

	if strings.ToUpper(scanner.Text()) == "YES" { // 判断输入是否为 "YES"
		return true, true, scanner.Text() // 返回 true, true, 输入的字符串
	} else if strings.ToUpper(scanner.Text()) == "NO" { // 判断输入是否为 "NO"
		return true, false, scanner.Text() // 返回 true, false, 输入的字符串
	} else {
		return false, false, scanner.Text() // 返回 false, false, 输入的字符串
	}
}

// 其余函数的作用和功能类似，这里不再重复注释

```