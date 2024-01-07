# `basic-computer-games\00_Alternate_Languages\25_Chief\C\chief.c`

```

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

//check if windows or linux for the clear screen
#ifdef _WIN32
#define CLEAR "cls"  // 定义清屏命令为 cls
#else
#define CLEAR "clear"  // 定义清屏命令为 clear
#endif

void show_solution(float guess);  // 声明函数 show_solution，用于展示解题过程
float guess_number(float number);  // 声明函数 guess_number，用于计算猜测的数字
void game();  // 声明函数 game，用于进行游戏

float guess_number(float number){  // 定义函数 guess_number，用于计算猜测的数字
    float guess;
    guess = ((((number - 4) * 5) / 8) * 5 - 3);  // 根据给定的计算公式计算猜测的数字
    return guess;  // 返回计算结果
}

void game(){  // 定义函数 game，用于进行游戏
    float number,guess;  // 定义变量 number 和 guess
    char answer[4];  // 定义字符数组 answer，用于存储用户输入的答案
    printf("Think a number\n");  // 输出提示信息
    // ...（以下类似，输出一系列提示信息，获取用户输入，进行判断和处理）
}

void show_solution(float guess){  // 定义函数 show_solution，用于展示解题过程
    // ...（根据猜测的数字展示解题过程）
}

void main(){  // 主函数
    char answer[4];  // 定义字符数组 answer，用于存储用户输入的答案
    printf("I am CHIEF NUMBERS FREEK, The GREAT INDIAN MATH GOD.\n");  // 输出欢迎信息
    // ...（获取用户输入，进行判断和处理）
}

```