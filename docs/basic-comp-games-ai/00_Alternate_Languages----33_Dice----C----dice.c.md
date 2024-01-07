# `basic-computer-games\00_Alternate_Languages\33_Dice\C\dice.c`

```

#include <stdio.h> // 包含标准输入输出库
#include <stdlib.h> // 包含标准库
#include <time.h> // 包含时间库

float percent(int number, int total){ // 定义一个函数，计算一个数在总数中的百分比
    float percent; // 定义百分比变量
    percent = (float)number / (float)total * 100; // 计算百分比
    return percent; // 返回百分比
}

int main(){ // 主函数
    int dice1,dice2,times,rolls[13] = {0}; // 定义骰子1、骰子2、投掷次数、每个和出现次数的数组
    srand(time(NULL)); // 用当前时间作为随机数种子
    printf("This program simulates the rolling of a pair of dice\n"); // 输出提示信息
    printf("How many times do you want to roll the dice?(Higher the number longer the waiting time): "); // 输出提示信息
    scanf("%d",&times); // 读取用户输入的投掷次数
    for(int i = 0; i < times; i++){ // 循环投掷骰子
        dice1 = rand() % 6 + 1; // 生成1-6之间的随机数作为骰子1的点数
        dice2 = rand() % 6 + 1; // 生成1-6之间的随机数作为骰子2的点数
        rolls[dice1 + dice2]+=1; // 统计每个和出现的次数
    }
    printf("The number of times each sum was rolled is:\n"); // 输出提示信息
    for(int i = 2; i <= 12; i++){ // 遍历每个和出现的次数
        printf("%d: rolled %d times, or %f%c of the times\n",i,rolls[i],percent(rolls[i],times),(char)37); // 输出每个和出现的次数和百分比
    }
}

```