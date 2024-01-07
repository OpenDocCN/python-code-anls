# `basic-computer-games\00_Alternate_Languages\01_Acey_Ducey\c++\source\Aceyducey.h`

```

// 使用 #pragma once 指令确保头文件只被编译一次
#pragma once

// 声明欢迎消息函数
void WelcomeMessage();
// 声明绘制卡片函数，参数为引用类型的短整型变量
void DrawCard(short int& Card);
// 声明游戏进行函数，参数为引用类型的布尔变量
void Play(bool& isPlaying);
// 声明判断游戏是否结束的函数
bool isGameOver();
// 声明整型变量 Money，用于存储游戏中的金钱数额
int Money;

```