# `basic-computer-games\00_Alternate_Languages\01_Acey_Ducey\c++\source\Aceyducey.h`

```py
// 防止头文件被重复包含
#pragma once

// 声明欢迎消息函数
void WelcomeMessage();
// 声明绘制卡片函数，参数为引用类型的短整型变量
void DrawCard(short int& Card);
// 声明游戏进行函数，参数为引用类型的布尔变量
void Play(bool& isPlaying);
// 声明判断游戏是否结束的函数，返回布尔类型
bool isGameOver();
// 声明整型变量 Money
int Money;
```