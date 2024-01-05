# `d:/src/tocomm/basic-computer-games\00_Alternate_Languages\01_Acey_Ducey\c++\source\Aceyducey.h`

```
#pragma once  // 防止头文件被多次包含

void WelcomeMessage();  // 声明欢迎消息函数

void DrawCard(short int& Card);  // 声明抽卡函数，参数为引用类型的卡片变量

void Play(bool& isPlaying);  // 声明游戏进行函数，参数为引用类型的游戏状态变量

bool isGameOver();  // 声明游戏结束判断函数，返回布尔类型值

int Money;  // 声明整型变量 Money
```