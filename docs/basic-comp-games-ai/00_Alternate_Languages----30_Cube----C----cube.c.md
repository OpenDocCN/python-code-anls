# `basic-computer-games\00_Alternate_Languages\30_Cube\C\cube.c`

```

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

//check if windows or linux for the clear screen
#ifdef _WIN32
#define CLEAR "cls"  // 定义清屏命令为 cls
#else
#define CLEAR "clear"  // 定义清屏命令为 clear
#endif

typedef struct{
    int x;
    int y;
    int z;
}coords;  // 定义一个结构体 coords，包含三个整型变量 x, y, z

void instuctions(){
    // 输出游戏说明
    printf("\nThis is a game in which you will be playing against the\n");
    printf("random decisions of the computer. The field of play is a\n");
    printf("cube of side 3. Any of the 27 locations can be designated\n");
    printf("by inputing three numbers such as 2,3,1. At the start,\n");
    printf("you are automatically at location 1,1,1. The object of\n");
    printf("the game is to get to location 3,3,3. One minor detail:\n");
    printf("the computer will pick, at random, 5 locations at which\n");
    printf("it will plant land mines. If you hit one of these locations\n");
    printf("you lose. One other detail: You may move only one space\n");
    printf("in one direction each move. For example: From 1,1,2 you\n");
    printf("may move to 2,1,2 or 1,1,3. You may not change\n");
    printf("two of the numbers on the same move. If you make an illegal\n");
    printf("move, you lose and the computer takes the money you may\n");
    printf("have bet on that round.\n\n");
    printf("When stating the amount of a wager, printf only the number\n");
    printf("of dollars (example: 250) you are automatically started with\n");
    printf("500 dollars in your account.\n\n");
    printf("Good luck!\n");
}

}

void init(){
    int account = 500;  // 初始化账户余额为 500
    char choice;

    printf("Welcome to the game of Cube!\n");
    printf("wanna see the instructions? (y/n): ");
    scanf("%c",&choice);  // 获取用户输入的选择
    if(choice == 'y'){
        system(CLEAR);  // 清屏
        instuctions();  // 显示游戏说明
    }
    else if (choice == 'n'){
        system(CLEAR);  // 清屏
        printf("Ok, let's play!\n");
    }
    else{
        system(CLEAR);  // 清屏
        printf("Invalid choice. Try again...\n");
        init();  // 重新初始化
        exit(0);  // 退出程序
    }
    game(account);  // 调用游戏函数
    exit(0);  // 退出程序
}

void main(){
    init();  // 调用初始化函数
}

```