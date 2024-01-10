# `basic-computer-games\00_Alternate_Languages\25_Chief\C\chief.c`

```
#include <stdio.h>  // 包含标准输入输出库
#include <stdlib.h>  // 包含标准库
#include <string.h>  // 包含字符串处理库
#include <ctype.h>   // 包含字符处理库

// 根据操作系统定义清屏命令
#ifdef _WIN32
#define CLEAR "cls"
#else
#define CLEAR "clear"
#endif

// 声明函数，用于显示猜测的解决方案
void show_solution(float guess);
// 声明函数，用于猜测数字
float guess_number(float number);
// 声明函数，用于进行游戏
void game();

// 定义函数，用于猜测数字
float guess_number(float number){
    float guess;
    // 计算猜测的数字
    guess = ((((number - 4) * 5) / 8) * 5 - 3);
    return guess;
}

// 定义函数，用于进行游戏
void game(){
    float number,guess;
    char answer[4];
    // 提示用户进行计算
    printf("Think a number\n");
    printf("Then add to it 3 and divide it by 5\n");
    printf("Now multiply by 8, divide by 5 and then add 5\n");
    printf("Finally substract 1\n");
    printf("What is the number you got?(if you got decimals put them ex: 23.6): ");
    // 获取用户输入的数字
    scanf("%f",&number);
    // 进行猜测
    guess = guess_number(number);
    // 提示用户猜测的数字
    printf("The number you thought was %f am I right(Yes or No)?\n",guess);
    // 获取用户回答
    scanf("%s",answer);
    // 将回答转换为小写字母
    for(int i = 0; i < strlen(answer); i++){
        answer[i] = tolower(answer[i]);
    }
    // 根据用户回答进行处理
    if(strcmp(answer,"yes") == 0){
        printf("\nHuh, I Knew I was unbeatable");
        printf("And here is how i did it:\n");
        show_solution(guess);
    }
    else if (strcmp(answer,"no") == 0){
        printf("HUH!! what was you original number?: ");
        scanf("%f",&number);
        if(number == guess){
            printf("Huh, I Knew I was unbeatable");
            printf("And here is how i did it:\n");
            show_solution(guess);
        }
        else{
            printf("If I got it wrong I guess you are smarter than me");
        }
    }
    else{
        system(CLEAR);
        printf("I don't understand what you said\n");
        printf("Please answer with Yes or No\n");
        game();
    }

}

// 定义函数，用于显示猜测的解决方案
void show_solution(float guess){
    printf("%f plus 3 is %f\n",guess,guess + 3);
    printf("%f divided by 5 is %f\n",guess + 3,(guess + 3) / 5);
    printf("%f multiplied by 8 is %f\n",(guess + 3) / 5,(guess + 3) / 5 * 8);
    # 打印 guess 加 3 除以 5 乘以 8 的结果，以及该结果除以 5 的结果
    printf("%f divided by 5 is %f\n",(guess + 3) / 5 * 8,(guess + 3) / 5 * 8 / 5);
    # 打印 guess 加 3 除以 5 乘以 8 除以 5 的结果，以及该结果加上 5 的结果
    printf("%f plus 5 is %f\n",(guess + 3) / 5 * 8 / 5,(guess + 3) / 5 * 8 / 5 + 5);
    # 打印 guess 加 3 除以 5 乘以 8 除以 5 加上 5 的结果，以及该结果减去 1 的结果
    printf("%f minus 1 is %f\n",(guess + 3) / 5 * 8 / 5 + 5,(guess + 3) / 5 * 8 / 5 + 5 - 1);
// 主函数
void main(){
    // 定义存储用户输入的答案的字符数组
    char answer[4];
    // 打印提示信息
    printf("I am CHIEF NUMBERS FREEK, The GREAT INDIAN MATH GOD.\n");
    printf("Are you ready to take the test you called me out for(Yes or No)? ");
    // 读取用户输入的答案
    scanf("%s",answer);
    // 将用户输入的答案转换为小写
    for(int i = 0; i < strlen(answer); i++){
        answer[i] = tolower(answer[i]);
    }
    // 判断用户输入的答案
    if(strcmp(answer,"yes") == 0){
        // 如果答案是yes，则调用game函数
        game();
    }else if (strcmp(answer,"no") == 0){
        // 如果答案是no，则打印提示信息
        printf("You are a coward, I will not play with you.%d %s\n",strcmp(answer,"yes"),answer);
    }
    else{
        // 如果答案既不是yes也不是no，则打印提示信息，清空屏幕，重新调用main函数
        system(CLEAR);
        printf("I don't understand what you said\n");
        printf("Please answer with Yes or No\n");
        main();
    }
}
```