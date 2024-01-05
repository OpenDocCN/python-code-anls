# `d:/src/tocomm/basic-computer-games\00_Alternate_Languages\25_Chief\C\chief.c`

```
#include <stdio.h>  // 包含标准输入输出库
#include <stdlib.h>  // 包含标准库
#include <string.h>  // 包含字符串处理库
#include <ctype.h>   // 包含字符处理库

// 根据操作系统定义清空屏幕的命令
#ifdef _WIN32
#define CLEAR "cls"
#else
#define CLEAR "clear"
#endif

void show_solution(float guess);  // 声明显示答案的函数
float guess_number(float number);  // 声明猜测数字的函数
void game();  // 声明游戏函数

float guess_number(float number){  // 定义猜测数字的函数
    float guess;  // 定义猜测值变量
    guess = ((((number - 4) * 5) / 8) * 5 - 3);  // 计算猜测值
    return guess;  // 返回猜测值
}

void game(){
    float number,guess; // 声明浮点数变量number和guess
    char answer[4]; // 声明字符数组answer，用于存储用户输入的答案
    printf("Think a number\n"); // 输出提示信息
    printf("Then add to it 3 and divide it by 5\n"); // 输出提示信息
    printf("Now multiply by 8, divide by 5 and then add 5\n"); // 输出提示信息
    printf("Finally substract 1\n"); // 输出提示信息
    printf("What is the number you got?(if you got decimals put them ex: 23.6): "); // 输出提示信息，要求用户输入一个数字
    scanf("%f",&number); // 从用户输入中读取一个浮点数并存储在变量number中
    guess = guess_number(number); // 调用guess_number函数，传入number作为参数，将返回值赋给guess
    printf("The number you thought was %f am I right(Yes or No)?\n",guess); // 输出猜测的数字，并询问用户是否正确
    scanf("%s",answer); // 从用户输入中读取一个字符串并存储在数组answer中
    for(int i = 0; i < strlen(answer); i++){ // 循环遍历answer数组中的每个字符
        answer[i] = tolower(answer[i]); // 将每个字符转换为小写字母
    }
    if(strcmp(answer,"yes") == 0){ // 比较answer数组中的字符串是否等于"yes"
        printf("\nHuh, I Knew I was unbeatable"); // 输出提示信息
        printf("And here is how i did it:\n"); // 输出提示信息
        show_solution(guess);  # 调用名为show_solution的函数，传入参数guess
    }
    else if (strcmp(answer,"no") == 0){  # 如果用户输入的答案是"no"
        printf("HUH!! what was you original number?: ");  # 输出提示信息
        scanf("%f",&number);  # 从用户输入中读取一个浮点数，存入变量number
        if(number == guess){  # 如果用户输入的数字等于程序猜测的数字
            printf("Huh, I Knew I was unbeatable");  # 输出提示信息
            printf("And here is how i did it:\n");  # 输出提示信息
            show_solution(guess);  # 调用名为show_solution的函数，传入参数guess
        }
        else{  # 如果用户输入的数字不等于程序猜测的数字
            printf("If I got it wrong I guess you are smarter than me");  # 输出提示信息
        }
    }
    else{  # 如果用户输入的既不是"yes"也不是"no"
        system(CLEAR);  # 调用系统命令清空屏幕
        printf("I don't understand what you said\n");  # 输出提示信息
        printf("Please answer with Yes or No\n");  # 输出提示信息
        game();  # 调用名为game的函数
    }
void show_solution(float guess){
    // 打印猜测值加3的结果
    printf("%f plus 3 is %f\n",guess,guess + 3);
    // 打印猜测值加3再除以5的结果
    printf("%f divided by 5 is %f\n",guess + 3,(guess + 3) / 5);
    // 打印猜测值加3再除以5再乘以8的结果
    printf("%f multiplied by 8 is %f\n",(guess + 3) / 5,(guess + 3) / 5 * 8);
    // 打印猜测值加3再除以5再乘以8再除以5的结果
    printf("%f divided by 5 is %f\n",(guess + 3) / 5 * 8,(guess + 3) / 5 * 8 / 5);
    // 打印猜测值加3再除以5再乘以8再除以5再加5的结果
    printf("%f plus 5 is %f\n",(guess + 3) / 5 * 8 / 5,(guess + 3) / 5 * 8 / 5 + 5);
    // 打印猜测值加3再除以5再乘以8再除以5再加5再减去1的结果
    printf("%f minus 1 is %f\n",(guess + 3) / 5 * 8 / 5 + 5,(guess + 3) / 5 * 8 / 5 + 5 - 1);
}

void main(){
    // 创建一个长度为4的字符数组来存储用户输入的答案
    char answer[4];
    // 打印提示信息，询问用户是否准备好进行测试
    printf("I am CHIEF NUMBERS FREEK, The GREAT INDIAN MATH GOD.\n");
    printf("Are you ready to take the test you called me out for(Yes or No)? ");
    // 读取用户输入的答案
    scanf("%s",answer);
    // 将用户输入的答案转换为小写形式
    for(int i = 0; i < strlen(answer); i++){
        answer[i] = tolower(answer[i]);
    }
    # 如果用户输入的答案是 "yes"，则调用 game() 函数
    if(strcmp(answer,"yes") == 0){
        game();
    }
    # 如果用户输入的答案是 "no"，则输出相应的提示信息
    else if (strcmp(answer,"no") == 0){
        printf("You are a coward, I will not play with you.%d %s\n",strcmp(answer,"yes"),answer);
    }
    # 如果用户输入的既不是 "yes" 也不是 "no"，则清空屏幕并输出提示信息，然后重新调用 main() 函数
    else{
        system(CLEAR);
        printf("I don't understand what you said\n");
        printf("Please answer with Yes or No\n");
        main();
    }
}
```