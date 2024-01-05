# `00_Alternate_Languages\44_Hangman\C\main.c`

```
#include <stdio.h>  // 包含标准输入输出库
#include <stdlib.h>  // 包含标准库
#include <string.h>  // 包含字符串处理库
#include <time.h>    // 包含时间库
#define MAX_WORDS 100  // 定义最大单词数量为100

// 根据操作系统选择清屏命令
#ifdef _WIN32
#define CLEAR "cls"
#else
#define CLEAR "clear"
#endif

/**
 * @brief 根据猜错的次数打印 hangman 的阶段。
 * 
 * @param stage hangman 的阶段。
 */
void print_hangman(int stage){
    switch (stage){  // 根据猜错的次数选择不同的 hangman 阶段
        case 0:  # 如果情况为0
            printf("----------\n");  # 打印一行横线
            printf("|        |\n");  # 打印一个空格和竖线
            printf("|\n");  # 打印一个竖线
            printf("|\n");  # 打印一个竖线
            printf("|\n");  # 打印一个竖线
            printf("|\n");  # 打印一个竖线
            break;  # 结束此情况
        case 1:  # 如果情况为1
            printf("----------\n");  # 打印一行横线
            printf("|        |\n");  # 打印一个空格和竖线
            printf("|        O\n");  # 打印一个空格、竖线和字母O
            printf("|        |\n");  # 打印一个空格、竖线和竖线
            printf("|\n");  # 打印一个竖线
            printf("|\n");  # 打印一个竖线
            break;  # 结束此情况
        case 2:  # 如果情况为2
            printf("----------\n");  # 打印一行横线
            printf("|        |\n");  # 打印一个空格和竖线
            printf("|        o\n");  # 打印一个空格、竖线和小写字母o
            printf("|       /|\n");  # 打印一个人的头部和身体
            printf("|\n");  # 打印一个人的头部和身体
            printf("|\n");  # 打印一个人的头部和身体
            break;  # 结束当前的 case 分支
        case 3:
            printf("----------\n");  # 打印一个人的头部和身体
            printf("|        |\n");  # 打印一个人的头部和身体
            printf("|        o\n");  # 打印一个人的头部和身体
            printf("|       /|\\\n");  # 打印一个人的头部和身体
            printf("|\n");  # 打印一个人的头部和身体
            printf("|\n");  # 打印一个人的头部和身体
            break;  # 结束当前的 case 分支
        case 4:
            printf("----------\n");  # 打印一个人的头部和身体
            printf("|        |\n");  # 打印一个人的头部和身体
            printf("|        o\n");  # 打印一个人的头部和身体
            printf("|       /|\\\n");  # 打印一个人的头部和身体
            printf("|       /\n");  # 打印一个人的头部和身体
            printf("|\n");  # 打印一个人的头部和身体
            break;  # 结束当前的 case 分支
        case 5:  // 当 case 的值为 5 时执行以下代码
            printf("----------\n");  // 打印一行字符串
            printf("|        |\n");  // 打印一行字符串
            printf("|        o\n");  // 打印一行字符串
            printf("|       /|\\\n");  // 打印一行字符串
            printf("|       / \\\n");  // 打印一行字符串
            printf("|\n");  // 打印一行字符串
            break;  // 结束 switch 语句
        default:  // 如果 case 的值不是 5，则执行以下代码
            break;  // 结束 switch 语句
    }
}

/**
 * @brief 从字典中随机选择并返回一个单词。
 * 
 * @return 随机单词
 */
char* random_word_picker(){
    // 生成一个随机的英文单词
    # 分配内存以存储一个包含100个字符的字符串
    char* word = malloc(sizeof(char) * 100);
    # 以只读模式打开名为 "dictionary.txt" 的文件
    FILE* fp = fopen("dictionary.txt", "r");
    # 用当前时间作为随机数生成器的种子
    srand(time(NULL));
    # 如果文件指针为空，打印错误消息并退出程序
    if (fp == NULL){
        printf("Error opening dictionary.txt\n");
        exit(1);
    }
    # 生成一个小于 MAX_WORDS 的随机数
    int random_number = rand() % MAX_WORDS;
    # 从文件中读取随机数指定的行数的单词
    for (int j = 0; j < random_number; j++){
        fscanf(fp, "%s", word);
    }
    # 关闭文件
    fclose(fp);
    # 返回读取的单词
    return word;
}

void main(void){
    # 分配内存以存储一个包含100个字符的字符串
    char* word = malloc(sizeof(char) * 100);
```
这段代码是一个C语言程序，主要功能是从名为 "dictionary.txt" 的文件中随机读取一个单词并返回。首先使用 `malloc` 函数分配了一个包含100个字符的字符串的内存空间，然后以只读模式打开了 "dictionary.txt" 文件。接着使用 `srand` 函数设置随机数生成器的种子，然后生成一个小于 `MAX_WORDS` 的随机数。随后使用 `fscanf` 函数从文件中读取随机数指定的行数的单词，并将其存储在 `word` 变量中。最后关闭文件并返回读取的单词。在 `main` 函数中也分配了一个包含100个字符的字符串的内存空间。
    // 从随机单词选择器中获取一个单词
    word = random_word_picker();
    // 分配内存以存储隐藏的单词
    char* hidden_word = malloc(sizeof(char) * 100);
    // 将隐藏的单词初始化为下划线
    for (int i = 0; i < strlen(word); i++){
        hidden_word[i] = '_';
    }
    // 在隐藏的单词末尾添加字符串终止符
    hidden_word[strlen(word)] = '\0';
    // 初始化游戏阶段、错误猜测次数和正确猜测次数
    int stage = 0;
    int wrong_guesses = 0;
    int correct_guesses = 0;
    // 分配内存以存储猜测
    char* guess = malloc(sizeof(char) * 100);
    // 当错误猜测次数小于6且正确猜测次数小于单词长度时，执行循环
    while (wrong_guesses < 6 && correct_guesses < strlen(word)){
        // 清空屏幕
        CLEAR;
        // 打印当前的“绞刑”图案
        print_hangman(stage);
        // 打印隐藏的单词
        printf("%s\n", hidden_word);
        // 提示用户输入猜测
        printf("Enter a guess: ");
        // 读取用户输入的猜测
        scanf("%s", guess);
        // 遍历单词的每个字符
        for (int i = 0; i < strlen(word); i++){
            // 如果猜测与单词相同
            if (strcmp(guess,word) == 0){
                // 将正确猜测次数设置为单词长度
                correct_guesses = strlen(word);
            }
```
            else if (guess[0] == word[i]){  // 如果猜测的字母在单词中出现
                hidden_word[i] = guess[0];  // 将猜测的字母添加到已猜中的单词中
                correct_guesses++;  // 正确猜中的次数加一
            }
        }
        if (strchr(word, guess[0]) == NULL){  // 如果猜测的字母不在单词中出现
            wrong_guesses++;  // 错误猜测的次数加一
        }
        stage = wrong_guesses;  // 更新当前游戏阶段为错误猜测的次数
    }
    if (wrong_guesses == 6){  // 如果错误猜测的次数达到6次
        printf("You lose! The word was %s\n", word);  // 打印出游戏失败的消息和正确的单词
    }
    else {
        printf("You win!\n");  // 打印出游戏胜利的消息
    }
}
```