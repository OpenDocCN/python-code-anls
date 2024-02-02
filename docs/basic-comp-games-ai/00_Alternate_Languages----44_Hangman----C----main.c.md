# `basic-computer-games\00_Alternate_Languages\44_Hangman\C\main.c`

```py
#include <stdio.h>  // 包含标准输入输出库
#include <stdlib.h>  // 包含标准库
#include <string.h>  // 包含字符串处理库
#include <time.h>    // 包含时间库
#define MAX_WORDS 100  // 定义最大单词数量

// 根据操作系统定义清屏命令
#ifdef _WIN32
#define CLEAR "cls"
#else
#define CLEAR "clear"
#endif

/**
 * @brief 根据错误猜测的次数打印不同阶段的“绞刑”图案
 * 
 * @param stage 绞刑阶段
 */
void print_hangman(int stage){
    switch (stage){
        case 0:
            printf("----------\n");
            printf("|        |\n");
            printf("|\n");
            printf("|\n");
            printf("|\n");
            printf("|\n");
            break;
        case 1:
            printf("----------\n");
            printf("|        |\n");
            printf("|        O\n");
            printf("|        |\n");
            printf("|\n");
            printf("|\n");
            break;
        case 2:
            printf("----------\n");
            printf("|        |\n");
            printf("|        o\n");
            printf("|       /|\n");
            printf("|\n");
            printf("|\n");
            break;
        case 3:
            printf("----------\n");
            printf("|        |\n");
            printf("|        o\n");
            printf("|       /|\\\n");
            printf("|\n");
            printf("|\n");
            break; 
        case 4:
            printf("----------\n");
            printf("|        |\n");
            printf("|        o\n");
            printf("|       /|\\\n");
            printf("|       /\n");
            printf("|\n");
            break; 
        case 5:
            printf("----------\n");
            printf("|        |\n");
            printf("|        o\n");
            printf("|       /|\\\n");
            printf("|       / \\\n");
            printf("|\n");
            break;         
        default:
            break;
    }
}

/**
 * @brief 从字典中随机选择并返回一个单词
 * 
 * @return 随机单词
 */
char* random_word_picker(){
    // 生成一个随机的英文单词
    char* word = malloc(sizeof(char) * 100);  // 分配内存以存储单词
    FILE* fp = fopen("dictionary.txt", "r");  // 打开包含英文单词的文件
    srand(time(NULL));  // 使用当前时间作为随机数种子
    if (fp == NULL){  // 如果文件打开失败
        printf("Error opening dictionary.txt\n");  // 输出错误信息
        exit(1);  // 退出程序
    }
    int random_number = rand() % MAX_WORDS;  // 生成一个 0 到 MAX_WORDS-1 之间的随机数
    for (int j = 0; j < random_number; j++){  // 循环直到达到随机数指定的位置
        fscanf(fp, "%s", word);  // 从文件中读取单词
    }
    fclose(fp);  // 关闭文件
    return word;  // 返回随机单词
# 主函数，程序的入口
void main(void){
    # 为单词分配内存空间
    char* word = malloc(sizeof(char) * 100);
    # 从随机单词选择器中获取单词
    word = random_word_picker();
    # 为隐藏单词分配内存空间
    char* hidden_word = malloc(sizeof(char) * 100);
    # 将隐藏单词初始化为下划线
    for (int i = 0; i < strlen(word); i++){
        hidden_word[i] = '_';
    }
    hidden_word[strlen(word)] = '\0';
    # 游戏阶段
    int stage = 0;
    # 错误猜测次数
    int wrong_guesses = 0;
    # 正确猜测次数
    int correct_guesses = 0;
    # 猜测的单词
    char* guess = malloc(sizeof(char) * 100);
    # 当错误猜测次数小于6且正确猜测次数小于单词长度时进行循环
    while (wrong_guesses < 6 && correct_guesses < strlen(word)){
        CLEAR;  # 清空屏幕
        print_hangman(stage);  # 打印当前 hangman 的状态
        printf("%s\n", hidden_word);  # 打印隐藏单词
        printf("Enter a guess: ");  # 提示输入猜测
        scanf("%s", guess);  # 读取猜测
        # 遍历单词
        for (int i = 0; i < strlen(word); i++){
            # 如果猜测与单词相同，则正确猜测次数等于单词长度
            if (strcmp(guess,word) == 0){
                correct_guesses = strlen(word);
            }
            # 如果猜测与单词中的某个字符相同，则将隐藏单词中对应位置替换为猜测字符，并增加正确猜测次数
            else if (guess[0] == word[i]){
                hidden_word[i] = guess[0];
                correct_guesses++;
            }
        }
        # 如果猜测字符不在单词中，则增加错误猜测次数
        if (strchr(word, guess[0]) == NULL){
            wrong_guesses++;
        }
        stage = wrong_guesses;  # 更新 hangman 的状态
    }
    # 如果错误猜测次数等于6，则游戏失败，显示正确单词
    if (wrong_guesses == 6){
        printf("You lose! The word was %s\n", word);
    }
    # 否则游戏胜利
    else {
        printf("You win!\n");
    }
}
```