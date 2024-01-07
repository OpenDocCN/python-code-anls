# `basic-computer-games\00_Alternate_Languages\44_Hangman\C\main.c`

```

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define MAX_WORDS 100

//check if windows or linux for the clear screen
#ifdef _WIN32
#define CLEAR "cls"
#else
#define CLEAR "clear"
#endif

/**
 * @brief Prints the stage of the hangman based on the number of wrong guesses.
 * 
 * @param stage Hangman stage.
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
 * @brief Picks and return a random word from the dictionary.
 * 
 * @return Random word 
 */
char* random_word_picker(){
    //generate a random english word
    char* word = malloc(sizeof(char) * 100); // 分配内存以存储单词
    FILE* fp = fopen("dictionary.txt", "r"); // 打开包含单词的文件
    srand(time(NULL)); // 使用当前时间作为随机数种子
    if (fp == NULL){
        printf("Error opening dictionary.txt\n"); // 如果无法打开文件，则打印错误消息
        exit(1); // 退出程序
    }
    int random_number = rand() % MAX_WORDS; // 生成一个随机数
    for (int j = 0; j < random_number; j++){
        fscanf(fp, "%s", word); // 从文件中读取单词
    }
    fclose(fp); // 关闭文件
    return word; // 返回随机选取的单词
}




void main(void){
    char* word = malloc(sizeof(char) * 100); // 分配内存以存储单词
    word = random_word_picker(); // 从字典中随机选择一个单词
    char* hidden_word = malloc(sizeof(char) * 100); // 分配内存以存储隐藏的单词
    for (int i = 0; i < strlen(word); i++){
        hidden_word[i] = '_'; // 将隐藏的单词初始化为下划线
    }
    hidden_word[strlen(word)] = '\0'; // 添加字符串结束符
    int stage = 0; // 初始化游戏阶段
    int wrong_guesses = 0; // 初始化错误猜测次数
    int correct_guesses = 0; // 初始化正确猜测次数
    char* guess = malloc(sizeof(char) * 100); // 分配内存以存储猜测
    while (wrong_guesses < 6 && correct_guesses < strlen(word)){
        CLEAR; // 清空屏幕
        print_hangman(stage); // 打印当前的 hangman 图案
        printf("%s\n", hidden_word); // 打印隐藏的单词
        printf("Enter a guess: "); // 提示用户输入猜测
        scanf("%s", guess); // 读取用户输入的猜测
        for (int i = 0; i < strlen(word); i++){
            if (strcmp(guess,word) == 0){ // 如果猜测与单词相同
                correct_guesses = strlen(word); // 设置正确猜测次数为单词长度
            }
            else if (guess[0] == word[i]){ // 如果猜测的第一个字符与单词中的字符相同
                hidden_word[i] = guess[0]; // 更新隐藏的单词
                correct_guesses++; // 增加正确猜测次数
            }
        }
        if (strchr(word, guess[0]) == NULL){ // 如果猜测的字符不在单词中
            wrong_guesses++; // 增加错误猜测次数
        }
        stage = wrong_guesses; // 更新游戏阶段
    }
    if (wrong_guesses == 6){ // 如果错误猜测次数达到6次
        printf("You lose! The word was %s\n", word); // 打印失败消息和单词
    }
    else {
        printf("You win!\n"); // 打印胜利消息
    }
}

```