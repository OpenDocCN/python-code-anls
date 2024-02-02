# `basic-computer-games\00_Alternate_Languages\30_Cube\C\cube.c`

```py
#include <stdio.h>  // 包含标准输入输出头文件
#include <stdlib.h>  // 包含标准库头文件
#include <time.h>  // 包含时间头文件

// 根据操作系统定义清屏命令
#ifdef _WIN32
#define CLEAR "cls"
#else
#define CLEAR "clear"
#endif

typedef struct{  // 定义坐标结构体
    int x;
    int y;
    int z;
}coords;

void instuctions(){  // 定义游戏说明函数
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

void game(int money){  // 定义游戏函数，传入初始金额
    coords player,playerold,mines[5];  // 定义玩家当前位置、上一次位置和地雷位置数组
    int wager,account = money;  // 定义赌注和账户余额
    char choice;  // 定义选择变量
    if(money == 0){  // 如果没有余额，退出游戏
        printf("You have no money left. See ya next time.\n");
        exit(0);
    }
    player.x = 1;  // 玩家初始位置
    player.y = 1;
    player.z = 1;
    
    printf("You have $%d in your account.\n",account);  // 显示账户余额
    printf("How much do you want to wager? ");  // 询问赌注
    scanf("%d",&wager);  // 输入赌注
    # 当赌注大于账户余额时，进入循环
    while(wager > account){
        # 清空屏幕
        system(CLEAR);
        # 提示用户账户余额不足
        printf("You do not have that much money in your account.\n");
        # 提示用户输入赌注金额
        printf("How much do you want to wager? ");
        # 读取用户输入的赌注金额
        scanf("%d",&wager);
    }
    # 生成随机种子
    srand(time(NULL));
    # 循环5次，生成5个地雷的坐标
    for(int i=0;i<5;i++){
        # 生成地雷的 x 坐标
        mines[i].x = rand()%3+1;
        # 生成地雷的 y 坐标
        mines[i].y = rand()%3+1;
        # 生成地雷的 z 坐标
        mines[i].z = rand()%3+1;
        # 如果地雷的坐标为 (3,3,3)，则重新生成地雷坐标
        if(mines[i].x == 3 && mines[i].y == 3 && mines[i].z == 3){
            # 重新生成地雷坐标
            i--;
        }
    }
    # 当玩家的位置不是 (3, 3, 3) 时，执行循环
    while(player.x != 3 || player.y != 3 || player.z != 3){
        # 打印玩家当前位置
        printf("You are at location %d.%d.%d\n",player.x,player.y,player.z);
        # 如果玩家位置是 (1, 1, 1)，提示输入新位置
        if(player.x == 1 && player.y == 1 && player.z == 1)
        printf("Enter new location(use commas like 1,1,2 or else the program will break...): ");
        else printf("Enter new location: ");
        # 保存玩家当前位置
        playerold.x = player.x;
        playerold.y = player.y;
        playerold.z = player.z;
        # 输入新的位置
        scanf("%d,%d,%d",&player.x,&player.y,&player.z);
        # 检查新位置是否合法，如果不合法则结束游戏
        if(((player.x + player.y + player.z) > (playerold.x + playerold.y + playerold.z + 1)) || ((player.x + player.y + player.z) < (playerold.x + playerold.y + playerold.z -1))){
            system(CLEAR);
            printf("Illegal move!\n");
            printf("You lose $%d.\n",wager);
            game(account -= wager);
            break;
        }
        # 检查新位置是否在合法范围内，如果不在范围内则结束游戏
        if(player.x < 1 || player.x > 3 || player.y < 1 || player.y > 3 || player.z < 1 || player.z > 3){
            system(CLEAR);
            printf("Illegal move. You lose!\n");
            game(account -= wager);
            break;
        }
        # 检查新位置是否是地雷的位置，如果是则结束游戏
        for(int i=0;i<5;i++){
            if(player.x == mines[i].x && player.y == mines[i].y && player.z == mines[i].z){
                system(CLEAR);
                printf("You hit a mine!\n");
                printf("You lose $%d.\n",wager);
                game(account -= wager);
                exit(0);
            }
        }
        # 如果账户余额为 0，则结束游戏
        if(account == 0){
            system(CLEAR);
            printf("You have no money left!\n");
            printf("Game over!\n");
            exit(0);
        }
    }
    # 如果玩家位置是 (3, 3, 3)，则游戏胜利
    if(player.x == 3 && player.y == 3 && player.z == 3){
        system(CLEAR);
        printf("You made it to the end. You win!\n");
        game(account += wager);
        exit(0);
    }
# 初始化函数，设置初始账户金额和用户选择
void init(){
    # 初始化账户金额为500
    int account = 500;
    # 用户选择
    char choice;

    # 打印欢迎语
    printf("Welcome to the game of Cube!\n");
    # 打印是否查看游戏说明
    printf("wanna see the instructions? (y/n): ");
    # 读取用户输入
    scanf("%c",&choice);
    # 如果用户选择查看说明
    if(choice == 'y'){
        # 清空屏幕
        system(CLEAR);
        # 显示游戏说明
        instuctions();
    }
    # 如果用户选择不查看说明
    else if (choice == 'n'){
        # 清空屏幕
        system(CLEAR);
        # 打印开始游戏提示
        printf("Ok, let's play!\n");
    }
    # 如果用户输入无效
    else{
        # 清空屏幕
        system(CLEAR);
        # 提示用户输入无效，重新初始化
        printf("Invalid choice. Try again...\n");
        init();
        # 退出程序
        exit(0);
    }
    # 开始游戏
    game(account);
    # 退出程序
    exit(0);
}

# 主函数
void main(){
    # 调用初始化函数
    init();
}
```