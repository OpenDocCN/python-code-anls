# `d:/src/tocomm/basic-computer-games\00_Alternate_Languages\30_Cube\C\cube.c`

```
#include <stdio.h>  // 包含标准输入输出库
#include <stdlib.h>  // 包含标准库
#include <time.h>    // 包含时间库

// 根据操作系统判断清空屏幕的命令
#ifdef _WIN32
#define CLEAR "cls"
#else
#define CLEAR "clear"
#endif

// 定义坐标结构体
typedef struct{
    int x;
    int y;
    int z;
}coords;

// 显示游戏说明
void instuctions(){
    printf("\nThis is a game in which you will be playing against the\n");
    printf("random decisions of the computer. The field of play is a\n");
    # 输出游戏规则和提示信息
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
    printf("Good luck!\n")
}

def game(money):
    # 初始化玩家当前位置和上一次位置
    coords player, playerold
    # 初始化地雷的位置数组
    mines[5]
    # 定义赌注和账户余额变量，将 money 的值赋给 account
    int wager, account = money;
    # 定义选择变量
    char choice;
    # 如果账户余额为 0，则打印消息并退出程序
    if(money == 0){
        printf("You have no money left. See ya next time.\n");
        exit(0);
    }
    # 初始化玩家的 x、y、z 坐标
    player.x = 1;
    player.y = 1;
    player.z = 1;
    
    # 打印账户余额
    printf("You have $%d in your account.\n", account);
    # 提示玩家输入赌注金额，并将输入的值赋给 wager
    printf("How much do you want to wager? ");
    scanf("%d", &wager);
    # 当赌注大于账户余额时，循环提示玩家重新输入赌注金额
    while(wager > account){
        system(CLEAR);
        printf("You do not have that much money in your account.\n");
        printf("How much do you want to wager? ");
        scanf("%d", &wager);
    }
    # 生成随机种子
    srand(time(NULL));
    # 生成5个地雷的随机坐标，范围为1到3
    for(int i=0;i<5;i++){
        mines[i].x = rand()%3+1;
        mines[i].y = rand()%3+1;
        mines[i].z = rand()%3+1;
        # 如果地雷坐标为(3,3,3)，重新生成坐标
        if(mines[i].x == 3 && mines[i].y == 3 && mines[i].z == 3){
            i--;
        }
    }
    # 当玩家不在坐标(3,3,3)时，循环执行以下操作
    while(player.x != 3 || player.y != 3 || player.z != 3){
        # 打印玩家当前位置坐标
        printf("You are at location %d.%d.%d\n",player.x,player.y,player.z);
        # 如果玩家在坐标(1,1,1)，提示输入新坐标
        if(player.x == 1 && player.y == 1 && player.z == 1)
        printf("Enter new location(use commas like 1,1,2 or else the program will break...): ");
        else printf("Enter new location: ");
        # 保存玩家当前位置坐标
        playerold.x = player.x;
        playerold.y = player.y;
        playerold.z = player.z;
        # 输入新的玩家位置坐标
        scanf("%d,%d,%d",&player.x,&player.y,&player.z);
        # 如果玩家移动超过1个单位，清空屏幕并提示非法移动
        if(((player.x + player.y + player.z) > (playerold.x + playerold.y + playerold.z + 1)) || ((player.x + player.y + player.z) < (playerold.x + playerold.y + playerold.z -1))){
            system(CLEAR);
            printf("Illegal move!\n");
            printf("You lose $%d.\n",wager);  // 打印玩家失去的赌注金额
            game(account -= wager);  // 调用游戏函数，更新玩家账户余额
            break;  // 跳出循环
        }
        if(player.x < 1 || player.x > 3 || player.y < 1 || player.y > 3 || player.z < 1 || player.z > 3){
            system(CLEAR);  // 清空控制台
            printf("Illegal move. You lose!\n");  // 打印玩家非法移动导致的失败信息
            game(account -= wager);  // 调用游戏函数，更新玩家账户余额
            break;  // 跳出循环
        }
        for(int i=0;i<5;i++){
            if(player.x == mines[i].x && player.y == mines[i].y && player.z == mines[i].z){
                system(CLEAR);  // 清空控制台
                printf("You hit a mine!\n");  // 打印玩家踩到地雷的信息
                printf("You lose $%d.\n",wager);  // 打印玩家失去的赌注金额
                game(account -= wager);  // 调用游戏函数，更新玩家账户余额
                exit(0);  // 退出程序
            }
        }
        if(account == 0){  // 如果玩家账户余额为0
            system(CLEAR);  # 清空屏幕
            printf("You have no money left!\n");  # 打印玩家没有钱了的消息
            printf("Game over!\n");  # 打印游戏结束的消息
            exit(0);  # 退出程序
        }
    }
    if(player.x == 3 && player.y == 3 && player.z == 3){  # 如果玩家的位置是 (3, 3, 3)
        system(CLEAR);  # 清空屏幕
        printf("You made it to the end. You win!\n");  # 打印玩家赢得游戏的消息
        game(account += wager);  # 调用 game 函数并更新玩家账户余额
        exit(0);  # 退出程序
    }
}

void init(){
    int account = 500;  # 初始化玩家账户余额为 500
    char choice;  # 定义一个字符变量 choice

    printf("Welcome to the game of Cube!\n");  # 打印欢迎消息
    printf("wanna see the instructions? (y/n): ");  # 打印是否要查看游戏说明的提示
    # 从用户输入中读取一个字符，存储到变量 choice 中
    scanf("%c",&choice);
    # 如果 choice 等于 'y'，则执行下面的代码块
    if(choice == 'y'){
        # 清空屏幕
        system(CLEAR);
        # 调用 instructions 函数，显示游戏说明
        instuctions();
    }
    # 如果 choice 等于 'n'，则执行下面的代码块
    else if (choice == 'n'){
        # 清空屏幕
        system(CLEAR);
        # 打印消息 "Ok, let's play!"
        printf("Ok, let's play!\n");
    }
    # 如果 choice 不是 'y' 也不是 'n'，则执行下面的代码块
    else{
        # 清空屏幕
        system(CLEAR);
        # 打印消息 "Invalid choice. Try again..."
        printf("Invalid choice. Try again...\n");
        # 调用 init 函数，重新初始化游戏
        init();
        # 退出程序
        exit(0);
    }
    # 调用 game 函数，开始游戏
    game(account);
    # 退出程序
    exit(0);
}

# 主函数
void main(){
```
在这段代码中，我们使用注释解释了每个语句的作用，包括读取用户输入、根据用户选择执行不同的操作、调用函数以及退出程序。
    # 调用初始化函数
    init();
}
```