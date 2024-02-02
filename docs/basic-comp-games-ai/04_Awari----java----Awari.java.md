# `basic-computer-games\04_Awari\java\Awari.java`

```py
import java.util.Scanner;  // 导入 Scanner 类，用于接收用户输入
import java.util.Random;   // 导入 Random 类，用于生成随机数

public class Awari{  // 定义 Awari 类
    int []board;  // 定义整型数组 board，用于存储棋盘状态
    private final int playerPits;  // 定义玩家的小坑的位置
    private final int computerPits;  // 定义计算机的小坑的位置
    private final int playerHome;  // 定义玩家的家的位置
    private final int computerHome;  // 定义计算机的家的位置
    Scanner input;  // 定义 Scanner 对象 input，用于接收用户输入
    int sumPlayer;  // 定义整型变量 sumPlayer，用于存储玩家的总数
    int sumComputer;  // 定义整型变量 sumComputer，用于存储计算机的总数
    Awari(){  // 定义构造函数
        input = new Scanner(System.in);  // 初始化 Scanner 对象 input
        playerPits = 0;  // 初始化玩家的小坑的位置
        computerPits = 7;  // 初始化计算机的小坑的位置
        playerHome = 6;  // 初始化玩家的家的位置
        computerHome = 13;  // 初始化计算机的家的位置
        sumPlayer = 18;  // 初始化玩家的总数
        sumComputer = 18;  // 初始化计算机的总数
        board = new int [14];  // 初始化整型数组 board，长度为 14
        for (int i=0;i<6;i++){  // 循环初始化玩家和计算机的小坑的初始状态
            board[playerPits+i]=3;  // 玩家小坑初始状态为 3
            board[computerPits+i]=3;  // 计算机小坑初始状态为 3
        }
        System.out.println("         AWARI");  // 输出游戏标题
        System.out.println("CREATIVE COMPUTING MORRISTOWN, NEW JERSEY");  // 输出游戏信息
        printBoard();  // 调用打印棋盘的方法
        playerMove(true);  // 调用玩家移动的方法，传入参数 true
    }

    private void printBoard(){  // 定义打印棋盘的方法
        System.out.print("\n    ");  // 输出换行和空格
        for (int i=0;i<6;i++){  // 循环打印计算机小坑的状态
            System.out.print(String.format("%2d",board[12-i]));  // 格式化输出计算机小坑的状态
            System.out.print("  ");  // 输出空格
        }
        System.out.println("");  // 输出换行
        System.out.print(String.format("%2d",board[computerHome]));  // 输出计算机家的状态
        System.out.print("                          ");  // 输出空格
        System.out.println(String.format("%2d",board[playerHome]));  // 输出玩家家的状态
        System.out.print("    ");  // 输出空格
        for(int i=0;i<6;i++){  // 循环打印玩家小坑的状态
            System.out.print(String.format("%2d",board[playerPits+i]));  // 格式化输出玩家小坑的状态
                        System.out.print("  ");  // 输出空格
        }
        System.out.println("");  // 输出换行
    }
}
    // 玩家移动方法，根据参数确定是否是玩家的回合
    private void playerMove(boolean val){
        // 打印电脑和玩家的总数
        System.out.println("\nComputerSum PlayerSum"+sumComputer+" "+sumPlayer);
        // 根据参数确定输出提示信息
        if(val == true)
            System.out.print("YOUR MOVE? ");
        else
            System.out.print("AGAIN? ");
        // 获取玩家输入的移动位置
        int move =  input.nextInt();
        // 当移动位置无效时，要求玩家重新输入
        while(move<1||move>6||board[move-1]==0){
            System.out.print("INVALID MOVE!!! TRY AGAIN  ");
            move = input.nextInt();
        }
        // 获取移动位置的种子数，将该位置的种子数置为0，更新玩家总数
        int seeds = board[move-1];
        board[move-1] = 0;
        sumPlayer -= seeds;
        // 分发种子，获取最后一个位置
        int last_pos = distribute(seeds,move);
        // 如果最后一个位置是玩家的家，打印棋盘，检查游戏是否结束，如果结束则退出，否则继续玩家回合
        if(last_pos == playerHome){
            printBoard();
            if(isGameOver(true)){
                System.exit(0);
            }
            playerMove(false);
        }
        // 如果最后一个位置是非玩家家且只有一个种子，执行特殊操作，打印棋盘，检查游戏是否结束，如果结束则退出，否则继续电脑回合
        else if(board[last_pos] == 1&&last_pos != computerHome){
            int opp = calculateOpposite(last_pos);
            if(last_pos<6){
                sumPlayer+=board[opp];
                sumComputer-=board[opp];
            }
            else{
                sumComputer+=board[opp];
                sumPlayer-=board[opp];
            }
            board[last_pos]+=board[opp];
            board[opp] = 0;
            printBoard();
            if(isGameOver(false)){
                System.exit(0);
            }
            computerMove(true);
        }
        // 否则，打印棋盘，检查游戏是否结束，如果结束则退出，否则继续电脑回合
        else{
            printBoard();
            if(isGameOver(false)){
                System.exit(0);
            }
            computerMove(true);
        }
    }

    // 分发种子的方法，返回最后一个位置
    private int distribute(int seeds, int pos){
        while(seeds!=0){
            if(pos==14)
                pos=0;
            if(pos<6)
                sumPlayer++;
            else if(pos>6&&pos<13)
                sumComputer++;
            board[pos]++;
            pos++;
            seeds--;
        }
        return pos-1;
    }

    // 计算对面位置的方法
    private int calculateOpposite(int pos){
        return 12-pos;
    }
    // 检查游戏是否结束，如果show为true，则显示游戏板
    private boolean isGameOver(boolean show){
        // 如果玩家或电脑的棋子数为0，则游戏结束
        if(sumPlayer == 0 || sumComputer == 0){
            // 如果需要显示游戏板，则打印游戏板
            if(show)
                printBoard();
            // 打印游戏结束信息
            System.out.println("GAME OVER");
            // 判断玩家和电脑的得分情况，并打印对应的结果
            if(board[playerHome]>board[computerHome]){
                System.out.println(String.format("YOU WIN BY %d POINTS",board[playerHome]-board[computerHome]));
            }
            else if(board[playerHome]<board[computerHome]){
                System.out.println(String.format("YOU LOSE BY %d POINTS",board[computerHome]-board[playerHome]));
            }
            else{
                System.out.println("DRAW");
            }
            // 返回true表示游戏结束
            return true;
        }
        // 返回false表示游戏未结束
        return false;
    }
# 闭合前面的函数定义
```