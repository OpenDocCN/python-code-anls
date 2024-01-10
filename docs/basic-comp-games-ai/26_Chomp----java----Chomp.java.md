# `basic-computer-games\26_Chomp\java\Chomp.java`

```
// 导入 Scanner 类
import java.util.Scanner;

// 定义 Chomp 类
public class Chomp{
    // 定义实例变量
    int rows;
    int cols;
    int numberOfPlayers;
    int []board;
    Scanner scanner;

    // Chomp 类的构造函数
    Chomp(){
        // 输出游戏标题和介绍信息
        System.out.println("\t\t\t\tCHOMP");
        System.out.println("\t\tCREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
        System.out.println("THIS IS THE GAME OF CHOMP (SCIENTIFIC AMERICAN, JAN 1973)");
        System.out.print("Do you want the rules (1=Yes, 0=No!)  ");

        // 创建 Scanner 对象
        scanner = new Scanner(System.in);
        // 读取用户输入的选择
        int choice = scanner.nextInt();
        // 如果用户选择查看规则
        if(choice != 0){
            // 输出游戏规则
            System.out.println("Chomp is for 1 or more players (Humans only).\n");
            System.out.println("Here's how a board looks (This one is 5 by 7):");
            System.out.println("\t1 2 3 4 5 6 7");
            System.out.println(" 1     P * * * * * *\n 2     * * * * * * *\n 3     * * * * * * *\n 4     * * * * * * *\n 5     * * * * * * *");
            System.out.println("\nThe board is a big cookie - R rows high and C columns \nwide. You input R and C at the start. In the upper left\ncorner of the cookie is a poison square (P). The one who\nchomps the poison square loses. To take a chomp, type the\nrow and column of one of the squares on the cookie.\nAll of the squares below and to the right of that square\n(Including that square, too) disappear -- CHOMP!!\nNo fair chomping squares that have already been chomped,\nor that are outside the original dimensions of the cookie.\n");
            System.out.println("Here we go...\n");
        }
        // 开始游戏
        startGame();
    }
}
    // 启动游戏
    private void startGame(){
        // 提示输入玩家数量
        System.out.print("How many players ");
        // 读取输入的玩家数量
        numberOfPlayers = scanner.nextInt();
        // 确保玩家数量大于等于2
        while(numberOfPlayers < 2){
            // 提示重新输入玩家数量
            System.out.print("How many players ");
            numberOfPlayers = scanner.nextInt();
        }
        // 提示输入行数
        System.out.print("How many rows ");
        // 读取输入的行数
        rows = scanner.nextInt();
        // 确保行数在1到9之间
        while(rows<=0 || rows >9){
            // 如果行数小于等于0，提示最少需要1行
            if(rows <= 0){
                System.out.println("Minimun 1 row is required !!");
            }
            // 如果行数大于9，提示行数过多
            else{
                System.out.println("Too many rows(9 is maximum). ");
            }
            // 提示重新输入行数
            System.out.print("How many rows ");
            rows = scanner.nextInt();
        }
        // 提示输入列数
        System.out.print("How many columns ");
        // 读取输入的列数
        cols = scanner.nextInt();
        // 确保列数在1到9之间
        while(cols<=0 || cols >9){
            // 如果列数小于等于0，提示最少需要1列
            if(cols <= 0){
                System.out.println("Minimun 1 column is required !!");
            }
            // 如果列数大于9，提示列数过多
            else{
                System.out.println("Too many columns(9 is maximum). ");
            }
            // 提示重新输入列数
            System.out.print("How many columns ");
            cols = scanner.nextInt();
        }
        // 创建一个二维数组作为游戏棋盘
        board = new int[rows];
        // 初始化棋盘每行的列数
        for(int i=0;i<rows;i++){
            board[i]=cols;
        }
        // 打印游戏棋盘
        printBoard();
        // 读取换行符
        scanner.nextLine();
        // 开始游戏
        move(0);
    }
    // 打印游戏板的方法
    private void printBoard(){
        // 打印列号
        System.out.print("        ");
        for(int i=0;i<cols;i++){
            System.out.print(i+1);
            System.out.print(" ");
        }
        // 打印行号和每行的内容
        for(int i=0;i<rows;i++){
            System.out.print("\n ");
            System.out.print(i+1);
            System.out.print("      ");
            // 打印每行的内容
            for(int j=0;j<board[i];j++){
                // 如果是第一行第一列，则打印 "P "
                if(i == 0 && j == 0){
                    System.out.print("P ");
                }
                // 否则打印 "* "
                else{
                    System.out.print("* ");
                }
            }
        }
        // 换行
        System.out.println("");
    }

    // 主方法
    public static void main(String []args){
        // 创建 Chomp 对象
        new Chomp();
    }
# 闭合前面的函数定义
```