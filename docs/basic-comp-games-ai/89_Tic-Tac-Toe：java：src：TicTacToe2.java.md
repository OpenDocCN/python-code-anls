# `89_Tic-Tac-Toe\java\src\TicTacToe2.java`

```
import java.util.Scanner;  // 导入 Scanner 类，用于接收用户输入
import java.util.Random;   // 导入 Random 类，用于生成随机数

/**
 * @author Ollie Hensman-Crook  // 作者注释
 */
public class TicTacToe2 {  // 定义名为 TicTacToe2 的类
    public static void main(String[] args) {  // 主函数
        Board gameBoard = new Board();  // 创建名为 gameBoard 的 Board 对象
        Random compChoice = new Random();  // 创建名为 compChoice 的 Random 对象
        char yourChar;  // 定义字符型变量 yourChar
        char compChar;  // 定义字符型变量 compChar
        Scanner in = new Scanner(System.in);  // 创建名为 in 的 Scanner 对象，用于接收用户输入

        System.out.println("              TIC-TAC-TOE");  // 打印提示信息
        System.out.println("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");  // 打印提示信息
        System.out.println("\nTHE BOARD IS NUMBERED: ");  // 打印提示信息
        System.out.println(" 1  2  3\n 4  5  6\n 7  8  9\n");  // 打印提示信息

        while (true) {  // 进入循环
// 询问玩家想要选择 'X' 还是 'O'，如果输入有效则设置他们的游戏棋子
System.out.println("DO YOU WANT 'X' OR 'O'");
while (true) {
    try {
        char input;
        input = in.next().charAt(0);

        // 如果玩家选择 'X' 或者 'x'，则设置玩家棋子为 'X'，计算机棋子为 'O'
        if (input == 'X' || input == 'x') {
            yourChar = 'X';
            compChar = 'O';
            break;
        } 
        // 如果玩家选择 'O' 或者 'o'，则设置玩家棋子为 'O'，计算机棋子为 'X'
        else if (input == 'O' || input == 'o') {
            yourChar = 'O';
            compChar = 'X';
            break;
        } 
        // 如果玩家输入既不是 'X' 也不是 'O'，则提示玩家重新输入
        else {
            System.out.println("THATS NOT 'X' OR 'O', TRY AGAIN");
            in.nextLine();
        }
                } catch (Exception e) {  # 捕获可能发生的异常
                    System.out.println("THATS NOT 'X' OR 'O', TRY AGAIN");  # 打印错误提示信息
                    in.nextLine();  # 读取下一行输入
                }
            }

            while (true) {  # 进入无限循环
                System.out.println("WHERE DO YOU MOVE");  # 打印提示信息

                // check the user can move where they want to and if so move them there
                while (true) {  # 进入内部无限循环
                    int input;  # 声明整型变量input
                    try {  # 尝试执行以下代码
                        input = in.nextInt();  # 从输入中读取整数
                        if (gameBoard.getBoardValue(input) == ' ') {  # 检查用户输入的位置是否为空
                            gameBoard.setArr(input, yourChar);  # 如果是空的，将用户的标记放置在该位置
                            break;  # 退出内部循环
                        } else {  # 如果位置不为空
                            System.out.println("INVALID INPUT, TRY AGAIN");  # 打印错误提示信息
                }
                in.nextLine();  # 读取用户输入的下一行内容

                } catch (Exception e) {  # 捕获可能发生的异常
                    System.out.println("INVALID INPUT, TRY AGAIN");  # 打印错误提示信息
                    in.nextLine();  # 读取用户输入的下一行内容
                }
            }

            gameBoard.printBoard();  # 调用游戏板打印方法，打印游戏板
            System.out.println("THE COMPUTER MOVES TO");  # 打印电脑移动的提示信息

            while (true) {  # 进入无限循环
                int position = 1 + compChoice.nextInt(9);  # 生成1到9之间的随机整数
                if (gameBoard.getBoardValue(position) == ' ') {  # 判断游戏板上指定位置是否为空
                    gameBoard.setArr(position, compChar);  # 在游戏板上设置电脑的棋子
                    break;  # 跳出循环
                }
            }

            gameBoard.printBoard();  # 调用游戏板打印方法，打印游戏板
                // 如果玩家获胜，打印出玩家赢了，并询问是否要再玩一次
                if (gameBoard.checkWin(yourChar)) {
                    System.out.println("YOU WIN, PLAY AGAIN? (Y/N)");
                    gameBoard.clear(); // 清空游戏板
                    while (true) {
                        try {
                            char input;
                            input = in.next().charAt(0); // 读取用户输入的字符

                            if (input == 'Y' || input == 'y') { // 如果输入是Y或y，结束循环
                                break;
                            } else if (input == 'N' || input == 'n') { // 如果输入是N或n，退出程序
                                System.exit(0);
                            } else { // 如果输入既不是Y/y也不是N/n，提示用户重新输入
                                System.out.println("THATS NOT 'Y' OR 'N', TRY AGAIN");
                                in.nextLine(); // 清空输入缓冲区
                            }
                        } catch (Exception e) {  # 捕获异常
                            System.out.println("THATS NOT 'Y' OR 'N', TRY AGAIN");  # 打印错误提示信息
                            in.nextLine();  # 读取下一行输入
                        }
                    }
                    break;  # 跳出循环
                } else if (gameBoard.checkWin(compChar)) {  # 如果游戏板上电脑赢了
                    System.out.println("YOU LOSE, PLAY AGAIN? (Y/N)");  # 打印提示信息
                    gameBoard.clear();  # 清空游戏板
                    while (true) {  # 进入循环
                        try:  # 尝试执行以下代码
                            char input;  # 定义字符变量
                            input = in.next().charAt(0);  # 读取输入的第一个字符

                            if (input == 'Y' || input == 'y') {  # 如果输入是 'Y' 或 'y'
                                break;  # 跳出循环
                            } else if (input == 'N' || input == 'n') {  # 如果输入是 'N' 或 'n'
                                System.exit(0);  # 退出程序
                            } else {  # 如果输入既不是 'Y' 也不是 'N'
                                System.out.println("THATS NOT 'Y' OR 'N', TRY AGAIN");  # 打印错误提示信息
                    } else {
                        System.out.println("THATS NOT 'Y' OR 'N', TRY AGAIN");
                        in.nextLine();
                    }
                } catch (Exception e) {
                    System.out.println("THATS NOT 'Y' OR 'N', TRY AGAIN");
                    in.nextLine();
                }
            }
        }
        break;
    } else if (gameBoard.checkDraw()) {
        System.out.println("DRAW, PLAY AGAIN? (Y/N)");
        gameBoard.clear();
        while (true) {
            try {
                char input;
                input = in.next().charAt(0);

                if (input == 'Y' || input == 'y') {
                    break;
                } else if (input == 'N' || input == 'n') {
                    // 如果输入为 'N' 或 'n'，结束游戏
                    System.out.println("THANKS FOR PLAYING!");
                    System.exit(0);
                } else {
                    // 如果输入既不是 'Y' 也不是 'N'，提示用户重新输入
                    System.out.println("THATS NOT 'Y' OR 'N', TRY AGAIN");
                    in.nextLine();
                }
            } catch (Exception e) {
                // 捕获异常，提示用户重新输入
                System.out.println("THATS NOT 'Y' OR 'N', TRY AGAIN");
                in.nextLine();
            }
        }
    }
}
# 退出程序
System.exit(0);
# 如果用户输入不是 'Y' 或 'N'，则提示用户重新输入
} else {
    System.out.println("THATS NOT 'Y' OR 'N', TRY AGAIN");
    in.nextLine();
}
# 捕获异常并提示用户重新输入
} catch (Exception e) {
    System.out.println("THATS NOT 'Y' OR 'N', TRY AGAIN");
    in.nextLine();
}
```