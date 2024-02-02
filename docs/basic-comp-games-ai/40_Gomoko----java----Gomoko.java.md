# `basic-computer-games\40_Gomoko\java\Gomoko.java`

```py
import java.util.Arrays;  // 导入 Arrays 类
import java.util.InputMismatchException;  // 导入 InputMismatchException 类
import java.util.Scanner;  // 导入 Scanner 类

/**
 * GOMOKO
 * <p>
 * Converted from BASIC to Java by Aldrin Misquitta (@aldrinm)
 */
public class Gomoko {

    private static final int MIN_BOARD_SIZE = 7;  // 定义最小棋盘大小为 7
    private static final int MAX_BOARD_SIZE = 19;  // 定义最大棋盘大小为 19

    }

    //*** COMPUTER TRIES AN INTELLIGENT MOVE ***
    private static Move getComputerMove(Move playerMove, int[][] board, int boardSize) {
        for (int e = -1; e <= 1; e++) {  // 循环遍历 e 取值为 -1, 0, 1
            for (int f = -1; f <= 1; f++) {  // 循环遍历 f 取值为 -1, 0, 1
                if ((e + f - e * f) != 0) {  // 判断条件是否成立
                    var x = playerMove.i + f;  // 计算 x 坐标
                    var y = playerMove.j + f;  // 计算 y 坐标
                    final Move newMove = new Move(x, y);  // 创建新的移动对象
                    if (isLegalMove(newMove, boardSize)) {  // 判断新移动是否合法
                        if (board[newMove.i - 1][newMove.j - 1] != 0) {  // 判断新移动位置是否为空
                            newMove.i = newMove.i - e;  // 更新新移动的 x 坐标
                            newMove.i = newMove.j - f;  // 更新新移动的 y 坐标
                            if (!isLegalMove(newMove, boardSize)) {  // 判断更新后的移动是否合法
                                return null;  // 返回空值
                            } else {
                                if (board[newMove.i - 1][newMove.j - 1] == 0) {  // 判断更新后的移动位置是否为空
                                    return newMove;  // 返回新的移动对象
                                }
                            }
                        }
                    }
                }
            }
        }
        return null;  // 返回空值
    }

    private static void printBoard(int[][] board) {
        for (int[] ints : board) {  // 遍历棋盘数组
            for (int cell : ints) {  // 遍历每一行的元素
                System.out.printf(" %s", cell);  // 打印每个元素
            }
            System.out.println();  // 换行
        }
    }

    //*** COMPUTER TRIES A RANDOM MOVE ***
    // 从给定的棋盘中获取一个随机的合法移动
    private static Move getRandomMove(int[][] board, int boardSize) {
        // 初始化合法移动标志为假
        boolean legalMove = false;
        // 初始化随机移动为null
        Move randomMove = null;
        // 当没有找到合法移动时循环
        while (!legalMove) {
            // 获取一个随机移动
            randomMove = randomMove(boardSize);
            // 判断随机移动是否合法并且目标位置为空
            legalMove = isLegalMove(randomMove, boardSize) && board[randomMove.i - 1][randomMove.j - 1] == 0;
        }
        // 返回随机移动
        return randomMove;
    }

    // 生成一个随机移动
    private static Move randomMove(int boardSize) {
        // 生成随机的x坐标
        int x = (int) (boardSize * Math.random() + 1);
        // 生成随机的y坐标
        int y = (int) (boardSize * Math.random() + 1);
        // 返回移动对象
        return new Move(x, y);
    }

    // 判断移动是否合法
    private static boolean isLegalMove(Move move, int boardSize) {
        // 判断移动是否在棋盘范围内
        return (move.i >= 1) && (move.i <= boardSize) && (move.j >= 1) && (move.j <= boardSize);
    }

    // 打印游戏介绍
    private static void printIntro() {
        System.out.println("                                GOMOKO");
        System.out.println("              CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println("\n\n");
        System.out.println("WELCOME TO THE ORIENTAL GAME OF GOMOKO.");
        System.out.println("\n");
        System.out.println("THE GAME IS PLAYED ON AN N BY N GRID OF A SIZE");
        System.out.println("THAT YOU SPECIFY.  DURING YOUR PLAY, YOU MAY COVER ONE GRID");
        System.out.println("INTERSECTION WITH A MARKER. THE OBJECT OF THE GAME IS TO GET");
        System.out.println("5 ADJACENT MARKERS IN A ROW -- HORIZONTALLY, VERTICALLY, OR");
        System.out.println("DIAGONALLY.  ON THE BOARD DIAGRAM, YOUR MOVES ARE MARKED");
        System.out.println("WITH A '1' AND THE COMPUTER MOVES WITH A '2'.");
        System.out.println("\nTHE COMPUTER DOES NOT KEEP TRACK OF WHO HAS WON.");
        System.out.println("TO END THE GAME, TYPE -1,-1 FOR YOUR MOVE.\n ");
    }
    // 读取用户输入的棋盘大小，要求最小为7，最大为19
    private static int readBoardSize(Scanner scan) {
        System.out.print("WHAT IS YOUR BOARD SIZE (MIN 7/ MAX 19)? ");

        boolean validInput = false;
        int input = 0;
        while (!validInput) {
            try {
                input = scan.nextInt();
                // 如果输入小于最小值或大于最大值，则提示用户重新输入
                if (input < MIN_BOARD_SIZE || input > MAX_BOARD_SIZE) {
                    System.out.printf("I SAID, THE MINIMUM IS %s, THE MAXIMUM IS %s.\n", MIN_BOARD_SIZE, MAX_BOARD_SIZE);
                } else {
                    validInput = true;
                }
            } catch (InputMismatchException ex) {
                // 如果输入不是数字，则提示用户重新输入
                System.out.println("!NUMBER EXPECTED - RETRY INPUT LINE\n");
                validInput = false;
            } finally {
                // 清空输入缓冲区
                scan.nextLine();
            }
        }
        return input;
    }

    // 读取用户输入的下棋位置
    private static Move readMove(Scanner scan) {
        System.out.print("YOUR PLAY (I,J)? ");
        boolean validInput = false;
        Move move = new Move();
        while (!validInput) {
            // 读取用户输入的字符串，并按逗号分割
            String input = scan.nextLine();
            final String[] split = input.split(",");
            try {
                // 将分割后的字符串转换为整数，作为下棋位置
                move.i = Integer.parseInt(split[0]);
                move.j = Integer.parseInt(split[1]);
                validInput = true;
            } catch (NumberFormatException nfe) {
                // 如果输入不是数字，则提示用户重新输入
                System.out.println("!NUMBER EXPECTED - RETRY INPUT LINE\n? ");
            }

        }
        return move;
    }

    // 下棋位置的数据结构
    private static class Move {
        int i;
        int j;

        public Move() {
        }

        public Move(int i, int j) {
            this.i = i;
            this.j = j;
        }

        @Override
        public String toString() {
            return "Move{" +
                    "i=" + i +
                    ", j=" + j +
                    '}';
        }
    }
# 闭合前面的函数定义
```