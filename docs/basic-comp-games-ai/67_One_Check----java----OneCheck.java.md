# `67_One_Check\java\OneCheck.java`

```
import java.util.Arrays;  # 导入 java.util.Arrays 包
import java.util.Scanner;  # 导入 java.util.Scanner 包

/**
 * Game of One Check
 * <p>
 * Based on the BASIC game of One Check here
 * https://github.com/coding-horror/basic-computer-games/blob/main/67%20One%20Check/onecheck.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's BASIC game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 *
 * Converted from BASIC to Java by Darren Cardenas.
 */

public class OneCheck {

  private final Scanner scan;  // For user input  # 创建一个 Scanner 对象用于用户输入

  private enum Step {  # 创建一个枚举类型 Step
    SHOW_INSTRUCTIONS, SHOW_BOARD, GET_MOVE, GET_SUMMARY, QUERY_RETRY
  }
  // 定义枚举类型，包含显示说明、显示棋盘、获取移动、获取总结、询问重试等操作

  public OneCheck() {
    // OneCheck 类的构造函数
    scan = new Scanner(System.in);
    // 创建一个 Scanner 对象，用于从标准输入读取数据
  }  // End of constructor OneCheck

  public void play() {
    // play 方法，用于开始游戏
    showIntro();
    // 调用 showIntro 方法显示游戏介绍
    startGame();
    // 调用 startGame 方法开始游戏
  }  // End of method play

  private static void showIntro() {
    // showIntro 方法，用于显示游戏介绍
    System.out.println(" ".repeat(29) + "ONE CHECK");
    // 打印游戏标题 "ONE CHECK"，并在前面添加空格
    System.out.println(" ".repeat(14) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
    // 打印创意计算机公司的信息，包括地点
    System.out.println("\n\n");  // 打印两个换行符

  }  // End of method showIntro  // showIntro 方法结束

  private void startGame() {  // startGame 方法开始

    int fromSquare = 0;  // 定义并初始化变量 fromSquare
    int numJumps = 0;  // 定义并初始化变量 numJumps
    int numPieces = 0;  // 定义并初始化变量 numPieces
    int square = 0;  // 定义并初始化变量 square
    int startPosition = 0;  // 定义并初始化变量 startPosition
    int toSquare = 0;  // 定义并初始化变量 toSquare

    // Move legality test variables  // 移动合法性测试变量
    int fromTest1 = 0;  // 定义并初始化变量 fromTest1
    int fromTest2 = 0;  // 定义并初始化变量 fromTest2
    int toTest1 = 0;  // 定义并初始化变量 toTest1
    int toTest2 = 0;  // 定义并初始化变量 toTest2

    int[] positions = new int[65];  // 创建一个长度为65的整型数组 positions
    Step nextStep = Step.SHOW_INSTRUCTIONS; // 定义并初始化一个名为nextStep的枚举类型变量，表示下一步的操作

    String lineContent = ""; // 定义并初始化一个空字符串变量lineContent，用于存储读取的每行内容
    String userResponse = ""; // 定义并初始化一个空字符串变量userResponse，用于存储用户的响应

    // 开始外部while循环
    while (true) {

      // 开始switch语句
      switch (nextStep) {

        case SHOW_INSTRUCTIONS: // 如果nextStep的值为SHOW_INSTRUCTIONS

          System.out.println("SOLITAIRE CHECKER PUZZLE BY DAVID AHL\n"); // 打印游戏说明
          System.out.println("48 CHECKERS ARE PLACED ON THE 2 OUTSIDE SPACES OF A"); // 打印游戏规则
          System.out.println("STANDARD 64-SQUARE CHECKERBOARD.  THE OBJECT IS TO"); // 打印游戏规则
          System.out.println("REMOVE AS MANY CHECKERS AS POSSIBLE BY DIAGONAL JUMPS"); // 打印游戏规则
          System.out.println("(AS IN STANDARD CHECKERS).  USE THE NUMBERED BOARD TO"); // 打印游戏规则
          System.out.println("INDICATE THE SQUARE YOU WISH TO JUMP FROM AND TO.  ON"); // 打印游戏规则
```
          // 打印每一轮的棋盘，'1' 表示一个棋子，'0' 表示一个空格。当没有可能的跳跃时，回答问题'JUMP FROM ?'时输入'0'
          System.out.println("THE BOARD PRINTED OUT ON EACH TURN '1' INDICATES A");
          System.out.println("CHECKER AND '0' AN EMPTY SQUARE.  WHEN YOU HAVE NO");
          System.out.println("POSSIBLE JUMPS REMAINING, INPUT A '0' IN RESPONSE TO");
          System.out.println("QUESTION 'JUMP FROM ?'\n");
          System.out.println("HERE IS THE NUMERICAL BOARD:\n");

          nextStep = Step.SHOW_BOARD;
          break;

        case SHOW_BOARD:

          // 开始循环遍历所有方格
          for (square = 1; square <= 57; square += 8) {

            lineContent = String.format("% -4d%-4d%-4d%-4d%-4d%-4d%-4d%-4d", square, square + 1, square + 2,
                                        square + 3, square + 4, square + 5, square + 6, square + 7);
            System.out.println(lineContent);

          }  // 结束循环遍历所有方格
          System.out.println("");  // 打印空行
          System.out.println("AND HERE IS THE OPENING POSITION OF THE CHECKERS.");  // 打印提示信息
          System.out.println("");  // 打印空行

          Arrays.fill(positions, 1);  // 使用值 1 填充数组 positions

          // 开始生成起始位置
          for (square = 19; square <= 43; square += 8) {  // 循环遍历起始位置的方格

            for (startPosition = square; startPosition <= square + 3; startPosition++) {  // 循环遍历每个方格的起始位置

              positions[startPosition] = 0;  // 将起始位置设置为 0

            }
          }  // 结束生成起始位置

          numJumps = 0;  // 将跳跃次数设置为 0

          printBoard(positions);  // 调用打印棋盘的函数，传入起始位置数组作为参数
          nextStep = Step.GET_MOVE;  # 设置下一步操作为获取移动步骤
          break;  # 跳出当前的 switch 语句

        case GET_MOVE:

          System.out.print("JUMP FROM? ");  # 打印提示信息，要求用户输入起始位置
          fromSquare = scan.nextInt();  # 从用户输入中获取起始位置
          scan.nextLine();  // Discard newline  # 丢弃输入中的换行符

          // User requested summary  # 如果用户请求摘要
          if (fromSquare == 0) {  # 如果起始位置为0
            nextStep = Step.GET_SUMMARY;  # 设置下一步操作为获取摘要
            break;  # 跳出当前的 switch 语句
          }

          System.out.print("TO? ");  # 打印提示信息，要求用户输入目标位置
          toSquare = scan.nextInt();  # 从用户输入中获取目标位置
          scan.nextLine();  // Discard newline  # 丢弃输入中的换行符
          System.out.println("");  # 打印空行
          // 检查移动的合法性
          fromTest1 = (int) Math.floor((fromSquare - 1.0) / 8.0); // 计算起始位置的行数
          fromTest2 = fromSquare - 8 * fromTest1; // 计算起始位置的列数
          toTest1 = (int) Math.floor((toSquare - 1.0) / 8.0); // 计算目标位置的行数
          toTest2 = toSquare - 8 * toTest1; // 计算目标位置的列数

          if ((fromTest1 > 7) || // 如果起始位置的行数超出棋盘范围
              (toTest1 > 7) || // 或者目标位置的行数超出棋盘范围
              (fromTest2 > 8) || // 或者起始位置的列数超出棋盘范围
              (toTest2 > 8) || // 或者目标位置的列数超出棋盘范围
              (Math.abs(fromTest1 - toTest1) != 2) || // 或者行数的差值不为2
              (Math.abs(fromTest2 - toTest2) != 2) || // 或者列数的差值不为2
              (positions[(toSquare + fromSquare) / 2] == 0) || // 或者中间位置没有棋子
              (positions[fromSquare] == 0) || // 或者起始位置没有棋子
              (positions[toSquare] == 1)) { // 或者目标位置已经有棋子

            System.out.println("ILLEGAL MOVE.  TRY AGAIN..."); // 打印非法移动的提示信息
            nextStep = Step.GET_MOVE; // 设置下一步为获取移动
            break; // 跳出循环
          }
          positions[toSquare] = 1;  // 将目标位置标记为有棋子
          positions[fromSquare] = 0;  // 将起始位置标记为无棋子
          positions[(toSquare + fromSquare) / 2] = 0;  // 将跳跃位置标记为无棋子
          numJumps++;  // 跳跃次数加一

          printBoard(positions);  // 打印当前棋盘状态

          nextStep = Step.GET_MOVE;  // 设置下一步为获取移动
          break;  // 跳出 switch 语句

        case GET_SUMMARY:

          numPieces = 0;  // 初始化剩余棋子数量为0

          // 计算剩余棋子数量
          for (square = 1; square <= 64; square++) {
            numPieces += positions[square];  // 统计每个位置上的棋子数量
          }
          System.out.println("");  // 打印空行
          System.out.println("YOU MADE " + numJumps + " JUMPS AND HAD " + numPieces + " PIECES");  // 打印玩家跳跃次数和剩余棋子数量
          System.out.println("REMAINING ON THE BOARD.\n");  // 打印剩余在棋盘上的棋子数量

          nextStep = Step.QUERY_RETRY;  // 设置下一步操作为查询重试
          break;  // 跳出 switch 语句

        case QUERY_RETRY:  // 查询重试情况

          while (true) {  // 进入循环，直到用户输入合法的回答
            System.out.print("TRY AGAIN? ");  // 提示用户再试一次
            userResponse = scan.nextLine();  // 读取用户输入
            System.out.println("");  // 打印空行

            if (userResponse.toUpperCase().equals("YES")) {  // 如果用户输入是"YES"
              nextStep = Step.SHOW_BOARD;  // 设置下一步操作为展示棋盘
              break;  // 跳出循环
            }
            else if (userResponse.toUpperCase().equals("NO")) {  // 如果用户输入是"NO"
              System.out.println("O.K.  HOPE YOU HAD FUN!!");  // 打印消息
              return;  // 结束当前方法的执行，返回到调用该方法的地方
            }
            else {
              System.out.println("PLEASE ANSWER 'YES' OR 'NO'.");  // 打印提示信息
            }
          }
          break;

        default:
          System.out.println("INVALID STEP");  // 打印提示信息
          nextStep = Step.QUERY_RETRY;  // 设置下一步的操作为重试查询
          break;

      }  // End of switch  // switch 语句结束
    }  // End outer while loop  // 外部 while 循环结束
  }  // End of method startGame  // startGame 方法结束

  public void printBoard(int[] positions) {  // 定义一个名为 printBoard 的方法，参数为一个整型数组 positions
    int column = 0;  // 初始化列数为0
    int row = 0;  // 初始化行数为0
    String lineContent = "";  // 初始化行内容为空字符串

    // 开始循环遍历所有行
    for (row = 1; row <= 57; row += 8) {

      // 开始循环遍历所有列
      for (column = row; column <= row + 7; column++) {

        lineContent += " " + positions[column];  // 将当前位置的值添加到行内容中

      }  // 结束循环遍历所有列

      System.out.println(lineContent);  // 打印行内容
      lineContent = "";  // 重置行内容为空字符串

    }  // 结束循环遍历所有行
    System.out.println("");  // 打印空行

  }  // End of method printBoard  // 打印板方法结束

  public static void main(String[] args) {  // 主方法开始

    OneCheck game = new OneCheck();  // 创建一个OneCheck对象
    game.play();  // 调用play方法开始游戏

  }  // End of method main  // 主方法结束

}  // End of class OneCheck  // OneCheck类结束
```