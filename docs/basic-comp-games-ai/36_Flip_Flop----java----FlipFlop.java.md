# `36_Flip_Flop\java\FlipFlop.java`

```
import java.util.Scanner;  # 导入 java.util.Scanner 包
import java.lang.Math;  # 导入 java.lang.Math 包

/**
 * Game of FlipFlop
 * <p>
 * Based on the BASIC game of FlipFlop here
 * https://github.com/coding-horror/basic-computer-games/blob/main/36%20Flip%20Flop/flipflop.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's BASIC game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 *
 * Converted from BASIC to Java by Darren Cardenas.
 */

public class FlipFlop {

  private final Scanner scan;  // For user input  # 创建一个 Scanner 对象用于用户输入

  private enum Step {  # 创建一个枚举类型 Step
    RANDOMIZE, INIT_BOARD, GET_NUMBER, ILLEGAL_ENTRY, FLIP_POSITION, SET_X_FIRST, SET_X_SECOND,
    GENERATE_R_FIRST, GENERATE_R_SECOND, PRINT_BOARD, QUERY_RETRY
  }
  // 定义一个枚举类型，包含了游戏中可能用到的一些操作

  public FlipFlop() {
    // 构造函数，初始化一个 Scanner 对象，用于接收用户输入
    scan = new Scanner(System.in);
  }  // End of constructor FlipFlop
  // 构造函数结束

  public void play() {
    // 游戏主程序入口
    showIntro();
    startGame();
  }  // End of method play
  // play 方法结束

  private static void showIntro() {
    // 显示游戏介绍
    System.out.println(" ".repeat(31) + "FLIPFLOP");
  }
  // showIntro 方法结束
    System.out.println(" ".repeat(14) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
    # 打印输出一行字符串，使用空格重复14次，然后输出"CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"
    System.out.println("");
    # 打印输出一个空行

  }  // End of method showIntro
  # 方法showIntro结束

  private void startGame() {
  # 定义一个私有方法startGame

    double mathVal = 0;
    # 定义一个双精度浮点数变量mathVal，并初始化为0
    double randVal = 0;
    # 定义一个双精度浮点数变量randVal，并初始化为0
    double tmpVal = 0;
    # 定义一个双精度浮点数变量tmpVal，并初始化为0

    int index = 0;
    # 定义一个整数变量index，并初始化为0
    int match = 0;
    # 定义一个整数变量match，并初始化为0
    int numFlip = 0;
    # 定义一个整数变量numFlip，并初始化为0
    int numGuesses = 0;
    # 定义一个整数变量numGuesses，并初始化为0

    Step nextStep = Step.RANDOMIZE;
    # 定义一个Step类型的变量nextStep，并初始化为Step.RANDOMIZE

    String userResponse = "";
    # 定义一个字符串类型的变量userResponse，并初始化为空字符串
    // 创建一个长度为21的字符串数组
    String[] board = new String[21];

    // 打印游戏说明
    System.out.println("THE OBJECT OF THIS PUZZLE IS TO CHANGE THIS:");
    System.out.println("");
    System.out.println("X X X X X X X X X X");
    System.out.println("");
    System.out.println("TO THIS:");
    System.out.println("");
    System.out.println("O O O O O O O O O O");
    System.out.println("");
    System.out.println("BY TYPING THE NUMBER CORRESPONDING TO THE POSITION OF THE");
    System.out.println("LETTER ON SOME NUMBERS, ONE POSITION WILL CHANGE, ON");
    System.out.println("OTHERS, TWO WILL CHANGE.  TO RESET LINE TO ALL X'S, TYPE 0");
    System.out.println("(ZERO) AND TO START OVER IN THE MIDDLE OF A GAME, TYPE ");
    System.out.println("11 (ELEVEN).");
    System.out.println("");

    // 开始外部循环
    while (true) {
      // 开始 switch 语句
      switch (nextStep) {

        case RANDOMIZE:

          // 生成一个随机数
          randVal = Math.random();

          // 打印起始线
          System.out.println("HERE IS THE STARTING LINE OF X'S.");
          System.out.println("");

          // 重置猜测次数
          numGuesses = 0;
          // 设置下一步为初始化游戏板
          nextStep = Step.INIT_BOARD;
          break;

        case INIT_BOARD:

          // 打印游戏板的初始状态
          System.out.println("1 2 3 4 5 6 7 8 9 10");
          System.out.println("X X X X X X X X X X");
          System.out.println("");
          // 避免越界错误，从零开始循环
          for (index = 0; index <= 10; index++) {
            board[index] = "X";
          }

          // 设置下一步操作为获取数字
          nextStep = Step.GET_NUMBER;
          break;

        case GET_NUMBER:

          // 提示用户输入数字
          System.out.print("INPUT THE NUMBER? ");
          // 读取用户输入
          userResponse = scan.nextLine();

          // 尝试将用户输入的字符串转换为整数
          try {
            numFlip = Integer.parseInt(userResponse);
          }
          // 如果转换失败，设置下一步操作为非法输入
          catch (NumberFormatException ex) {
            nextStep = Step.ILLEGAL_ENTRY;
            break;
          }
          // 如果翻转次数为11，开始一个新游戏
          if (numFlip == 11) {
            nextStep = Step.RANDOMIZE;
            break;
          }

          // 如果翻转次数大于11，表示非法操作
          if (numFlip > 11) {
            nextStep = Step.ILLEGAL_ENTRY;
            break;
          }

          // 如果翻转次数为0，重置游戏板
          if (numFlip == 0) {
            nextStep = Step.INIT_BOARD;
            break;
          }

          // 如果匹配次数等于翻转次数，翻转位置
          if (match == numFlip) {
            nextStep = Step.FLIP_POSITION;
            break;  # 结束当前的 switch 语句的执行，跳出循环
          }

          match = numFlip;  # 将 numFlip 的值赋给 match

          if (board[numFlip].equals("O")) {  # 如果 board[numFlip] 的值等于 "O"
            nextStep = Step.SET_X_FIRST;  # 将 nextStep 设置为 Step.SET_X_FIRST
            break;  # 结束当前的 switch 语句的执行，跳出循环
          }

          board[numFlip] = "O";  # 将 board[numFlip] 的值设置为 "O"
          nextStep = Step.GENERATE_R_FIRST;  # 将 nextStep 设置为 Step.GENERATE_R_FIRST
          break;  # 结束当前的 switch 语句的执行，跳出循环

        case ILLEGAL_ENTRY:  # 如果当前状态是 ILLEGAL_ENTRY
          System.out.println("ILLEGAL ENTRY--TRY AGAIN.");  # 打印 "ILLEGAL ENTRY--TRY AGAIN."
          nextStep = Step.GET_NUMBER;  # 将 nextStep 设置为 Step.GET_NUMBER
          break;  # 结束当前的 switch 语句的执行，跳出循环

        case GENERATE_R_FIRST:  # 如果当前状态是 GENERATE_R_FIRST
# 计算数学值，使用 Math.tan()、Math.sin() 函数
mathVal = Math.tan(randVal + numFlip / randVal - numFlip) - Math.sin(randVal / numFlip) + 336 * Math.sin(8 * numFlip);

# 取得 mathVal 的小数部分
tmpVal = mathVal - (int)Math.floor(mathVal);

# 将小数部分乘以 10 并取整，得到 numFlip
numFlip = (int)(10 * tmpVal);

# 如果 board[numFlip] 的值等于 "O"，则将 nextStep 设置为 Step.SET_X_FIRST，并跳出循环
if (board[numFlip].equals("O")) {
    nextStep = Step.SET_X_FIRST;
    break;
}

# 将 board[numFlip] 的值设置为 "O"
board[numFlip] = "O";

# 将 nextStep 设置为 Step.PRINT_BOARD，并跳出循环
nextStep = Step.PRINT_BOARD;
break;

# 如果 nextStep 为 SET_X_FIRST，则将 board[numFlip] 的值设置为 "X"
case SET_X_FIRST:
    board[numFlip] = "X";
          if (match == numFlip) {  # 如果匹配等于翻转次数
            nextStep = Step.GENERATE_R_FIRST;  # 下一步是生成R的第一步
          } else {
            nextStep = Step.PRINT_BOARD;  # 否则下一步是打印棋盘
          }
          break;

        case FLIP_POSITION:

          if (board[numFlip].equals("O")) {  # 如果棋盘上numFlip位置的值等于"O"
            nextStep = Step.SET_X_SECOND;  # 下一步是设置X的第二步
            break;
          }

          board[numFlip] = "O";  # 否则将棋盘上numFlip位置的值设置为"O"
          nextStep = Step.GENERATE_R_SECOND;  # 下一步是生成R的第二步
          break;

        case GENERATE_R_SECOND:
          # 计算数学值
          mathVal = 0.592 * (1 / Math.tan(randVal / numFlip + randVal)) / Math.sin(numFlip * 2 + randVal)
                    - Math.cos(numFlip);

          # 计算临时值
          tmpVal = mathVal - (int)mathVal;
          numFlip = (int)(10 * tmpVal);

          # 如果棋盘上的位置为"O"，则执行下一步操作为设置"X"
          if (board[numFlip].equals("O")) {
            nextStep = Step.SET_X_SECOND;
            break;
          }

          # 将棋盘上的位置设置为"O"
          board[numFlip] = "O";
          # 执行下一步操作为打印棋盘
          nextStep = Step.PRINT_BOARD;
          break;

        case SET_X_SECOND:

          # 将棋盘上的位置设置为"X"
          board[numFlip] = "X";
          # 如果匹配的位置等于numFlip，则执行下一步操作为生成R_SECOND
          if (match == numFlip) {
            nextStep = Step.GENERATE_R_SECOND;
            break;  # 结束当前循环或者 switch 语句的执行，跳出当前的循环或者 switch 语句
          }

          nextStep = Step.PRINT_BOARD;  # 设置下一步的操作为打印游戏板
          break;  # 结束当前 switch 语句的执行

        case PRINT_BOARD:  # 当前操作为打印游戏板
          System.out.println("1 2 3 4 5 6 7 8 9 10");  # 打印游戏板的列号

          for (index = 1; index <= 10; index++) {  # 遍历游戏板的每一列
            System.out.print(board[index] + " ");  # 打印当前列的状态
          }

          numGuesses++;  # 猜测次数加一

          System.out.println("");  # 打印换行

          for (index = 1; index <= 10; index++) {  # 再次遍历游戏板的每一列
            if (!board[index].equals("O")) {  # 如果当前列不是空位
              nextStep = Step.GET_NUMBER;  # 设置下一步的操作为获取用户输入
```

          break;  # 结束当前的循环或者switch语句的执行，跳出当前的循环或者switch语句
        }
      }

      if (nextStep == Step.GET_NUMBER) {  # 如果下一步是获取数字
        break;  # 结束当前的循环或者switch语句的执行，跳出当前的循环或者switch语句
      }

      if (numGuesses > 12) {  # 如果猜测次数大于12
        System.out.println("TRY HARDER NEXT TIME.  IT TOOK YOU " + numGuesses + " GUESSES.");  # 打印提示信息
      } else {
        System.out.println("VERY GOOD.  YOU GUESSED IT IN ONLY " + numGuesses + " GUESSES.");  # 打印提示信息
      }
      nextStep = Step.QUERY_RETRY;  # 设置下一步为查询重试
      break;  # 结束当前的循环或者switch语句的执行，跳出当前的循环或者switch语句

    case QUERY_RETRY:  # 如果当前步骤是查询重试

      System.out.print("DO YOU WANT TO TRY ANOTHER PUZZLE? ");  # 打印提示信息
      userResponse = scan.nextLine();  # 从控制台获取用户输入
          if (userResponse.toUpperCase().charAt(0) == 'N') {  // 如果用户输入的响应转换为大写后的第一个字符是 'N'
            return;  // 返回，结束方法
          }
          System.out.println("");  // 打印空行
          nextStep = Step.RANDOMIZE;  // 将 nextStep 设置为 Step.RANDOMIZE
          break;  // 跳出 switch 语句

        default:  // 默认情况
          System.out.println("INVALID STEP");  // 打印 "INVALID STEP"
          nextStep = Step.QUERY_RETRY;  // 将 nextStep 设置为 Step.QUERY_RETRY
          break;  // 跳出 switch 语句

      }  // switch 语句结束

    }  // 外部 while 循环结束

  }  // startGame 方法结束

  public static void main(String[] args) {  // 主方法开始
    FlipFlop game = new FlipFlop();  // 创建一个名为game的FlipFlop对象
    game.play();  // 调用FlipFlop对象的play方法

  }  // End of method main  // main方法结束

}  // End of class FlipFlop  // FlipFlop类结束
```