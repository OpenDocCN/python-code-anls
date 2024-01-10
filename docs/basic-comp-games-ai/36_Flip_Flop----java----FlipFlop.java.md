# `basic-computer-games\36_Flip_Flop\java\FlipFlop.java`

```
import java.util.Scanner;  // 导入 Scanner 类，用于用户输入
import java.lang.Math;  // 导入 Math 类，用于数学计算

/**
 * FlipFlop 游戏
 * <p>
 * 基于 BASIC 版本的 FlipFlop 游戏
 * https://github.com/coding-horror/basic-computer-games/blob/main/36%20Flip%20Flop/flipflop.bas
 * <p>
 * 注意：本版本旨在将 1970 年代的 BASIC 游戏转换为 Java 版本，没有引入新功能 - 没有添加额外的文本、错误检查等。
 *
 * 由 Darren Cardenas 从 BASIC 转换为 Java。
 */

public class FlipFlop {

  private final Scanner scan;  // 用于用户输入的 Scanner 对象

  private enum Step {  // 定义枚举类型 Step
    RANDOMIZE, INIT_BOARD, GET_NUMBER, ILLEGAL_ENTRY, FLIP_POSITION, SET_X_FIRST, SET_X_SECOND,
    GENERATE_R_FIRST, GENERATE_R_SECOND, PRINT_BOARD, QUERY_RETRY
  }

  public FlipFlop() {

    scan = new Scanner(System.in);  // 初始化 Scanner 对象

  }  // 构造方法 FlipFlop 结束

  public void play() {

    showIntro();  // 显示游戏介绍
    startGame();  // 开始游戏

  }  // 方法 play 结束

  private static void showIntro() {

    System.out.println(" ".repeat(31) + "FLIPFLOP");  // 打印游戏标题
    System.out.println(" ".repeat(14) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");  // 打印创意计算机的信息
    System.out.println("");

  }  // 方法 showIntro 结束

  private void startGame() {

    double mathVal = 0;  // 定义数学值变量
    double randVal = 0;  // 定义随机值变量
    double tmpVal = 0;  // 定义临时值变量

    int index = 0;  // 定义索引变量
    int match = 0;  // 定义匹配变量
    int numFlip = 0;  // 定义翻转次数变量
    int numGuesses = 0;  // 定义猜测次数变量

    Step nextStep = Step.RANDOMIZE;  // 初始化下一步操作为 RANDOMIZE

    String userResponse = "";  // 用户响应字符串

    String[] board = new String[21];  // 定义长度为 21 的字符串数组

    System.out.println("THE OBJECT OF THIS PUZZLE IS TO CHANGE THIS:");  // 打印游戏目标
    System.out.println("");
    System.out.println("X X X X X X X X X X");  // 打印初始状态
    System.out.println("");
    System.out.println("TO THIS:");  // 打印目标状态
    System.out.println("");
    System.out.println("O O O O O O O O O O");  // 打印目标状态
    System.out.println("");
    System.out.println("BY TYPING THE NUMBER CORRESPONDING TO THE POSITION OF THE");  // 打印游戏说明
    System.out.println("LETTER ON SOME NUMBERS, ONE POSITION WILL CHANGE, ON");  // 打印游戏说明
  }
    // 打印提示信息
    System.out.println("OTHERS, TWO WILL CHANGE.  TO RESET LINE TO ALL X'S, TYPE 0");
    // 打印提示信息
    System.out.println("(ZERO) AND TO START OVER IN THE MIDDLE OF A GAME, TYPE ");
    // 打印空行
    System.out.println("");
    
    // 开始外部 while 循环
    }  // 结束外部 while 循环

  }  // startGame 方法结束

  public static void main(String[] args) {

    // 创建 FlipFlop 对象
    FlipFlop game = new FlipFlop();
    // 调用 play 方法
    game.play();

  }  // main 方法结束
# 结束 FlipFlop 类的定义
}  // End of class FlipFlop
```