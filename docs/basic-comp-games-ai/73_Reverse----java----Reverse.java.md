# `basic-computer-games\73_Reverse\java\Reverse.java`

```py
// 导入 Scanner 类和 Math 类
import java.util.Scanner;
import java.lang.Math;

/**
 * Reverse 游戏
 * <p>
 * 基于 BASIC 游戏 Reverse，链接在这里
 * https://github.com/coding-horror/basic-computer-games/blob/main/73%20Reverse/reverse.bas
 * <p>
 * 注意：这个想法是在 Java 中创建一个 1970 年代 BASIC 游戏的版本，不引入新功能 - 没有添加额外的文本、错误检查等。
 *
 * 由 Darren Cardenas 从 BASIC 转换为 Java。
 */

public class Reverse {

  private final int NUMBER_COUNT = 9;  // 数字的数量

  private final Scanner scan;  // 用于用户输入

  private enum Step {
    INITIALIZE, PERFORM_REVERSE, TRY_AGAIN, END_GAME  // 步骤枚举
  }

  public Reverse() {
    scan = new Scanner(System.in);  // 初始化 Scanner 对象
  }  // 构造函数 Reverse 结束

  public void play() {
    showIntro();  // 显示游戏介绍
    startGame();  // 开始游戏
  }  // 方法 play 结束

  private static void showIntro() {
    System.out.println(" ".repeat(31) + "REVERSE");  // 打印游戏标题
    System.out.println(" ".repeat(14) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");  // 打印创意计算机的信息
    System.out.println("\n\n");
    System.out.println("REVERSE -- A GAME OF SKILL");  // 打印游戏类型
    System.out.println("");
  }  // 方法 showIntro 结束

  private void startGame() {
    int index = 0;  // 索引
    int numMoves = 0;  // 移动次数
    int numReverse = 0;  // 反转次数
    int tempVal = 0;  // 临时值
    int[] numList = new int[NUMBER_COUNT + 1];  // 数字列表

    Step nextStep = Step.INITIALIZE;  // 下一步骤

    String userResponse = "";  // 用户响应

    System.out.print("DO YOU WANT THE RULES? ");  // 打印提示信息
    userResponse = scan.nextLine();  // 获取用户输入

    if (!userResponse.toUpperCase().equals("NO")) {  // 如果用户响应不是 "NO"
      this.printRules();  // 打印游戏规则
    }

    // 开始外部 while 循环
    while (true) {
      // 开始外部 switch
    }  // 外部 while 循环结束
  }  // 方法 startGame 结束

  public boolean findDuplicates(int[] board, int length) {
    int index = 0;  // 索引

    for (index = 1; index <= length - 1; index++) {  // 循环遍历数组

      // 识别重复项
      if (board[length] == board[index]) {  // 如果找到重复项
        return true;  // 找到重复项
      }
    }
    return false;  // 没有找到重复项，返回 false

  }  // 方法 findDuplicates 结束

  public void printBoard(int[] board) {

    int index = 0;

    System.out.println("");

    for (index = 1; index <= NUMBER_COUNT; index++) {

      System.out.format("%2d", board[index]);
    }

    System.out.println("\n");

  }  // 方法 printBoard 结束

  public void printRules() {

    System.out.println("");
    System.out.println("THIS IS THE GAME OF 'REVERSE'.  TO WIN, ALL YOU HAVE");
    System.out.println("TO DO IS ARRANGE A LIST OF NUMBERS (1 THROUGH " + NUMBER_COUNT + ")");
    System.out.println("IN NUMERICAL ORDER FROM LEFT TO RIGHT.  TO MOVE, YOU");
    System.out.println("TELL ME HOW MANY NUMBERS (COUNTING FROM THE LEFT) TO");
    System.out.println("REVERSE.  FOR EXAMPLE, IF THE CURRENT LIST IS:");
    System.out.println("");
    System.out.println("2 3 4 5 1 6 7 8 9");
    System.out.println("");
    System.out.println("AND YOU REVERSE 4, THE RESULT WILL BE:");
    System.out.println("");
    System.out.println("5 4 3 2 1 6 7 8 9");
    System.out.println("");
    System.out.println("NOW IF YOU REVERSE 5, YOU WIN!");
    System.out.println("");
    System.out.println("1 2 3 4 5 6 7 8 9");
    System.out.println("");
    System.out.println("NO DOUBT YOU WILL LIKE THIS GAME, BUT");
    System.out.println("IF YOU WANT TO QUIT, REVERSE 0 (ZERO).");
    System.out.println("");

  }  // 方法 printRules 结束

  public static void main(String[] args) {

    Reverse game = new Reverse();
    game.play();

  }  // 方法 main 结束
}  // 类 Reverse 的结束
```