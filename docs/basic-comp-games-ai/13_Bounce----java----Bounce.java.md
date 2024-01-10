# `basic-computer-games\13_Bounce\java\Bounce.java`

```
// 导入 Scanner 类和 Math 类
import java.util.Scanner;
import java.lang.Math;

/**
 * Bounce 游戏
 * <p>
 * 基于这里的 BASIC Bounce 游戏
 * https://github.com/coding-horror/basic-computer-games/blob/main/13%20Bounce/bounce.bas
 * <p>
 * 注意：这个想法是在 Java 中创建一个 1970 年代 BASIC 游戏的版本，没有引入新功能 - 没有添加额外的文本、错误检查等。
 *
 * 由 Darren Cardenas 从 BASIC 转换为 Java。
 */
public class Bounce {

  private final Scanner scan;  // 用于用户输入

  public Bounce() {
    // 初始化 Scanner 对象
    scan = new Scanner(System.in);
  }  // Bounce 构造函数结束

  public void play() {
    // 显示游戏介绍
    showIntro();
    // 开始游戏
    startGame();
  }  // play 方法结束

  private void showIntro() {
    // 显示游戏介绍
    System.out.println(" ".repeat(32) + "BOUNCE");
    System.out.println(" ".repeat(14) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
    System.out.println("\n\n");
  }  // showIntro 方法结束

  private void startGame() {
    // 初始化变量
    double coefficient = 0;
    double height = 0;
    double timeIncrement = 0;
    double timeIndex = 0;
    double timeTotal = 0;
    double velocity = 0;

    double[] timeData = new double[21];

    int heightInt = 0;
    int index = 0;
    int maxData = 0;

    String lineContent = "";

    System.out.println("THIS SIMULATION LETS YOU SPECIFY THE INITIAL VELOCITY");
    System.out.println("OF A BALL THROWN STRAIGHT UP, AND THE COEFFICIENT OF");
    System.out.println("ELASTICITY OF THE BALL.  PLEASE USE A DECIMAL FRACTION");
    System.out.println("COEFFICIENCY (LESS THAN 1).");
    System.out.println("");
    System.out.println("YOU ALSO SPECIFY THE TIME INCREMENT TO BE USED IN");
    System.out.println("'STROBING' THE BALL'S FLIGHT (TRY .1 INITIALLY).");
    System.out.println("");

    // 开始外部 while 循环
  }  // startGame 方法结束

  public static void main(String[] args) {
    // 创建 Bounce 对象并开始游戏
    Bounce game = new Bounce();
    game.play();
  }  // main 方法结束
}
}  # 类 Bounce 的结束
```