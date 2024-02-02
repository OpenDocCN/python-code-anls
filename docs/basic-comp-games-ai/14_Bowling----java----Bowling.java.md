# `basic-computer-games\14_Bowling\java\Bowling.java`

```py
import java.util.Scanner;  // 导入 Scanner 类，用于用户输入
import java.lang.Math;  // 导入 Math 类，用于数学计算

/**
 * 保龄球游戏
 * <p>
 * 基于这里的 BASIC 保龄球游戏
 * https://github.com/coding-horror/basic-computer-games/blob/main/14%20Bowling/bowling.bas
 * <p>
 * 注意：这个想法是在 Java 中创建一个 1970 年代 BASIC 游戏的版本，不引入新功能 - 没有添加额外的文本、错误检查等。
 *
 * 由 Darren Cardenas 从 BASIC 转换为 Java。
 */

public class Bowling {

  private final Scanner scan;  // 用于用户输入的 Scanner 对象

  public Bowling() {

    scan = new Scanner(System.in);  // 初始化 Scanner 对象

  }  // Bowling 构造函数结束

  public void play() {

    showIntro();  // 显示游戏介绍
    startGame();  // 开始游戏

  }  // play 方法结束

  private static void showIntro() {

    System.out.println(" ".repeat(33) + "BOWL");  // 打印游戏标题
    System.out.println(" ".repeat(14) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");  // 打印创意计算机的信息
    System.out.println("\n\n");  // 打印空行

  }  // showIntro 方法结束

  private void startGame() {

    int ball = 0;  // 球的数量
    int bell = 0;  // 钟声的数量
    int frame = 0;  // 框的数量
    int ii = 0;  // 循环迭代器
    int jj = 0;  // 循环迭代器
    int kk = 0;  // 循环迭代器
    int numPlayers = 0;  // 玩家数量
    int pinsDownBefore = 0;  // 击倒的瓶子数量（之前）
    int pinsDownNow = 0;  // 击倒的瓶子数量（现在）
    int player = 0;  // 玩家
    int randVal = 0;  // 随机值
    int result = 0;  // 结果

    int[] pins = new int[16];  // 球瓶数组

    int[][] scores = new int[101][7];  // 分数数组

    String userResponse = "";  // 用户响应

    System.out.println("WELCOME TO THE ALLEY");  // 打印欢迎词
    System.out.println("BRING YOUR FRIENDS");  // 打印邀请朋友
    System.out.println("OKAY LET'S FIRST GET ACQUAINTED");  // 打印认识一下
    System.out.println("");  // 打印空行
    System.out.println("THE INSTRUCTIONS (Y/N)");  // 打印是否阅读游戏说明
    System.out.print("? ");  // 打印提示符

    userResponse = scan.nextLine();  // 获取用户输入

    if (userResponse.toUpperCase().equals("Y")) {  // 如果用户输入是 Y
      printRules();  // 打印游戏规则
    }

    System.out.print("FIRST OF ALL...HOW MANY ARE PLAYING? ");  // 打印询问玩家数量
    numPlayers = Integer.parseInt(scan.nextLine());  // 获取玩家数量

    System.out.println("");  // 打印空行
    System.out.println("VERY GOOD...");  // 打印提示

    // 开始外部 while 循环
  }  // 结束外部 while 循环

  }  // 结束 startGame 方法

  public static void printRules() {

    System.out.println("保龄球比赛需要技巧和技能。在比赛过程中，电脑会记录分数。你可以和其他玩家（最多四个）一起比赛。比赛共有十轮。在引脚图中，'O' 表示引脚倒下，'+' 表示引脚站立。比赛结束后，电脑会展示你的分数。");

  }  // 结束 printRules 方法

  public static void main(String[] args) {

    Bowling game = new Bowling();
    game.play();

  }  // 结束 main 方法
# 类 Bowling 的结束
```