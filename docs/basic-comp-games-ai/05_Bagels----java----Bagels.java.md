# `basic-computer-games\05_Bagels\java\Bagels.java`

```py
// 导入 Scanner 类，用于接收用户输入
import java.util.Scanner;

// 定义 Bagels 类
public class Bagels {

  // 主函数
  public static void main(String[] args) {

    // 初始化游戏赢得的次数
    int gamesWon = 0;

    // 输出游戏介绍文本
    System.out.println("\n\n                Bagels");
    System.out.println("Creative Computing  Morristown, New Jersey");
    System.out.println("\n\n");
    System.out.print("Would you like the rules (Yes or No)? ");

    // 需要说明吗？
    // 创建 Scanner 对象，用于接收用户输入
    Scanner scan = new Scanner(System.in);
    // 读取用户输入
    String s = scan.nextLine();
    // 如果输入字符串为空或者第一个字符不是大写的N，则输出游戏规则说明
    if (s.length() == 0 || s.toUpperCase().charAt(0) != 'N') {
      System.out.println();
      System.out.println("I am thinking of a three-digit number.  Try to guess");
      System.out.println("my number and I will give you clues as follows:");
      System.out.println("   PICO   - One digit correct but in the wrong position");
      System.out.println("   FERMI  - One digit correct and in the right position");
      System.out.println("   BAGELS - No digits correct");
    }

    // 循环进行多次游戏
    boolean stillPlaying = true;
    while(stillPlaying) {

      // 设置一个新游戏
      BagelGame game = new BagelGame();
      System.out.println("\nO.K.  I have a number in mind.");

      // 循环进行猜测和回应，直到游戏结束
      while (!game.isOver()) {
        String guess = getValidGuess(game);
        String response = game.makeGuess(guess);
        // 如果游戏已经赢了，则不输出回应
        if (!game.isWon()) {
          System.out.println(response);
        }
      }

      // 游戏结束。是赢了还是输了？
      if (game.isWon()) {
        System.out.println("You got it!!!\n");
        gamesWon++;
      } else {
        System.out.println("Oh well");
        System.out.print("That's " + BagelGame.MAX_GUESSES + " guesses.  ");
        System.out.println("My number was " + game.getSecretAsString());
      }

      stillPlaying = getReplayResponse();
    }

    // 输出道别信息
    if (gamesWon > 0) {
      System.out.println("\nA " + gamesWon + " point Bagels buff!!");
    }
    System.out.println("Hope you had fun.  Bye.\n");
  }

  // 获取有效的猜测
  private static String getValidGuess(BagelGame game) {
    // 一直要求输入猜测，直到合法为止
    Scanner scan = new Scanner(System.in);
    boolean valid = false;
    String guess = "";
    String error;
    // 当输入不合法时循环，直到输入合法为止
    while (!valid) {
      // 打印猜测次数和提示信息
      System.out.print("Guess # " + game.getGuessNum() + "     ? ");
      // 获取用户输入并去除首尾空格
      guess = scan.nextLine().trim();
      // 验证用户输入的猜测是否合法
      error = game.validateGuess(guess);
      // 如果没有错误，设置 valid 为 true
      if (error == "") {
        valid = true;
      } else {
        // 如果有错误，打印错误信息
        System.out.println(error);
      }
    }
    // 返回用户的猜测
    return guess;
  }

  // 获取用户是否想要再玩一次的响应
  private static boolean getReplayResponse() {
    // 创建一个 Scanner 对象来获取用户输入
    Scanner scan = new Scanner(System.in);
    // 循环直到输入合法
    while (true) {
      // 提示用户输入是否想再玩一次
      System.out.print("Play again (Yes or No)? ");
      // 获取用户输入并去除首尾空格
      String response = scan.nextLine().trim();
      // 如果输入长度大于 0，返回输入的响应是否以 Y 开头（不区分大小写）
      if (response.length() > 0) {
        return response.toUpperCase().charAt(0) == 'Y';
      }
    }
  }
# 闭合前面的函数定义
```