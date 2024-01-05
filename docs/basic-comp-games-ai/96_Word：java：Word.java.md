# `96_Word\java\Word.java`

```
import java.util.Arrays;  # 导入 java.util.Arrays 包，用于操作数组
import java.util.Scanner;  # 导入 java.util.Scanner 包，用于接收用户输入

/**
 * Game of Word
 * <p>
 * Based on the BASIC game of Word here
 * https://github.com/coding-horror/basic-computer-games/blob/main/96%20Word/word.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's BASIC game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 *
 * Converted from BASIC to Java by Darren Cardenas.
 */

public class Word {

  private final static String[] WORDS = {  # 定义一个包含单词的字符串数组

  "DINKY", "SMOKE", "WATER", "GRASS", "TRAIN", "MIGHT",  # 单词列表
  "FIRST", "CANDY", "CHAMP", "WOULD", "CLUMP", "DOPEY"  # 定义一个字符串数组，包含了游戏中的单词

  };

  private final Scanner scan;  // 用于用户输入的扫描器对象

  private enum Step {  // 定义一个枚举类型 Step，包含了游戏的不同阶段
    INITIALIZE, MAKE_GUESS, USER_WINS
  }

  public Word() {  // Word 类的构造函数

    scan = new Scanner(System.in);  // 初始化扫描器对象，用于接收用户输入

  }  // 构造函数 Word 的结束

  public void play() {  // 定义一个 play 方法，用于开始游戏

    showIntro();  // 调用 showIntro 方法，展示游戏介绍
    startGame();  // 调用 startGame 方法，开始游戏
  }  // End of method play

  private void showIntro() {
    // 打印游戏标题
    System.out.println(" ".repeat(32) + "WORD");
    // 打印游戏信息
    System.out.println(" ".repeat(14) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
    System.out.println("\n\n");

    // 打印游戏提示
    System.out.println("I AM THINKING OF A WORD -- YOU GUESS IT.  I WILL GIVE YOU");
    System.out.println("CLUES TO HELP YOU GET IT.  GOOD LUCK!!");
    System.out.println("\n");

  }  // End of method showIntro

  private void startGame() {
    // 创建一个包含8个元素的字符数组
    char[] commonLetters = new char[8];
    // 创建一个包含8个元素的字符数组
    char[] exactLetters = new char[8];
    int commonIndex = 0;  // 初始化公共索引
    int ii = 0;  // 循环迭代器
    int jj = 0;  // 循环迭代器
    int numGuesses = 0;  // 猜测次数
    int numMatches = 0;  // 匹配次数
    int wordIndex = 0;  // 单词索引

    Step nextStep = Step.INITIALIZE;  // 初始化下一步操作

    String commonString = "";  // 公共字符串
    String exactString = "";  // 精确字符串
    String guessWord = "";  // 猜测的单词
    String secretWord = "";  // 秘密单词
    String userResponse = "";  // 用户响应

    // 开始外部循环
    while (true) {

      switch (nextStep) {
        case INITIALIZE:  // 初始化游戏状态
          System.out.println("\n");  // 打印空行
          System.out.println("YOU ARE STARTING A NEW GAME...");  // 打印提示信息

          // 从单词列表中随机选择一个作为秘密单词
          wordIndex = (int) (Math.random() * WORDS.length);
          secretWord = WORDS[wordIndex];

          numGuesses = 0;  // 猜测次数初始化为0

          // 将exactLetters和commonLetters数组的索引1到6的元素填充为'-'和'\0'
          Arrays.fill(exactLetters, 1, 6, '-');
          Arrays.fill(commonLetters, 1, 6, '\0');

          nextStep = Step.MAKE_GUESS;  // 设置下一步为进行猜测
          break;

        case MAKE_GUESS:  // 进行猜测
          System.out.print("GUESS A FIVE LETTER WORD? ");  // 提示玩家猜测一个五个字母的单词
          guessWord = scan.nextLine().toUpperCase();  // 从用户输入中读取猜测的单词并转换为大写

          numGuesses++;  // 猜测次数加一

          // 胜利条件
          if (guessWord.equals(secretWord)) {  // 如果猜测的单词与秘密单词相同
            nextStep = Step.USER_WINS;  // 设置下一步为用户获胜
            continue;  // 继续执行下一步
          }

          Arrays.fill(commonLetters, 1, 8, '\0');  // 将数组commonLetters的索引1到8的元素填充为'\0'

          // 投降条件
          if (guessWord.equals("?")) {  // 如果猜测的单词为"？"
            System.out.println("THE SECRET WORD IS " + secretWord);  // 打印出秘密单词
            System.out.println("");
            nextStep = Step.INITIALIZE;  // 设置下一步为初始化，即重新开始游戏
            continue;  // 继续执行下一步
          }
          // 检查输入是否有效
          if (guessWord.length() != 5) {
            System.out.println("YOU MUST GUESS A 5 LETTER WORD.  START AGAIN.");
            numGuesses--;
            nextStep = Step.MAKE_GUESS;  // 再次猜测
            continue;
          }

          numMatches = 0;  // 匹配数量初始化为0
          commonIndex = 1;  // 公共索引初始化为1

          for (ii = 1; ii <= 5; ii++) {  // 循环遍历秘密单词和猜测单词的每个字母

            for (jj = 1; jj <= 5; jj++) {

              if (secretWord.charAt(ii - 1) != guessWord.charAt(jj - 1)) {  // 如果秘密单词和猜测单词的字母不匹配，则继续下一次循环
                continue;
              }

              // 避免数组越界错误
              if (commonIndex <= 5) {  # 如果commonIndex小于等于5
                commonLetters[commonIndex] = guessWord.charAt(jj - 1);  # 将guessWord中第(jj-1)个字符赋值给commonLetters数组中的第commonIndex个位置
                commonIndex++;  # commonIndex加1
              }

              if (ii == jj) {  # 如果ii等于jj
                exactLetters[jj] = guessWord.charAt(jj - 1);  # 将guessWord中第(jj-1)个字符赋值给exactLetters数组中的第jj个位置
              }

              // Avoid out of bounds errors  # 避免数组越界错误
              if (numMatches < 5) {  # 如果numMatches小于5
                numMatches++;  # numMatches加1
              }
            }
          }

          exactString = "";  # 将exactString置空
          commonString = "";  # 将commonString置空

          // Build the exact letters string  # 构建exact letters字符串
          for (ii = 1; ii <= 5; ii++) {
            // 从 exactLetters 数组中取出前五个元素拼接成字符串
            exactString += exactLetters[ii];
          }

          // 构建 commonString 字符串
          for (ii = 1; ii <= numMatches; ii++) {
            // 从 commonLetters 数组中取出 numMatches 个元素拼接成字符串
            commonString += commonLetters[ii];
          }

          // 打印匹配数量和公共字母字符串
          System.out.println("THERE WERE " + numMatches + " MATCHES AND THE COMMON LETTERS WERE..."
                             + commonString);

          // 打印从精确匹配的字母中得到的信息
          System.out.println("FROM THE EXACT LETTER MATCHES, YOU KNOW................" + exactString);

          // 胜利条件
          if (exactString.equals(secretWord)) {
            // 如果 exactString 等于 secretWord，则设置下一步为用户获胜，并继续执行
            nextStep = Step.USER_WINS;
            continue;
          }
          // 如果没有匹配项
          if (numMatches <= 1) {
            System.out.println("");
            System.out.println("如果你放弃了，下一个猜测输入'?'");
          }

          System.out.println("");
          nextStep = Step.MAKE_GUESS;  // 设置下一步为进行猜测
          break;

        case USER_WINS:

          System.out.println("你猜对了单词。你猜了 " + numGuesses + " 次！");
          System.out.println("");

          System.out.print("想再玩一次吗？ ");
          userResponse = scan.nextLine();

          if (userResponse.toUpperCase().equals("YES")) {
            nextStep = Step.INITIALIZE;  // 重新开始游戏
          } else {
            return;  // 退出游戏
          }
          break;

        default:
          System.out.println("INVALID STEP");  // 打印无效步骤
          break;

      }

    }  // 结束外部 while 循环

  }  // 结束 startGame 方法

  public static void main(String[] args) {

    Word word = new Word();
    word.play();  // 调用 play 方法开始游戏
  }  // 结束 main 方法

}  // 结束 Word 类
```