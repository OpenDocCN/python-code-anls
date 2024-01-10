# `basic-computer-games\96_Word\java\Word.java`

```
// 导入必要的类
import java.util.Arrays;
import java.util.Scanner;

/**
 * Word 游戏
 * <p>
 * 基于 BASIC 版本的 Word 游戏
 * https://github.com/coding-horror/basic-computer-games/blob/main/96%20Word/word.bas
 * <p>
 * 注意：这个想法是在 Java 中创建一个 1970 年代 BASIC 游戏的版本，没有引入新功能 - 没有添加额外的文本、错误检查等。
 *
 * 由 Darren Cardenas 从 BASIC 转换为 Java。
 */

public class Word {

  // 可能的单词列表
  private final static String[] WORDS = {
    "DINKY", "SMOKE", "WATER", "GRASS", "TRAIN", "MIGHT",
    "FIRST", "CANDY", "CHAMP", "WOULD", "CLUMP", "DOPEY"
  };

  private final Scanner scan;  // 用于用户输入

  // 游戏步骤枚举
  private enum Step {
    INITIALIZE, MAKE_GUESS, USER_WINS
  }

  // Word 类的构造函数
  public Word() {
    scan = new Scanner(System.in);
  }  // End of constructor Word

  // 开始游戏
  public void play() {
    showIntro();
    startGame();
  }  // End of method play

  // 显示游戏介绍
  private void showIntro() {
    System.out.println(" ".repeat(32) + "WORD");
    System.out.println(" ".repeat(14) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
    System.out.println("\n\n");
    System.out.println("I AM THINKING OF A WORD -- YOU GUESS IT.  I WILL GIVE YOU");
    System.out.println("CLUES TO HELP YOU GET IT.  GOOD LUCK!!");
    System.out.println("\n");
  }  // End of method showIntro

  // 开始游戏
  private void startGame() {
    char[] commonLetters = new char[8];
    char[] exactLetters = new char[8];
    int commonIndex = 0;
    int ii = 0;  // 循环迭代器
    int jj = 0;  // 循环迭代器
    int numGuesses = 0;
    int numMatches = 0;
    int wordIndex = 0;
    Step nextStep = Step.INITIALIZE;
    String commonString = "";
    String exactString = "";
    String guessWord = "";
    String secretWord = "";
    String userResponse = "";
    // 开始外部 while 循环
  }  // End outer while loop

}  // End of method startGame

// 主方法
public static void main(String[] args) {
  Word word = new Word();
  word.play();
}  // End of method main
# 类 Word 的结束标记
```