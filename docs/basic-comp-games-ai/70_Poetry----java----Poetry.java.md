# `basic-computer-games\70_Poetry\java\Poetry.java`

```py
/**
 * Game of Poetry
 * <p>
 * Based on the BASIC game of Poetry here
 * https://github.com/coding-horror/basic-computer-games/blob/main/70%20Poetry/poetry.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's BASIC game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 *
 * Converted from BASIC to Java by Darren Cardenas.
 */

public class Poetry {

  private final static double COMMA_RATE = 0.19;  // 定义逗号出现的概率
  private final static double SPACE_RATE = 0.65;  // 定义空格出现的概率
  private final static int PARAGRAPH_RATE = 20;  // 定义段落出现的概率

  private enum Step {  // 定义枚举类型 Step
    WORD_GROUP1, WORD_GROUP2, WORD_GROUP3, WORD_GROUP4, RANDOMIZE_COMMA,
    RANDOMIZE_WHITESPACE, RANDOMIZE_COUNTERS
  }

  public void play() {

    showIntro();  // 调用 showIntro 方法
    startGame();  // 调用 startGame 方法

  }  // End of method play

  private void showIntro() {

    System.out.println(" ".repeat(29) + "POETRY");  // 打印标题
    System.out.println(" ".repeat(14) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");  // 打印创意计算机的位置
    System.out.println("\n\n");  // 打印两个空行

  }  // End of method showIntro

  private void startGame() {

    int groupIndex = 0;  // 初始化词组索引
    int paragraphIndex = 0;  // 初始化段落索引
    int punctuationIndex = 0;  // 初始化标点符号索引
    int wordIndex = 1;  // 初始化单词索引

    Step nextStep = Step.WORD_GROUP1;  // 初始化下一步操作为 WORD_GROUP1

    // Begin outer while loop
    }  // End outer while loop

  }  // End of method startGame

  public static void main(String[] args) {

    Poetry poetry = new Poetry();  // 创建 Poetry 对象
    poetry.play();  // 调用 play 方法

  }  // End of method main

}  // End of class Poetry
```