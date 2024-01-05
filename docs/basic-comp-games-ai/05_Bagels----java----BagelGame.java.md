# `05_Bagels\java\BagelGame.java`

```
/******************************************************************************
*
* Encapsulates all the state and game logic for one single game of Bagels
* 封装了一个Bagels游戏的所有状态和游戏逻辑
* Used by Bagels.java
* 被Bagels.java使用
* Jeff Jetton, 2020
*
******************************************************************************/

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

public class BagelGame {

  public static final String CORRECT = "FERMI FERMI FERMI";
  // 定义一个常量字符串CORRECT，值为"FERMI FERMI FERMI"
  public static final int MAX_GUESSES = 20; // 设置最大猜测次数为20

  enum GameState { // 定义游戏状态枚举类型
      RUNNING, // 游戏进行中
      WON, // 游戏胜利
      LOST // 游戏失败
    }

  private GameState      state = GameState.RUNNING; // 初始化游戏状态为进行中
  private List<Integer>  secretNum; // 用于存储秘密数字的列表
  private int            guessNum = 1; // 初始化猜测次数为1

  public BagelGame() { // 无参构造函数，当不需要设置种子时使用
    this(new Random()); // 调用带有Random参数的构造函数
  }

  public BagelGame(long seed) { // 带有长整型种子参数的构造函数
    this(new Random(seed)); // 调用带有Random参数的构造函数，设置种子值
  }

  public BagelGame(Random rand) {
    // This is the "real" constructor, which expects an instance of
    // Random to use for shuffling the digits of the secret number.

    // Since the digits cannot repeat in our "number", we can't just
    // pick three random 0-9 integers. Instead, we'll treat it like
    // a deck of ten cards, numbered 0-9.
    List<Integer> digits = new ArrayList<Integer>(10);
    // The 10 specified initial allocation, not actual size,
    // which is why we add rather than set each element...
    for (int i = 0; i < 10; i++) {
      digits.add(i); // 将数字 0-9 添加到列表中
    }
    // Collections offers a handy-dandy shuffle method. Normally it
    // uses a fresh Random class PRNG, but we're supplying our own
    // to give us controll over whether or not we set the seed
    Collections.shuffle(digits, rand); // 使用指定的随机数生成器对列表进行洗牌
    // 只取前三位数字
    secretNum = digits.subList(0, 3);
  }

  public boolean isOver() {
    // 返回游戏状态是否不是运行中
    return state != GameState.RUNNING;
  }

  public boolean isWon() {
    // 返回游戏状态是否是胜利
    return state == GameState.WON;
  }

  public int getGuessNum() {
    // 返回猜测的数字
    return guessNum;
  }

  public String getSecretAsString() {
    // 将秘密数字转换为一个三位字符的字符串
    String secretString = "";
    for (int n : secretNum) {
      secretString += n;  // 将变量 n 的值添加到 secretString 变量中
    }
    return secretString;  // 返回拼接后的 secretString 变量
  }

  @Override
  public String toString() {
    // 用于快速报告游戏状态以进行调试
    String s = "Game is " + state + "\n";  // 将游戏状态和换行符拼接到 s 变量中
    s += "Current Guess Number: " + guessNum + "\n";  // 将当前猜测次数和换行符拼接到 s 变量中
    s += "Secret Number: " + secretNum;  // 将秘密数字和 s 变量拼接
    return s;  // 返回拼接后的 s 变量
  }

  public String validateGuess(String guess) {
    // 检查传入的字符串，如果是有效的猜测（即，正好三个数字字符），则返回 null
    // 如果无效，则返回一个“错误”字符串以显示给用户
    String error = "";  // 初始化 error 变量为空字符串
    if (guess.length() == 3) {  // 检查猜测的数字是否是3位数
      // Correct length. Are all the characters numbers?
      try {
        Integer.parseInt(guess);  // 尝试将猜测的字符串转换为整数，如果不能转换则抛出NumberFormatException异常
      } catch (NumberFormatException ex) {
        error = "What?";  // 如果转换失败，将错误信息设置为"What?"
      }
      if (error == "") {  // 如果没有错误信息
        // Check for unique digits by placing each character in a set
        Set<Character> uniqueDigits = new HashSet<Character>();  // 创建一个存放唯一数字的集合
        for (int i = 0; i < guess.length(); i++){  // 遍历猜测的数字的每一位
          uniqueDigits.add(guess.charAt(i));  // 将每一位数字添加到集合中
        }
        if (uniqueDigits.size() != guess.length()) {  // 如果集合中的数字个数不等于猜测的数字长度
          error = "Oh, I forgot to tell you that the number I have in mind\n";  // 设置错误信息为特定提示
          error += "has no two digits the same.";  // 添加额外的错误信息
        }
      }
    } else {
      error = "Try guessing a three-digit number.";  // 如果猜测的数字不是3位数，设置错误信息为提示信息
    }

    return error;
  }
```
这部分代码是一个方法的结束和一个错误返回。

```
  public String makeGuess(String s) throws IllegalArgumentException {
    // Processes the passed guess string (which, ideally, should be
    // validated by previously calling validateGuess)
    // Return a response string (PICO, FERMI, etc.) if valid
    // Also sets game state accordingly (sets win state or increments
    // number of guesses)
```
这部分代码是一个名为makeGuess的公共方法，它接受一个字符串参数s，并且声明可能抛出IllegalArgumentException异常。注释解释了该方法的作用，即处理传递的猜测字符串，并且应该通过之前调用validateGuess进行验证。如果有效，返回一个响应字符串（PICO，FERMI等），并相应地设置游戏状态（设置赢得状态或增加猜测次数）。

```
    // Convert string to integer list, just to keep things civil
    List<Integer> guess = new ArrayList<Integer>(3);
    for (int i = 0; i < 3; i++) {
      guess.add((int)s.charAt(i) - 48);
    }
```
这部分代码将字符串转换为整数列表，以便保持事情有序。它创建了一个名为guess的整数列表，并将字符串s的每个字符转换为整数并添加到列表中。

```
    // Build response string...
    String response = "";
```
这部分代码声明了一个名为response的空字符串，用于构建响应字符串。
    # 正确的数字，但位置错误？
    for (int i = 0; i < 2; i++) {
      if (secretNum.get(i) == guess.get(i+1)) {
        response += "PICO ";
      }
      if (secretNum.get(i+1) == guess.get(i)) {
        response += "PICO ";
      }
    }
    if (secretNum.get(0) == guess.get(2)) {
      response += "PICO ";
    }
    if (secretNum.get(2) == guess.get(0)) {
      response += "PICO ";
    }
    # 正确的数字且位置正确？
    for (int i = 0; i < 3; i++) {
      if (secretNum.get(i) == guess.get(i)) {
        response += "FERMI ";
      }
    // 如果响应为空，则将其设置为"BAGELS"
    if (response == "") {
      response = "BAGELS";
    }
    // 去除可能存在的末尾空格
    response = response.trim();
    // 如果答案正确，则改变游戏状态为WON
    if (response.equals(CORRECT)) {
      state = GameState.WON;
    } else {
      // 如果答案不正确，则增加猜测次数并检查是否游戏结束
      guessNum++;
      if (guessNum > MAX_GUESSES) {
        state = GameState.LOST;
      }
    }
    // 返回响应
    return response;
  }
# 关闭 ZIP 对象
zip.close()  # 关闭 ZIP 对象，释放资源，避免内存泄漏。
```