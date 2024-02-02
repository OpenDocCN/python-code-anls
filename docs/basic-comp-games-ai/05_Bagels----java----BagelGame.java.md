# `basic-computer-games\05_Bagels\java\BagelGame.java`

```py
/******************************************************************************
*
* Encapsulates all the state and game logic for one single game of Bagels
*
* Used by Bagels.java
*
* Jeff Jetton, 2020
*
******************************************************************************/

import java.util.ArrayList;
import java.util.Collections;
importjava.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

public class BagelGame {

  public static final String CORRECT = "FERMI FERMI FERMI";  // 定义常量字符串，表示猜对时的提示
  public static final int MAX_GUESSES = 20;  // 定义常量整数，表示最大猜测次数

  enum GameState {  // 定义枚举类型，表示游戏状态
      RUNNING,  // 游戏进行中
      WON,  // 游戏胜利
      LOST  // 游戏失败
    }

  private GameState      state = GameState.RUNNING;  // 初始化游戏状态为进行中
  private List<Integer>  secretNum;  // 用于存储秘密数字的列表
  private int            guessNum = 1;  // 初始化猜测次数为1

  public BagelGame() {
    // No-arg constructor for when you don't need to set the seed
    this(new Random());  // 调用带有 Random 参数的构造函数，使用默认的随机数生成器
  }

  public BagelGame(long seed) {
    // Setting the seed as a long value
    this(new Random(seed));  // 调用带有 long 参数的构造函数，使用指定的种子值创建随机数生成器
  }

  public BagelGame(Random rand) {
    // This is the "real" constructor, which expects an instance of
    // Random to use for shuffling the digits of the secret number.

    // Since the digits cannot repeat in our "number", we can't just
    // pick three random 0-9 integers. Instead, we'll treat it like
    // a deck of ten cards, numbered 0-9.
    List<Integer> digits = new ArrayList<Integer>(10);  // 创建一个包含10个元素的整数列表
    // The 10 specified initial allocation, not actual size,
    // which is why we add rather than set each element...
    for (int i = 0; i < 10; i++) {  // 循环10次，向列表中添加0-9的数字
      digits.add(i);
    }
    // Collections offers a handy-dandy shuffle method. Normally it
    // uses a fresh Random class PRNG, but we're supplying our own
    // to give us controll over whether or not we set the seed
    Collections.shuffle(digits, rand);  // 使用指定的随机数生成器对列表进行洗牌

    // Just take the first three digits
    secretNum = digits.subList(0, 3);  // 从洗牌后的列表中取出前三个数字作为秘密数字
  }

  public boolean isOver() {
    return state != GameState.RUNNING;  // 判断游戏是否结束
  }

  public boolean isWon() {
  // 返回游戏状态是否为胜利状态
  return state == GameState.WON;
}

public int getGuessNum() {
  // 返回猜测次数
  return guessNum;
}

public String getSecretAsString() {
  // 将秘密数字转换为一个三位字符的字符串
  String secretString = "";
  for (int n : secretNum) {
    secretString += n;
  }
  return secretString;
}

@Override
public String toString() {
  // 用于调试目的的快速报告游戏状态
  String s = "Game is " + state + "\n";
  s += "Current Guess Number: " + guessNum + "\n";
  s += "Secret Number: " + secretNum;
  return s;
}

public String validateGuess(String guess) {
  // 检查传入的字符串，如果是有效的猜测（即恰好包含三个数字字符），则返回null
  // 如果无效，则返回一个“错误”字符串以显示给用户
  String error = "";

  if (guess.length() == 3) {
    // 长度正确。所有字符都是数字吗？
    try {
      Integer.parseInt(guess);
    } catch (NumberFormatException ex) {
      error = "What?";
    }
    if (error == "") {
      // 通过将每个字符放入集合中，检查唯一数字
      Set<Character> uniqueDigits = new HashSet<Character>();
      for (int i = 0; i < guess.length(); i++){
        uniqueDigits.add(guess.charAt(i));
      }
      if (uniqueDigits.size() != guess.length()) {
        error = "Oh, I forgot to tell you that the number I have in mind\n";
        error += "has no two digits the same.";
      }
    }
  } else {
    error = "Try guessing a three-digit number.";
  }

  return error;
}

public String makeGuess(String s) throws IllegalArgumentException {
  // 处理传入的猜测字符串（理想情况下，应该先调用validateGuess进行验证）
  // 如果有效，返回响应字符串（PICO、FERMI等）
  // 还相应地设置游戏状态（设置胜利状态或增加猜测次数）
    // 将字符串转换为整数列表，只是为了保持代码规范
    List<Integer> guess = new ArrayList<Integer>(3);
    for (int i = 0; i < 3; i++) {
      guess.add((int)s.charAt(i) - 48);
    }

    // 构建响应字符串...
    String response = "";
    // 正确的数字，但位置错误？
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
    // 正确的数字且位置正确？
    for (int i = 0; i < 3; i++) {
      if (secretNum.get(i) == guess.get(i)) {
        response += "FERMI ";
      }
    }
    // 一个都没对？
    if (response == "") {
      response = "BAGELS";
    }
    // 去掉可能在末尾的任何空格
    response = response.trim();
    // 如果正确，改变状态
    if (response.equals(CORRECT)) {
      state = GameState.WON;
    } else {
      // 如果不正确，增加猜测次数并检查是否游戏结束
      guessNum++;
      if (guessNum > MAX_GUESSES) {
        state = GameState.LOST;
      }
    }
    return response;
  }
# 闭合前面的函数定义
```