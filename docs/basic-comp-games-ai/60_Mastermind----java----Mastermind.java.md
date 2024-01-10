# `basic-computer-games\60_Mastermind\java\Mastermind.java`

```
import java.util.ArrayList;  // 导入 ArrayList 类
import java.util.Arrays;  // 导入 Arrays 类
import java.util.BitSet;  // 导入 BitSet 类
import java.util.List;  // 导入 List 类
import java.util.Objects;  // 导入 Objects 类
import java.util.Random;  // 导入 Random 类
import java.util.Scanner;  // 导入 Scanner 类
import java.util.function.Function;  // 导入 Function 函数接口
import java.util.function.Predicate;  // 导入 Predicate 函数接口
import java.util.stream.Collectors;  // 导入 Collectors 类
import java.util.stream.IntStream;  // 导入 IntStream 类

/**
 * 一个基于 BASIC 的 Mastermind 游戏的 Java 版本。
 *
 * 与原始 BASIC 版本的不同之处：
 *    使用数字基数转换方法将解决方案 ID 转换为颜色代码字符串。原始版本使用了低效的加法和进位技术，每次进位都会增加下一个位置的颜色 ID。
 *
 *    实现了对秘密代码中位置数量的上限检查，以防止内存耗尽。因为计算机用于推断玩家秘密代码的算法会搜索整个可能的解决方案范围。这个范围可能非常大，因为它是（颜色数量）^（位置数量）。如果这个数字太大，原始版本会愉快地尝试分配系统上的所有内存。如果它成功地在一个大的解决方案集上分配了内存，那么它也会花费太长时间来通过前面提到的技术计算代码字符串。
 *
 *    在开始时会额外给出一条消息，提醒玩家关于 BOARD 和 QUIT 命令。
 */
public class Mastermind {
  final Random random = new Random();  // 创建 Random 对象
  // 一些不太冗长的打印方法
  static private void pf(String s, Object... o){ System.out.printf(s, o);}  // 打印格式化字符串
  static private void pl(String s){System.out.println(s);}  // 打印字符串
  static private void pl(){System.out.println();}  // 打印空行

  public static void main(String[] args) {
    title();  // 调用 title 方法
    Mastermind game = setup();  // 创建 Mastermind 对象并调用 setup 方法
    game.play();  // 调用 play 方法
  }

  /**
   * 八种可能的颜色代码。
   */
  private enum Color {
    B("BLACK"), W("WHITE"), R("RED"), G("GREEN"),  // 定义枚举类型 Color，包含四种颜色
    // 定义枚举类型，表示颜色
    O("ORANGE"), Y("YELLOW"), P("PURPLE"), T("TAN");
    // 声明枚举类型的属性
    public final String name;

    // 枚举类型的构造函数
    Color(String name) {
      this.name = name;
    }
  }

  /**
   * 表示一个猜测以及随后在正确位置的颜色数量（黑色）和存在但不在正确位置的颜色数量（白色）
   */
  private record Guess(int guessNum, String guess, int blacks, int whites){}


  private void play() {
    // 使用 IntStream 迭代游戏回合
    IntStream.rangeClosed(1,rounds).forEach(this::playRound);
    // 输出游戏结束信息
    pl("GAME OVER");
    pl("FINAL SCORE: ");
    // 输出最终得分
    pl(getScore());
  }

  /**
   * 用于创建 Mastermind 游戏的构建器模式
   * @return Mastermind 游戏对象
   */
  private static Mastermind setup() {
    int numOfColors;
    // 获取用户输入的颜色数量
    pf("NUMBER OF COLORS? > ");
    numOfColors = getPositiveNumberUpTo(Color.values().length);
    // 获取用户输入的最大位置数量
    int maxPositions = getMaxPositions(numOfColors);
    pf("NUMBER OF POSITIONS (MAX %d)? > ", maxPositions);
    int positions = getPositiveNumberUpTo(maxPositions);
    // 获取用户输入的回合数量
    pf("NUMBER OF ROUNDS? > ");
    int rounds = getPositiveNumber();
    // 输出游戏提示信息
    pl("ON YOUR TURN YOU CAN ENTER 'BOARD' TO DISPLAY YOUR PREVIOUS GUESSES,");
    pl("OR 'QUIT' TO GIVE UP.");
    // 创建并返回 Mastermind 游戏对象
    return new Mastermind(numOfColors, positions, rounds, 10);
  }

  /**
   * 计算允许的位置数量，以防止计算机需要检查的总可能解决方案集合过多，以及防止内存错误。
   *
   * 计算机猜测算法使用 BitSet，其位数限制为 2^31（Integer.MAX_VALUE 位）。由于任何 Mastermind 游戏的可能解决方案数量为 (numColors) ^ (numPositions)，我们需要通过找到 Log|base-NumOfColors|(2^31) 来找到最大位置数量。
   *
   * @param numOfColors  不同颜色的数量
   * @return             密码中的最大位置数量
   */
  private static int getMaxPositions(int numOfColors){
    // 返回最大整数除以颜色数量的对数
    return (int)(Math.log(Integer.MAX_VALUE)/Math.log(numOfColors));
  }

  final int numOfColors, positions, rounds, possibilities;
  int humanMoves, computerMoves;
  final BitSet solutionSet;
  final Color[] colors;
  final int maxTries;

  // 用于记录每轮中玩家猜测的列表，用于BOARD命令
  final List<Guess> guesses = new ArrayList<>();

  // 用于验证用户猜测字符串的正则表达式
  final String guessValidatorRegex;

  public Mastermind(int numOfColors, int positions, int rounds, int maxTries) {
    this.numOfColors = numOfColors;
    this.positions = positions;
    this.rounds = rounds;
    this.maxTries = maxTries;
    this.humanMoves = 0;
    this.computerMoves = 0;
    // 生成颜色代码字符串
    String colorCodes = Arrays.stream(Color.values())
                              .limit(numOfColors)
                              .map(Color::toString)
                              .collect(Collectors.joining());
    // 限制猜测的颜色代码数量和数量的正则表达式
    this.guessValidatorRegex = "^[" + colorCodes + "]{" + positions + "}$";
    this.colors = Color.values();
    this.possibilities = (int) Math.round(Math.pow(numOfColors, positions));
    pf("TOTAL POSSIBILITIES =% d%n", possibilities);
    this.solutionSet = new BitSet(possibilities);
    // 显示颜色代码
    displayColorCodes(numOfColors);
  }

  private void playRound(int round) {
    pf("ROUND NUMBER % d ----%n%n",round);
    // 进行一轮游戏
    humanTurn();
    computerTurn();
    // 显示得分
    pl(getScore());
  }

  private void humanTurn() {
    // 清空猜测列表
    guesses.clear();
    // 生成秘密代码
    String secretCode = generateColorCode();
    pl("GUESS MY COMBINATION. \n");
    int guessNumber = 1;
    while (true) {   // 用户输入循环
      pf("MOVE #%d GUESS ?", guessNumber);  // 打印猜测次数和提示信息
      final String guess = getWord();  // 获取用户输入的猜测
      if (guess.equals(secretCode)) {  // 如果猜测正确
        guesses.add(new Guess(guessNumber, guess, positions, 0));  // 将猜测结果添加到列表中
        pf("YOU GUESSED IT IN %d MOVES!%n", guessNumber);  // 打印猜测次数
        humanMoves++;  // 用户猜测次数加一
        pl(getScore());  // 打印得分
        return;  // 结束循环
      } else if ("BOARD".equals(guess)) {  // 如果用户输入为"BOARD"
        displayBoard();  // 显示游戏板
      } else if ("QUIT".equals(guess))  {  // 如果用户输入为"QUIT"
        quit(secretCode);  // 退出游戏
      } else if (!validateGuess(guess)) {  // 如果用户输入的猜测不合法
        pl(guess + " IS UNRECOGNIZED.");  // 打印提示信息
      } else {
        Guess g = evaluateGuess(guessNumber, guess, secretCode);  // 评估用户猜测
        pf("YOU HAVE %d BLACKS AND %d WHITES.%n", g.blacks(), g.whites());  // 打印猜测结果
        guesses.add(g);  // 将猜测结果添加到列表中
        humanMoves++;  // 用户猜测次数加一
        guessNumber++;  // 猜测次数加一
      }
      if (guessNumber > maxTries) {  // 如果猜测次数超过最大尝试次数
        pl("YOU RAN OUT OF MOVES!  THAT'S ALL YOU GET!");  // 打印提示信息
        pl("THE ACTUAL COMBINATION WAS: " + secretCode);  // 打印正确的密码组合
        return;  // 结束循环
      }
    }
  }

  private void computerTurn(){
    // 进入无限循环，直到条件为假
    while (true) {
      // 打印提示信息
      pl("NOW I GUESS.  THINK OF A COMBINATION.");
      pl("HIT RETURN WHEN READY:");
      // 将所有位设置为真
      solutionSet.set(0, possibilities);  // set all bits to true
      // 获取用户输入
      getInput("RETURN KEY", Scanner::nextLine, Objects::nonNull);
      // 猜测次数
      int guessNumber = 1;
      // 再次进入无限循环
      while(true){
        // 如果解集合中的位数为0
        if (solutionSet.cardinality() == 0) {
          // 用户提供了错误信息，取消了任何可能的有效解决方案
          pl("YOU HAVE GIVEN ME INCONSISTENT INFORMATION.");
          pl("TRY AGAIN, AND THIS TIME PLEASE BE MORE CAREFUL.");
          // 退出内层循环
          break;
        }
        // 随机选择一个未尝试的解决方案
        int solution = solutionSet.nextSetBit(generateSolutionID());
        if (solution == -1) {
          solution = solutionSet.nextSetBit(0);
        }
        // 将解码为颜色代码的解决方案
        String guess = solutionIdToColorCode(solution);
        // 打印猜测结果
        pf("MY GUESS IS: %s  BLACKS, WHITES ? ",guess);
        // 获取黑白色标记的数量
        int[] bAndWPegs = getPegCount(positions);
        // 如果黑白色标记的数量等于位置数
        if (bAndWPegs[0] == positions) {
          // 打印猜测次数
          pf("I GOT IT IN % d MOVES!%n", guessNumber);
          // 计算机猜测次数增加
          computerMoves+=guessNumber;
          // 返回
          return;
        }
        // 错误的猜测，从解集合中移除这个猜测
        solutionSet.clear(solution);
        int index = 0;
        // 循环遍历剩余的解集合，标记任何不完全匹配用户对我们猜测的描述的解决方案为无效
        while ((index = solutionSet.nextSetBit(index)) != -1) {
          String solutionStr = solutionIdToColorCode(index);
          Guess possibleSolution = evaluateGuess(0, solutionStr, guess);
          if (possibleSolution.blacks() != bAndWPegs[0] ||
              possibleSolution.whites() != bAndWPegs[1]) {
            solutionSet.clear(index);
          }
          index++;
        }
        guessNumber++;
      }
    }
  }

  // 计算黑白色标记的数量
  private Guess evaluateGuess(int guessNum, String guess, String secretCode) {
    int blacks = 0, whites = 0;
    // 将猜测字符串转换为字符数组
    char[] g = guess.toCharArray();
    // 将秘密代码字符串转换为字符数组
    char[] sc = secretCode.toCharArray();
    // 用于标记已经被计算为黑色或白色钉子的位置的递增数字
    char visited = 0x8000;
    // 循环遍历猜测的字母，并检查与秘密代码的颜色和位置是否匹配
    // 如果两者匹配，则标记为黑色
    // 否则循环遍历剩余的秘密代码字母，并检查颜色是否匹配
    // 如果匹配，则必须对猜测字母与秘密代码字母在这个位置进行预防性检查
    // 以防它在下一个循环中被计为黑色
    for (int j = 0; j < positions; j++) {
      if (g[j] == sc[j]) {
        blacks++;
        g[j] = visited++;
        sc[j] = visited++;
      }
      for (int k = 0; k < positions; k++) {
        if (g[j] == sc[k] && g[k] != sc[k]) {
          whites++;
          g[j] = visited++;
          sc[k] = visited++;
        }
      }
    }
    // 返回一个包含猜测结果的 Guess 对象
    return new Guess(guessNum, guess, blacks, whites);
  }

  // 验证猜测字符串是否符合要求
  private boolean validateGuess(String guess) {
    return guess.length() == positions && guess.matches(guessValidatorRegex);
  }

  // 获取得分信息
  private String getScore() {
    return "SCORE:%n\tCOMPUTER \t%d%n\tHUMAN \t%d%n"
        .formatted(computerMoves, humanMoves);
  }

  // 打印猜测结果
  private void printGuess(Guess g){
    pf("% 3d%9s% 15d% 10d%n",g.guessNum(),g.guess(),g.blacks(),g.whites());
  }
  
  // 显示游戏板
  private void displayBoard() {
    pl();
    pl("BOARD");
    pl("MOVE     GUESS          BLACK     WHITE");
    guesses.forEach(this::printGuess);
    pl();
  }

  // 退出游戏
  private void quit(String secretCode) {
    pl("QUITTER!  MY COMBINATION WAS: " + secretCode);
    pl("GOOD BYE");
    System.exit(0);
  }

  /**
   * 随机生成一组颜色代码
   */
  private String generateColorCode() {
    int solution = generateSolutionID();
  // 返回给定解决方案的颜色代码
  return solutionIdToColorCode(solution);
}

/**
 * 从在构造时创建的总可能解决方案中随机选择一个。
 *
 * @return 许多可能解决方案中的一个
 */
private int generateSolutionID() {
  return random.nextInt(0, this.possibilities);
}

/**
 * 给定秘密代码中颜色和位置的数量，将其中一个排列解码为表示彩色插销的字母字符串。
 *
 * 可以轻松地将模式解码为以 `numOfColors` 和 `positions` 为基数的数字，表示为数字。例如，如果 numOfColors 是 5，positions 是 3，那么模式将转换为一个基数为 5 的数字，有三位数。然后，每个数字映射到特定的颜色。
 *
 * @param solution 许多可能解决方案中的一个
 * @return 表示此解决方案颜色组合的字符串
 */
private String solutionIdToColorCode(final int solution) {
  StringBuilder secretCode = new StringBuilder();
  int pos = possibilities;
  int remainder = solution;
  for (int i = positions - 1; i > 0; i--) {
    pos = pos / numOfColors;
    secretCode.append(colors[remainder / pos].toString());
    remainder = remainder % pos;
  }
  secretCode.append(colors[remainder].toString());
  return secretCode.toString();
}

private static void displayColorCodes(int numOfColors) {
  pl("\n\nCOLOR     LETTER\n=====     ======");
  Arrays.stream(Color.values())
        .limit(numOfColors)
        .map(c -> c.name + " ".repeat(13 - c.name.length()) + c)
        .forEach(Mastermind::pl);
  pl();pl();
}

private static void title() {
  pl("""    
                                MASTERMIND
                 CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY%n%n%n
  """);
}

/////////////////////////////////////////////////////
// User input functions from here on

/**
 * Base input function to be called from a specific input function.
 * Re-prompts upon unexpected or invalid user input.
 * Discards any remaining input in the line of user entered input once
 * it gets what it wants.
 * @param descriptor  Describes explicit type of input expected
 * @param extractor   The method performed against a Scanner object to parse
 *                    the type of input.
 * @param conditional A test that the input meets a minimum validation.
 * @param <T>         Input type returned.
 * @return            input type for this line of user input.
 */
private static <T> T getInput(String descriptor,
                              Function<Scanner, T> extractor,
                              Predicate<T> conditional) {

  Scanner scanner = new Scanner(System.in);
  while (true) {
    try {
      T input = extractor.apply(scanner);
      if (conditional.test(input)) {
        return input;
      }
    } catch (Exception ex) {
      try {
        // If we are here then a call on the scanner was most likely unable to
        // parse the input. We need to flush whatever is leftover from this
        // line of interactive user input so that we can re-prompt for new input.
        scanner.nextLine();
      } catch (Exception ns_ex) {
        // if we are here then the input has been closed, or we received an
        // EOF (end of file) signal, usually in the form of a ctrl-d or
        // in the case of Windows, a ctrl-z.
        pl("END OF INPUT, STOPPING PROGRAM.");
        System.exit(1);
      }
    }
    pf("!%s EXPECTED - RETRY INPUT LINE%n? ", descriptor);
  }
}

private static int getPositiveNumber() {
  return getInput("NUMBER", Scanner::nextInt, num -> num > 0);
}

private static int getPositiveNumberUpTo(long to) {
  // 从用户输入获取一个介于1到to之间的数字
  return getInput(
      "NUMBER FROM 1 TO " + to,
      Scanner::nextInt,
      num -> num > 0 && num <= to);
}

// 获取一个包含两个数字的数组，这两个数字必须小于等于upperBound
private static int[] getPegCount(int upperBound) {
  int[] nums = {Integer.MAX_VALUE, Integer.MAX_VALUE};
  while (true) {
    // 从用户输入获取两个数字，用逗号或空格分隔
    String input = getInput(
        "NUMBER, NUMBER",
        Scanner::nextLine,
        s -> s.matches("\\d+[\\s,]+\\d+$"));
    // 将输入的字符串分割成两个数字
    String[] numbers = input.split("[\\s,]+");
    nums[0] = Integer.parseInt(numbers[0].trim());
    nums[1] = Integer.parseInt(numbers[1].trim());
    // 如果两个数字都在0到upperBound之间，则返回这两个数字组成的数组
    if (nums[0] <= upperBound && nums[1] <= upperBound &&
        nums[0] >= 0 && nums[1] >= 0) {
      return nums;
    }
    // 如果数字不在指定范围内，则提示用户重新输入
    pf("NUMBERS MUST BE FROM 0 TO %d.%n? ", upperBound);
  }
}

// 获取一个单词作为输入
private static String getWord() {
  return getInput("WORD", Scanner::next, word -> !"".equals(word));
}
# 闭合前面的函数定义
```