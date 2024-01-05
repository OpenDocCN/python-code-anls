# `60_Mastermind\java\Mastermind.java`

```
# 导入所需的 Java 类库
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
importjava.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.Scanner;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * 将 BASIC Mastermind 游戏的 Java 版本
 *
 * 与原始 BASIC 版本的不同之处：
 *    使用数字基数转换方法将解决方案 ID 转换为颜色代码字符串。原始版本使用低效的加法和进位技术，每次进位都会增加下一个位置的颜色 ID。
# 实现对秘密代码位置数量的上限检查，以防止内存耗尽。由于计算机用于推断玩家秘密代码的算法，它会搜索整个可能的解决方案范围。这个数字可能非常大，因为它是（颜色数量）^（位置数量）。如果原始代码尝试分配系统上的所有内存，如果这个数字太大，它将乐意这样做。如果它成功地在一个大的解决方案集上分配了内存，那么通过前面提到的技术计算代码字符串也会花费太长时间。
# 在开始时提供额外的消息，提醒玩家关于“BOARD”和“QUIT”命令。
```
```python
class Mastermind:
    # 创建一个随机数生成器对象
    final Random random = new Random()
    # 一些简洁的打印方法
    static private void pf(String s, Object... o){ System.out.printf(s, o);}
    static private void pl(String s){System.out.println(s);}
    static private void pl(){System.out.println();}
  public static void main(String[] args) {
    title(); // 调用title方法，显示游戏标题
    Mastermind game = setup(); // 调用setup方法，初始化游戏
    game.play(); // 调用play方法，开始游戏
  }

  /**
   * The eight possible color codes.
   */
  private enum Color {
    B("BLACK"), W("WHITE"), R("RED"), G("GREEN"), // 定义8种颜色代码
    O("ORANGE"), Y("YELLOW"), P("PURPLE"), T("TAN");
    public final String name; // 定义颜色名称变量

    Color(String name) { // 颜色枚举的构造函数
      this.name = name; // 初始化颜色名称
    }
  }
  /**
   * 表示一个猜测以及随后的正确位置的颜色数量（黑色）和存在但不在正确位置的颜色数量（白色）。
   */
  private record Guess(int guessNum, String guess, int blacks, int whites){}


  private void play() {
    // 使用 IntStream 创建一个范围为 1 到 rounds 的流，并对每个数字调用 playRound 方法
    IntStream.rangeClosed(1,rounds).forEach(this::playRound);
    // 打印游戏结束的消息
    pl("GAME OVER");
    // 打印最终得分
    pl("FINAL SCORE: ");
    pl(getScore());
  }

  /**
   * 用于创建 Mastermind 游戏的类似构建器的模式
   * @return Mastermind 游戏对象
   */
  private static Mastermind setup() {
    // 声明一个整型变量 numOfColors
    int numOfColors;
    // 打印提示信息，要求用户输入颜色的数量
    pf("NUMBER OF COLORS? > ");
    // 调用 getPositiveNumberUpTo 方法获取用户输入的正整数，并赋值给 numOfColors
    numOfColors = getPositiveNumberUpTo(Color.values().length);
    // 声明一个整型变量 maxPositions，并初始化为 getMaxPositions 方法返回的值
    int maxPositions = getMaxPositions(numOfColors);
    // 打印提示信息，要求用户输入位置的数量，最大值为 maxPositions
    pf("NUMBER OF POSITIONS (MAX %d)? > ", maxPositions);
    // 调用 getPositiveNumberUpTo 方法获取用户输入的正整数，并赋值给 positions
    int positions = getPositiveNumberUpTo(maxPositions);
    // 打印提示信息，要求用户输入回合的数量
    pf("NUMBER OF ROUNDS? > ");
    // 调用 getPositiveNumber 方法获取用户输入的正整数，并赋值给 rounds
    int rounds = getPositiveNumber();
    // 打印提示信息，提示用户可以输入 'BOARD' 来显示之前的猜测，或者输入 'QUIT' 放弃游戏
    pl("ON YOUR TURN YOU CAN ENTER 'BOARD' TO DISPLAY YOUR PREVIOUS GUESSES,");
    pl("OR 'QUIT' TO GIVE UP.");
    // 返回一个新的 Mastermind 对象，参数为 numOfColors, positions, rounds, 10
    return new Mastermind(numOfColors, positions, rounds, 10);
  }

  /**
   * 计算允许的位置数量，以防止计算机需要检查的总可能解集过多，避免内存错误。
   *
   * 计算机猜测算法使用 BitSet，其位数限制为 2^31 (Integer.MAX_VALUE 位)。由于可能解的数量
   * 太多，可能会导致内存错误。
   */
   * 计算可以有的最大位置数，根据颜色数量和最大整数值来确定
   *
   * @param numOfColors  不同颜色的数量
   * @return             密码的最大可能位置数
   */
  private static int getMaxPositions(int numOfColors){
    return (int)(Math.log(Integer.MAX_VALUE)/Math.log(numOfColors));
  }

  // 颜色数量、位置数量、回合数、可能性数量的初始化
  final int numOfColors, positions, rounds, possibilities;
  int humanMoves, computerMoves;  // 人类和计算机的移动次数
  final BitSet solutionSet;  // 解决方案的位集合
  final Color[] colors;  // 颜色数组
  final int maxTries;  // 最大尝试次数

  // 用于记录本轮人类猜测的列表，用于BOARD命令
  final List<Guess> guesses = new ArrayList<>();

  // 用于验证用户猜测字符串的正则表达式
  final String guessValidatorRegex; // 声明一个不可变的字符串变量，用于存储猜测的验证正则表达式

  public Mastermind(int numOfColors, int positions, int rounds, int maxTries) {
    this.numOfColors = numOfColors; // 初始化颜色数量
    this.positions = positions; // 初始化位置数量
    this.rounds = rounds; // 初始化回合数
    this.maxTries = maxTries; // 初始化最大尝试次数
    this.humanMoves = 0; // 初始化玩家猜测次数
    this.computerMoves = 0; // 初始化计算机猜测次数
    String colorCodes = Arrays.stream(Color.values()) // 从Color枚举中获取颜色值
                              .limit(numOfColors) // 限制颜色数量
                              .map(Color::toString) // 将颜色值转换为字符串
                              .collect(Collectors.joining()); // 将颜色值连接成一个字符串
    // 生成用于验证猜测的正则表达式，限制颜色代码的数量和数量
    this.guessValidatorRegex = "^[" + colorCodes + "]{" + positions + "}$";
    this.colors = Color.values(); // 初始化颜色数组
    this.possibilities = (int) Math.round(Math.pow(numOfColors, positions)); // 计算可能的猜测数量
    pf("TOTAL POSSIBILITIES =% d%n", possibilities); // 打印可能的猜测数量
    this.solutionSet = new BitSet(possibilities); // 初始化用于存储可能解的BitSet
    displayColorCodes(numOfColors); // 显示颜色代码
  }

  private void playRound(int round) {
    // 打印当前回合数
    pf("ROUND NUMBER % d ----%n%n",round);
    // 玩家回合
    humanTurn();
    // 电脑回合
    computerTurn();
    // 打印得分
    pl(getScore());
  }

  private void humanTurn() {
    // 清空猜测列表
    guesses.clear();
    // 生成秘密代码
    String secretCode = generateColorCode();
    // 提示玩家猜测组合
    pl("GUESS MY COMBINATION. \n");
    int guessNumber = 1;
    // 用户输入循环
    while (true) {
      // 提示用户猜测
      pf("MOVE #%d GUESS ?", guessNumber);
      // 获取用户输入的猜测
      final String guess = getWord();
      // 如果猜测正确
      if (guess.equals(secretCode)) {
        // 将猜测结果添加到猜测列表
        guesses.add(new Guess(guessNumber, guess, positions, 0));
        // 打印猜测次数
        pf("YOU GUESSED IT IN %d MOVES!%n", guessNumber);
        humanMoves++;  # 增加玩家的移动次数
        pl(getScore());  # 打印当前得分
        return;  # 返回
      } else if ("BOARD".equals(guess)) {  # 如果玩家输入的是"BOARD"
        displayBoard();  # 显示游戏板
      } else if ("QUIT".equals(guess))  {  # 如果玩家输入的是"QUIT"
        quit(secretCode);  # 退出游戏并显示秘密代码
      } else if (!validateGuess(guess)) {  # 如果玩家输入的猜测无效
        pl(guess + " IS UNRECOGNIZED.");  # 打印出猜测无效的消息
      } else {  # 如果玩家输入的是有效的猜测
        Guess g = evaluateGuess(guessNumber, guess, secretCode);  # 评估玩家的猜测
        pf("YOU HAVE %d BLACKS AND %d WHITES.%n", g.blacks(), g.whites());  # 打印出玩家猜测的结果
        guesses.add(g);  # 将玩家的猜测结果添加到猜测列表中
        humanMoves++;  # 增加玩家的移动次数
        guessNumber++;  # 增加猜测次数
      }
      if (guessNumber > maxTries) {  # 如果猜测次数超过了最大尝试次数
        pl("YOU RAN OUT OF MOVES!  THAT'S ALL YOU GET!");  # 打印出玩家用尽了所有的尝试次数
        pl("THE ACTUAL COMBINATION WAS: " + secretCode);  # 打印出实际的秘密代码
        return;  # 返回
      }
    }
  }
  private void computerTurn(){
    while (true) {
      pl("NOW I GUESS.  THINK OF A COMBINATION.");  // 输出提示信息
      pl("HIT RETURN WHEN READY:");  // 输出提示信息
      solutionSet.set(0, possibilities);  // 将所有位设置为 true
      getInput("RETURN KEY", Scanner::nextLine, Objects::nonNull);  // 获取用户输入
      int guessNumber = 1;  // 初始化猜测次数
      while(true){
        if (solutionSet.cardinality() == 0) {  // 如果解集合中没有剩余的可能解
          // user has given wrong information, thus we have cancelled out
          // any remaining possible valid solution.
          pl("YOU HAVE GIVEN ME INCONSISTENT INFORMATION.");  // 输出提示信息
          pl("TRY AGAIN, AND THIS TIME PLEASE BE MORE CAREFUL.");  // 输出提示信息
          break;  // 退出内层循环
        }
        // Randomly pick an untried solution.
        int solution = solutionSet.nextSetBit(generateSolutionID());  // 随机选择一个未尝试的解
        if (solution == -1) {  // 如果没有找到未尝试的解
          solution = solutionSet.nextSetBit(0);  // 从头开始查找未尝试的解
        }
        // 根据猜测的解法ID获取颜色编码
        String guess = solutionIdToColorCode(solution);
        // 打印猜测的结果
        pf("MY GUESS IS: %s  BLACKS, WHITES ? ",guess);
        // 获取黑白棋子的数量
        int[] bAndWPegs = getPegCount(positions);
        // 如果猜测正确，打印猜测次数并返回
        if (bAndWPegs[0] == positions) {
          pf("I GOT IT IN % d MOVES!%n", guessNumber);
          computerMoves+=guessNumber;
          return;
        }
        // 错误的猜测，首先从解决方案集中移除这个猜测
        solutionSet.clear(solution);
        int index = 0;
        // 循环遍历剩余的解决方案集，标记任何不完全匹配用户对我们猜测的描述的解决方案为无效
        while ((index = solutionSet.nextSetBit(index)) != -1) {
          String solutionStr = solutionIdToColorCode(index);
          // 评估猜测的结果
          Guess possibleSolution = evaluateGuess(0, solutionStr, guess);
          // 如果可能的解决方案的黑白棋子数量与用户描述的不匹配，则将其从解决方案集中移除
          if (possibleSolution.blacks() != bAndWPegs[0] ||
              possibleSolution.whites() != bAndWPegs[1]) {
            solutionSet.clear(index);
  }
  index++;
}
guessNumber++;
```
这部分代码是一个循环的结束和递增计数器的操作。

```
// tally black and white pegs
```
这是一个注释，解释了下面函数的作用。

```
private Guess evaluateGuess(int guessNum, String guess, String secretCode) {
```
这是一个私有函数，用于评估猜测的结果。

```
int blacks = 0, whites = 0;
char[] g = guess.toCharArray();
char[] sc = secretCode.toCharArray();
```
这里定义了变量并将字符串转换为字符数组。

```
char visited = 0x8000;
```
这是一个用于标记已经计算过的位置的变量。

```
// Cycle through guess letters and check for color and position match
// with the secretCode. If both match, mark it black.
// Else cycle through remaining secretCode letters and check if color
// matches. If this matches, a preventative check must be made against
```
这是一段注释，解释了下面代码的作用。
    // 用于猜测的字母与秘密代码中相同位置的字母进行比较，如果相同则计为黑色
    for (int j = 0; j < positions; j++) {
      if (g[j] == sc[j]) {
        blacks++;
        g[j] = visited++;
        sc[j] = visited++;
      }
      // 用于猜测的字母与秘密代码中不同位置的字母进行比较，如果相同则计为白色
      for (int k = 0; k < positions; k++) {
        if (g[j] == sc[k] && g[k] != sc[k]) {
          whites++;
          g[j] = visited++;
          sc[k] = visited++;
        }
      }
    }
    // 返回一个包含猜测次数、猜测结果、黑色和白色数量的 Guess 对象
    return new Guess(guessNum, guess, blacks, whites);
  }

  // 验证猜测是否符合规则
  private boolean validateGuess(String guess) {
    return guess.length() == positions && guess.matches(guessValidatorRegex);
  }
  // 检查猜测的长度是否等于位置数，并且猜测是否符合猜测验证正则表达式

  private String getScore() {
    return "SCORE:%n\tCOMPUTER \t%d%n\tHUMAN \t%d%n"
        .formatted(computerMoves, humanMoves);
  }
  // 返回得分字符串，包括计算机和人类的移动次数

  private void printGuess(Guess g){
    pf("% 3d%9s% 15d% 10d%n",g.guessNum(),g.guess(),g.blacks(),g.whites());
  }
  // 打印猜测的序号、猜测、黑色猜中数、白色猜中数

  private void displayBoard() {
    pl();
    pl("BOARD");
    pl("MOVE     GUESS          BLACK     WHITE");
    guesses.forEach(this::printGuess);
    pl();
  }
  // 显示游戏板，包括移动、猜测、黑色猜中数、白色猜中数
  private void quit(String secretCode) {
    // 打印退出消息和秘密代码
    pl("QUITTER!  MY COMBINATION WAS: " + secretCode);
    // 打印再见消息
    pl("GOOD BYE");
    // 退出程序
    System.exit(0);
  }

  /**
   * 生成一组随机的颜色代码。
   */
  private String generateColorCode() {
    // 生成一个解决方案ID
    int solution = generateSolutionID();
    // 将解决方案ID转换为颜色代码
    return solutionIdToColorCode(solution);
  }

  /**
   * 从在构造时创建的总可能解决方案数量中随机选择一个。
   *
   * @return 可能解决方案中的一个
   */
  private int generateSolutionID() {
    // 生成一个随机的解决方案ID，范围在0到possibilities之间
    return random.nextInt(0, this.possibilities);
  }

  /**
   * 给定秘密代码中颜色和位置的数量，将一个排列（解决方案编号）解码为表示彩色插销的字母字符串。
   *
   * 该模式可以轻松地解码为一个以`numOfColors`和`positions`为基数的数字，表示为数字。例如，如果numOfColors为5，positions为3，则将模式转换为一个基数为5的数字，有三位数字。然后，每个数字映射到特定的颜色。
   *
   * @param solution 众多可能解决方案中的一个
   * @return 表示此解决方案颜色组合的字符串。
   */
  private String solutionIdToColorCode(final int solution) {
    StringBuilder secretCode = new StringBuilder();
    int pos = possibilities;
```
```java
    // ...
  }
```

在这个示例中，我们为给定的Java代码添加了注释，解释了每个方法的作用和参数的含义。这样做有助于其他程序员理解代码的功能和实现细节。
    int remainder = solution;  // 声明一个变量remainder，用于存储solution的余数
    for (int i = positions - 1; i > 0; i--) {  // 循环，从positions-1开始，到大于0结束
      pos = pos / numOfColors;  // 计算pos的值
      secretCode.append(colors[remainder / pos].toString());  // 将colors数组中remainder/pos位置的元素转换为字符串并添加到secretCode中
      remainder = remainder % pos;  // 计算remainder的值
    }
    secretCode.append(colors[remainder].toString());  // 将colors数组中remainder位置的元素转换为字符串并添加到secretCode中
    return secretCode.toString();  // 返回secretCode的字符串形式
  }

  private static void displayColorCodes(int numOfColors) {  // 定义一个静态方法displayColorCodes，参数为numOfColors
    pl("\n\nCOLOR     LETTER\n=====     ======");  // 打印标题
    Arrays.stream(Color.values())  // 将Color枚举类型转换为流
          .limit(numOfColors)  // 限制流的元素数量为numOfColors
          .map(c -> c.name + " ".repeat(13 - c.name.length()) + c)  // 将每个元素的名称和对齐的空格以及枚举值转换为字符串
          .forEach(Mastermind::pl);  // 对每个字符串执行pl方法
    pl();pl();  // 打印两个空行
  }

  private static void title() {  // 定义一个静态方法title
    pl("""    
                                  MASTERMIND
                   CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY%n%n%n
    """);
  }
```
这段代码是一个打印函数，用于在游戏开始时显示游戏的标题和信息。

```
  /////////////////////////////////////////////////////////
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
```
这段代码是一个用户输入函数的注释，描述了输入函数的作用和参数。函数的作用是从特定的输入函数中调用基本输入函数。在用户输入意外或无效时重新提示用户输入，并在获取所需输入后丢弃用户输入行中的任何剩余输入。函数的参数包括描述输入类型的描述符，对Scanner对象执行的方法来解析输入类型的提取器，以及输入满足最小验证的测试。函数返回这一行用户输入的输入类型。
    */
    // 定义一个静态方法，接受一个描述符、一个从Scanner到T的函数和一个从T到布尔值的函数
    private static <T> T getInput(String descriptor,
                                Function<Scanner, T> extractor,
                                Predicate<T> conditional) {

      // 创建一个Scanner对象，用于从标准输入读取数据
      Scanner scanner = new Scanner(System.in);
      // 无限循环，直到条件满足才返回输入值
      while (true) {
        try {
          // 使用提供的函数从Scanner中提取输入值
          T input = extractor.apply(scanner);
          // 如果输入值满足条件，则返回该值
          if (conditional.test(input)) {
            return input;
          }
        } catch (Exception ex) {
          try {
            // 如果出现异常，清空输入行的剩余内容，以便重新提示用户输入
            scanner.nextLine();
          } catch (Exception ns_ex) {
            // 如果输入已关闭或者接收到异常
          // 输出“输入结束，停止程序”的提示信息
          pl("END OF INPUT, STOPPING PROGRAM.");
          // 退出程序
          System.exit(1);
        }
      }
      // 输出“!预期的输入 - 重试输入行”的提示信息
      pf("!%s EXPECTED - RETRY INPUT LINE%n? ", descriptor);
    }
  }

  // 获取正整数
  private static int getPositiveNumber() {
    return getInput("NUMBER", Scanner::nextInt, num -> num > 0);
  }

  // 获取小于等于给定值的正整数
  private static int getPositiveNumberUpTo(long to) {
    return getInput(
        "NUMBER FROM 1 TO " + to,
        Scanner::nextInt,
        num -> num > 0 && num <= to);
  }
  private static int[] getPegCount(int upperBound) {
    int[] nums = {Integer.MAX_VALUE, Integer.MAX_VALUE};  # 创建一个包含两个整数的数组，初始值为整数的最大值
    while (true) {  # 进入无限循环
      String input = getInput(  # 调用 getInput函数，获取用户输入的字符串
          "NUMBER, NUMBER",  # 提示用户输入的格式
          Scanner::nextLine,  # 使用Scanner类的nextLine方法获取用户输入
          s -> s.matches("\\d+[\\s,]+\\d+$"));  # 使用Lambda表达式检查用户输入是否符合指定格式
      String[] numbers = input.split("[\\s,]+");  # 使用正则表达式将用户输入的字符串分割成数组
      nums[0] = Integer.parseInt(numbers[0].trim());  # 将第一个数字字符串转换为整数并赋值给数组的第一个元素
      nums[1] = Integer.parseInt(numbers[1].trim());  # 将第二个数字字符串转换为整数并赋值给数组的第二个元素
      if (nums[0] <= upperBound && nums[1] <= upperBound &&  # 检查两个数字是否都小于等于上限并且大于等于0
          nums[0] >= 0 && nums[1] >= 0) {
        return nums;  # 如果满足条件，返回数组
      }
      pf("NUMBERS MUST BE FROM 0 TO %d.%n? ", upperBound);  # 打印错误提示信息
    }
  }

  private static String getWord() {
    return getInput("WORD", Scanner::next, word -> !"".equals(word));
  }
}
```

注释：
- 使用 getInput 方法获取用户输入的单词
- 参数 "WORD" 表示输入类型为单词
- Scanner::next 表示使用 Scanner 类的 next 方法来获取输入
- word -> !"".equals(word) 表示输入的单词不为空
- 方法结束
```