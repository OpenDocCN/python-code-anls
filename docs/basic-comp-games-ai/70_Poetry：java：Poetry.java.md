# `d:/src/tocomm/basic-computer-games\70_Poetry\java\Poetry.java`

```
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

  private final static double COMMA_RATE = 0.19; // 定义逗号出现的概率
  private final static double SPACE_RATE = 0.65; // 定义空格出现的概率
  private final static int PARAGRAPH_RATE = 20; // 定义段落出现的概率

  private enum Step { // 定义枚举类型Step
    WORD_GROUP1, WORD_GROUP2, WORD_GROUP3, WORD_GROUP4, RANDOMIZE_COMMA, // 定义枚举值
    RANDOMIZE_WHITESPACE, RANDOMIZE_COUNTERS
  }
  # 定义一个枚举类型，包含两个枚举值：RANDOMIZE_WHITESPACE和RANDOMIZE_COUNTERS

  public void play() {
    # 定义一个公共方法play，用于开始游戏
    showIntro();
    # 调用showIntro方法，显示游戏介绍
    startGame();
    # 调用startGame方法，开始游戏
  }  # End of method play
  # play方法结束

  private void showIntro() {
    # 定义一个私有方法showIntro，用于显示游戏介绍
    System.out.println(" ".repeat(29) + "POETRY");
    # 打印包含29个空格和"POETRY"的字符串
    System.out.println(" ".repeat(14) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
    # 打印包含14个空格和"CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"的字符串
    System.out.println("\n\n");
    # 打印两个换行符
  }  # End of method showIntro
  # showIntro方法结束

  private void startGame() {
    # 定义一个私有方法startGame，用于开始游戏
    # 初始化四个索引变量，用于记录当前处理的单词组、段落、标点和单词的位置
    int groupIndex = 0;
    int paragraphIndex = 0;
    int punctuationIndex = 0;
    int wordIndex = 1;

    # 初始化下一步操作为处理第一个单词组
    Step nextStep = Step.WORD_GROUP1;

    # 开始外部循环
    while (true) {

      # 根据下一步操作进行处理
      switch (nextStep) {

        # 处理第一个单词组
        case WORD_GROUP1:

          # 如果当前单词位置为1，打印"MIDNIGHT DREARY"，并设置下一步操作为随机化逗号
          if (wordIndex == 1) {

            System.out.print("MIDNIGHT DREARY");
            nextStep = Step.RANDOMIZE_COMMA;

          # 如果当前单词位置为2
          } else if (wordIndex == 2) {
            # 打印"FIERY EYES"
            System.out.print("FIERY EYES");
            # 设置下一步操作为随机化逗号
            nextStep = Step.RANDOMIZE_COMMA;

          } else if (wordIndex == 3) {

            # 打印"BIRD OR FIEND"
            System.out.print("BIRD OR FIEND");
            # 设置下一步操作为随机化逗号
            nextStep = Step.RANDOMIZE_COMMA;

          } else if (wordIndex == 4) {

            # 打印"THING OF EVIL"
            System.out.print("THING OF EVIL");
            # 设置下一步操作为随机化逗号
            nextStep = Step.RANDOMIZE_COMMA;

          } else if (wordIndex == 5) {

            # 打印"PROPHET"
            System.out.print("PROPHET");
            # 设置下一步操作为随机化逗号
            nextStep = Step.RANDOMIZE_COMMA;
          }
          # 退出循环
          break;
        # 如果 wordIndex 等于 1
        if (wordIndex == 1) {
            # 打印 "BEGUILING ME"
            System.out.print("BEGUILING ME");
            # 设置下一步为 RANDOMIZE_COMMA
            nextStep = Step.RANDOMIZE_COMMA;
        # 如果 wordIndex 等于 2
        } else if (wordIndex == 2) {
            # 打印 "THRILLED ME"
            System.out.print("THRILLED ME");
            # 设置下一步为 RANDOMIZE_COMMA
            nextStep = Step.RANDOMIZE_COMMA;
        # 如果 wordIndex 等于 3
        } else if (wordIndex == 3) {
            # 打印 "STILL SITTING...."
            System.out.print("STILL SITTING....");
            # 设置下一步为 RANDOMIZE_WHITESPACE
            nextStep = Step.RANDOMIZE_WHITESPACE;
        # 如果 wordIndex 等于 4
        } else if (wordIndex == 4) {
            System.out.print("NEVER FLITTING");  // 打印字符串"NEVER FLITTING"
            nextStep = Step.RANDOMIZE_COMMA;  // 将下一步的操作设置为随机化逗号

          } else if (wordIndex == 5) {  // 如果单词索引为5

            System.out.print("BURNED");  // 打印字符串"BURNED"
            nextStep = Step.RANDOMIZE_COMMA;  // 将下一步的操作设置为随机化逗号
          }
          break;

        case WORD_GROUP3:  // 如果当前单词组为WORD_GROUP3

          if (wordIndex == 1) {  // 如果单词索引为1

            System.out.print("AND MY SOUL");  // 打印字符串"AND MY SOUL"
            nextStep = Step.RANDOMIZE_COMMA;  // 将下一步的操作设置为随机化逗号

          } else if (wordIndex == 2) {  // 如果单词索引为2

            System.out.print("DARKNESS THERE");  // 打印字符串"DARKNESS THERE"
            # 设置下一步操作为随机化逗号
            nextStep = Step.RANDOMIZE_COMMA;

          } else if (wordIndex == 3) {

            # 打印"SHALL BE LIFTED"
            System.out.print("SHALL BE LIFTED");
            # 设置下一步操作为随机化逗号
            nextStep = Step.RANDOMIZE_COMMA;

          } else if (wordIndex == 4) {

            # 打印"QUOTH THE RAVEN"
            System.out.print("QUOTH THE RAVEN");
            # 设置下一步操作为随机化逗号
            nextStep = Step.RANDOMIZE_COMMA;

          } else if (wordIndex == 5) {

            # 如果标点索引不为0，则打印"SIGN OF PARTING"
            if (punctuationIndex != 0) {
              System.out.print("SIGN OF PARTING");
            }
            # 设置下一步操作为随机化逗号
            nextStep = Step.RANDOMIZE_COMMA;
          }
          break;

        case WORD_GROUP4:

          if (wordIndex == 1) {  # 如果 wordIndex 等于 1
            System.out.print("NOTHING MORE");  # 打印 "NOTHING MORE"
            nextStep = Step.RANDOMIZE_COMMA;  # 设置下一步为随机化逗号
          } else if (wordIndex == 2) {  # 如果 wordIndex 等于 2
            System.out.print("YET AGAIN");  # 打印 "YET AGAIN"
            nextStep = Step.RANDOMIZE_COMMA;  # 设置下一步为随机化逗号
          } else if (wordIndex == 3) {  # 如果 wordIndex 等于 3
            System.out.print("SLOWLY CREEPING");  # 打印 "SLOWLY CREEPING"
            nextStep = Step.RANDOMIZE_WHITESPACE;  # 设置下一步为随机化空格
          } else if (wordIndex == 4) {
            # 如果单词索引为4，打印"...EVERMORE"，并将下一步设为随机插入逗号
            System.out.print("...EVERMORE");
            nextStep = Step.RANDOMIZE_COMMA;

          } else if (wordIndex == 5) {
            # 如果单词索引为5，打印"NEVERMORE"，并将下一步设为随机插入逗号
            System.out.print("NEVERMORE");
            nextStep = Step.RANDOMIZE_COMMA;
          }
          break;

        case RANDOMIZE_COMMA:
          # 随机插入逗号
          if ((punctuationIndex != 0) && (Math.random() <= COMMA_RATE)) {
            System.out.print(",");
            punctuationIndex = 2;
          }
          nextStep = Step.RANDOMIZE_WHITESPACE;  # 设置下一步操作为随机化空格

          break;  # 跳出当前的 switch 语句

        case RANDOMIZE_WHITESPACE:  # 当前操作为随机化空格

          // Insert spaces  # 插入空格
          if (Math.random() <= SPACE_RATE) {  # 如果随机数小于等于空格率

            System.out.print(" ");  # 输出一个空格
            punctuationIndex++;  # 标点索引加一

          }
          // Insert newlines  # 插入换行
          else {

            System.out.println("");  # 输出一个换行
            punctuationIndex = 0;  # 重置标点索引为0
          }
          nextStep = Step.RANDOMIZE_COUNTERS;  # 设置下一步操作为随机化计数器
          break;  # 结束当前的 case，跳出 switch 语句

        case RANDOMIZE_COUNTERS:  # 当前 case 的标签

          wordIndex = (int)((int)(10 * Math.random()) / 2) + 1;  # 生成一个随机的 wordIndex

          groupIndex++;  # groupIndex 自增
          paragraphIndex++;  # paragraphIndex 自增

          if ((punctuationIndex == 0) && (groupIndex % 2 == 0)):  # 如果 punctuationIndex 为 0 并且 groupIndex 为偶数
            System.out.print("     ");  # 打印空格

          if (groupIndex == 1):  # 如果 groupIndex 为 1
            nextStep = Step.WORD_GROUP1;  # 设置 nextStep 为 WORD_GROUP1

          else if (groupIndex == 2):  # 如果 groupIndex 为 2
            # 如果groupIndex等于2，将下一步设置为WORD_GROUP2
            nextStep = Step.WORD_GROUP2;

          # 如果groupIndex等于3，将下一步设置为WORD_GROUP3
          } else if (groupIndex == 3) {

            nextStep = Step.WORD_GROUP3;

          # 如果groupIndex等于4，将下一步设置为WORD_GROUP4
          } else if (groupIndex == 4) {

            nextStep = Step.WORD_GROUP4;

          # 如果groupIndex等于5
          } else if (groupIndex == 5) {

            # 将groupIndex重置为0
            groupIndex = 0;
            # 打印空行
            System.out.println("");

            # 如果paragraphIndex大于PARAGRAPH_RATE
            if (paragraphIndex > PARAGRAPH_RATE) {

              # 打印空行
              System.out.println("");
              # 将punctuationIndex重置为0
              punctuationIndex = 0;
              # 将paragraphIndex重置为0
              paragraphIndex = 0;
              nextStep = Step.WORD_GROUP2;  # 设置下一步的操作为WORD_GROUP2

            } else {

              nextStep = Step.RANDOMIZE_COUNTERS;  # 如果条件不满足，设置下一步的操作为RANDOMIZE_COUNTERS
            }
          }
          break;

        default:
          System.out.println("INVALID STEP");  # 如果switch语句中的值不匹配任何case，打印"INVALID STEP"
          break;
      }

    }  // End outer while loop  # 结束外部的while循环

  }  // End of method startGame  # 结束startGame方法

  public static void main(String[] args) {  # 主方法的开始
    Poetry poetry = new Poetry();  // 创建一个名为poetry的Poetry对象
    poetry.play();  // 调用Poetry对象的play方法

  }  // End of method main  // main方法结束

}  // End of class Poetry  // Poetry类结束
```