# `19_Bunny\java\src\Bunny.java`

```
import java.util.ArrayList;  # 导入 ArrayList 类
import java.util.Arrays;  # 导入 Arrays 类

/**
 * Bunny
 * <p>
 * Based on the Basic program Bunny
 * https://github.com/coding-horror/basic-computer-games/blob/main/19%20Bunny/bunny.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's Basic game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 */

public class Bunny {

    // First 4 elements are the text BUNNY, so skip those
    public static final int REAL_DATA_START_POS = 5;  # 定义常量 REAL_DATA_START_POS，表示真实数据的起始位置为第 5 个元素

    // Data for characters is not representative of three ASCII character, so we have
    // to add 64 to it as per original program design.
```
```python
    public static final int CONVERT_TO_ASCII = 64;  # 常量，用于将数据转换为ASCII码
    public static final int EOF = 4096;  # 文件结束标志常量
    public static final int EOL = -1;  # 行结束标志常量

    # 包含绘制图片的数据
    private final ArrayList<Integer> data;

    public Bunny() {
        data = loadData();  # 调用loadData()方法加载数据并赋值给data
    }

    /**
     * 展示介绍，然后绘制图片
     */
    public void process() {

        intro();  # 调用intro()方法展示介绍

        // 数据的前5个字符拼写出BUNNY，因此将其添加到一个字符串中
        // 创建一个 StringBuilder 对象，用于构建字符串
        StringBuilder bunnyBuilder = new StringBuilder();
        // 遍历数据列表，将数据转换为字符表示并添加到 StringBuilder 中
        for (int i = 0; i < REAL_DATA_START_POS; i++) {
            // 将数据转换为字符表示并添加到 StringBuilder 中
            // Ascii A=65, B=66 - see loadData method
            bunnyBuilder.append(Character.toChars(data.get(i) + CONVERT_TO_ASCII));
        }

        // 将 StringBuilder 中的内容转换为字符串
        String bunny = bunnyBuilder.toString();

        // 设置指针指向实际数据的起始位置
        int pos = REAL_DATA_START_POS;
        // 记录上一个位置的指针
        int previousPos = 0;

        // 循环直到遇到表示文件结束的数字
        while (true) {
            // 获取要开始绘制的位置
            int first = data.get(pos);
            // 如果遇到文件结束标志，则跳出循环
            if (first == EOF) {
                break;
            }
            if (first == EOL) { // 如果当前元素是换行符
                System.out.println(); // 输出换行
                previousPos = 0; // 重置上一个输出位置为0
                // 移动到 ArrayList 中的下一个元素
                pos++;
                continue; // 继续下一次循环
            }

            // 因为我们不使用屏幕定位，所以我们只需添加适当数量的空格，从我们想要的位置到上次输出的位置
            System.out.print(addSpaces(first - previousPos)); // 输出适当数量的空格

            // 保存当前位置，供下一次循环使用
            previousPos = first;

            // 移动到下一个元素
            pos++;
            // 这是我们想要停止绘制的位置
            int second = data.get(pos);
            // 现在我们循环遍历要绘制的字符数量，使用起始点和结束点
            for (int i = first; i <= second; i++) {
                // 循环实际字符数量，但使用取余运算符确保我们只使用兔子字符串中的字符
                System.out.print(bunny.charAt(i % bunny.length()));
                // 前进到下一个位置
                previousPos += 1;
            }
            // 指向下一个数据元素
            pos++;
        }

        System.out.println();

    }

    private void intro() {
        System.out.println(addSpaces(33) + "BUNNY");
        System.out.println(addSpaces(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println();
    }
```
这段代码用于打印一行文本，其中调用了addSpaces函数来添加空格。

```
    /**
     * Return a string of x spaces
     *
     * @param spaces number of spaces required
     * @return String with number of spaces
     */
    private String addSpaces(int spaces) {
        char[] spacesTemp = new char[spaces];
        Arrays.fill(spacesTemp, ' ');
        return new String(spacesTemp);
    }
```
这是一个用于返回指定数量空格的函数，它接受一个整数参数，返回一个包含指定数量空格的字符串。

```
    /**
     * Original Basic program had the data in DATA format.
     * We're importing all the data into an array for ease of processing.
     * Format of data is
```
这是一个注释块，解释了原始Basic程序中的数据格式以及将数据导入数组的目的。
# characters 0-4 is the letters that will be used in the output. 64 + the value represents the ASCII character
# ASCII code 65 = A, 66 = B, etc.  so 2+64=66 (B), 21+64=85 (U) and so on.
# Then we next have pairs of numbers.
# Looking at the this data
# 1,2,-1,0,2,45,50,-1
# That reads as
# 1,2 = draw characters - in this case BU
# -1 = go to a new line
# 0,2 = DRAW BUN
# 45,50 = DRAW BUNNYB starting at position 45
# and so on.
# 4096 is EOF

# @return ArrayList of type Integer containing the data
def loadData():
    theData = []  # Initialize an empty list to store the data

    # This is the data faithfully added from the original basic program.
        // Notes:
        // The following lines add integers to the list theData
        // These integers seem to represent some kind of data or code
        // It's not clear what the purpose of this data is without more context
        theData.addAll(Arrays.asList(2, 21, 14, 14, 25));
        theData.addAll(Arrays.asList(1, 2, -1, 0, 2, 45, 50, -1, 0, 5, 43, 52, -1, 0, 7, 41, 52, -1));
        theData.addAll(Arrays.asList(1, 9, 37, 50, -1, 2, 11, 36, 50, -1, 3, 13, 34, 49, -1, 4, 14, 32, 48, -1));
        theData.addAll(Arrays.asList(5, 15, 31, 47, -1, 6, 16, 30, 45, -1, 7, 17, 29, 44, -1, 8, 19, 28, 43, -1));
        theData.addAll(Arrays.asList(9, 20, 27, 41, -1, 10, 21, 26, 40, -1, 11, 22, 25, 38, -1, 12, 22, 24, 36, -1));
        theData.addAll(Arrays.asList(13, 34, -1, 14, 33, -1, 15, 31, -1, 17, 29, -1, 18, 27, -1));
        theData.addAll(Arrays.asList(19, 26, -1, 16, 28, -1, 13, 30, -1, 11, 31, -1, 10, 32, -1));
        theData.addAll(Arrays.asList(8, 33, -1, 7, 34, -1, 6, 13, 16, 34, -1, 5, 12, 16, 35, -1));
        theData.addAll(Arrays.asList(4, 12, 16, 35, -1, 3, 12, 15, 35, -1, 2, 35, -1, 1, 35, -1));
        theData.addAll(Arrays.asList(2, 34, -1, 3, 34, -1, 4, 33, -1, 6, 33, -1, 10, 32, 34, 34, -1));
        theData.addAll(Arrays.asList(14, 17, 19, 25, 28, 31, 35, 35, -1, 15, 19, 23, 30, 36, 36, -1));
        theData.addAll(Arrays.asList(14, 18, 21, 21, 24, 30, 37, 37, -1, 13, 18, 23, 29, 33, 38, -1));
        theData.addAll(Arrays.asList(12, 29, 31, 33, -1, 11, 13, 17, 17, 19, 19, 22, 22, 24, 31, -1));
        theData.addAll(Arrays.asList(10, 11, 17, 18, 22, 22, 24, 24, 29, 29, -1));
        theData.addAll(Arrays.asList(22, 23, 26, 29, -1, 27, 29, -1, 28, 29, -1, 4096));

        return theData;
# 定义一个名为main的函数，参数为args
def main(args):
    # 创建一个名为bunny的Bunny对象
    bunny = Bunny()
    # 调用Bunny对象的process方法
    bunny.process()
```
```python
# 调用main函数，并传入参数args
main(args)
```