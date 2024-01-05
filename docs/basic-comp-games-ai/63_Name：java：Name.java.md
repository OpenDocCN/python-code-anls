# `d:/src/tocomm/basic-computer-games\63_Name\java\Name.java`

```
# 导入 java.util.Arrays 和 java.util.Scanner 类
import java.util.Arrays;
import java.util.Scanner;

public class Name {

    # 定义一个静态方法 printempty，用于打印空行
    public static void printempty() { System.out.println(" "); }

    # 定义一个静态方法 print，用于打印指定内容
    public static void print(String toprint) { System.out.println(toprint); }

    # 主函数
    public static void main(String[] args) {
        # 打印指定内容
        print("                                          NAME");
        print("                         CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        printempty();  # 调用打印空行的方法
        printempty();  # 再次调用打印空行的方法
        print("HELLO.");  # 打印指定内容
        print("MY NAME iS CREATIVE COMPUTER.");  # 打印指定内容
        print("WHATS YOUR NAME? (FIRST AND LAST)");  # 打印指定内容

        # 创建一个 Scanner 对象，用于接收用户输入
        Scanner namesc = new Scanner(System.in);
        # 读取用户输入的字符串
        String name = namesc.nextLine();
        // 将文件名倒序排列
        String namereversed = new StringBuilder(name).reverse().toString();

        // 将文件名转换为字符数组并按字母顺序排序
        char namesorted[] = name.toCharArray();
        Arrays.sort(namesorted);

        // 打印空行
        printempty();
        // 打印倒序后的文件名
        print("THANK YOU, " + namereversed);
        // 打印空行
        printempty();
        // 打印提示信息，指出文件名倒序的错误
        print("OOPS!  I GUESS I GOT IT BACKWARDS.  A SMART");
        print("COMPUTER LIKE ME SHOULDN'T MAKE A MISTAKE LIKE THAT!");
        // 打印空行
        printempty();
        printempty();
        // 打印提示信息，指出文件名字母顺序错误
        print("BUT I JUST NOTICED YOUR LETTERS ARE OUT OF ORDER.");
        // 打印提示信息，将文件名按字母顺序排列后的结果
        print("LET'S PUT THEM IN ORDER LIKE THIS: " + new String(namesorted));
        // 打印空行
        printempty();
        printempty();
        // 打印提示信息，询问是否喜欢文件名按字母顺序排列后的结果
        print("DON'T YOU LIKE THAT BETTER?");
        printempty();  # 调用printempty函数，可能用于清空控制台输出

        Scanner agreementsc = new Scanner(System.in);  # 创建一个Scanner对象，用于从控制台读取输入
        String agreement = agreementsc.nextLine();  # 从控制台读取用户输入的内容并存储在agreement变量中

        if (agreement.equalsIgnoreCase("yes")) {  # 如果用户输入的内容忽略大小写后等于"yes"
            print("I KNEW YOU'D AGREE!!");  # 输出"I KNEW YOU'D AGREE!!"
        } else {  # 如果用户输入的内容不等于"yes"
            print("I'M SORRY YOU DON'T LIKE IT THAT WAY.");  # 输出"I'M SORRY YOU DON'T LIKE IT THAT WAY."
            printempty();  # 调用printempty函数，可能用于清空控制台输出
            print("I REALLY ENJOYED MEETING YOU, " + name);  # 输出"I REALLY ENJOYED MEETING YOU, "后跟着name变量的值
            print("HAVE A NICE DAY!");  # 输出"HAVE A NICE DAY!"
        }
    }
}
```