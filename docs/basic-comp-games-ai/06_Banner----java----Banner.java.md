# `basic-computer-games\06_Banner\java\Banner.java`

```
import java.util.Scanner;  // 导入 Scanner 类，用于用户输入
import java.util.HashMap;  // 导入 HashMap 类，用于创建符号数据的映射
import java.util.Map;  // 导入 Map 接口，用于存储符号数据的映射
import java.lang.Math;  // 导入 Math 类，用于数学计算

/**
 * Game of Banner
 * <p>
 * 基于 BASIC 游戏 Banner 的 Java 版本
 * https://github.com/coding-horror/basic-computer-games/blob/main/06%20Banner/banner.bas
 * <p>
 * 注意：这个想法是在 Java 中创建 1970 年代 BASIC 游戏的版本，没有引入新功能 - 没有添加额外的文本、错误检查等。
 *
 * 由 Darren Cardenas 从 BASIC 转换为 Java。
 */

public class Banner {

  private final Scanner scan;  // 用于用户输入的 Scanner 对象

  public Banner() {

    scan = new Scanner(System.in);  // 初始化 Scanner 对象，使用标准输入流

  }  // Banner 构造函数结束

  public void play() {

    int bitIndex = 0;  // 位索引
    int centerFlag = 0;  // 中心标志
    int dataIndex = 0;  // 数据索引
    int hIndex = 0;  // 水平索引
    int horizontal = 0;  // 水平
    int index = 0;  // 索引
    int letterIndex = 0;  // 字母索引
    int tempVal = 0;  // 临时值
    int vertical = 0;  // 垂直
    int vIndex = 0;  // 垂直索引
    int writeIndex = 0;  // 写入索引

    int[] writerMap = new int[10];  // 写入映射数组
    int[] writeLimit = new int[10];  // 写入限制数组

    String centerResponse = "";  // 中心响应
    String characters = "";  // 字符串
    String letter = "";  // 字母
    String lineContent = "";  // 行内容
    String setPage = "";  // 设置页面
    String statement = "";  // 语句
    String token = "";  // 打印标记

    Map<String, int[]> symbolData = new HashMap<String, int[]>();  // 符号数据的映射
    symbolData.put(" ", new int[]{0,0,0,0,0,0,0,0              });  // 空格符号的数据
    symbolData.put("A", new int[]{0,505,37,35,34,35,37,505     });  // A 符号的数据
    symbolData.put("G", new int[]{0,125,131,258,258,290,163,101});  // G 符号的数据
    symbolData.put("E", new int[]{0,512,274,274,274,274,258,258});  // E 符号的数据
    symbolData.put("T", new int[]{0,2,2,2,512,2,2,2            });  // T 符号的数据
    symbolData.put("W", new int[]{0,256,257,129,65,129,257,256 });  // W 符号的数据
    symbolData.put("L", new int[]{0,512,257,257,257,257,257,257});  // L 符号的数据
    symbolData.put("S", new int[]{0,69,139,274,274,274,163,69  });  // S 符号的数据
    symbolData.put("O", new int[]{0,125,131,258,258,258,131,125});  // O 符号的数据
    symbolData.put("N", new int[]{0,512,7,9,17,33,193,512      });  // N 符号的数据
    # 将字符"F"的符号数据放入字典中
    symbolData.put("F", new int[]{0,512,18,18,18,18,2,2        });
    # 将字符"K"的符号数据放入字典中
    symbolData.put("K", new int[]{0,512,17,17,41,69,131,258    });
    # 将字符"B"的符号数据放入字典中
    symbolData.put("B", new int[]{0,512,274,274,274,274,274,239});
    # 将字符"D"的符号数据放入字典中
    symbolData.put("D", new int[]{0,512,258,258,258,258,131,125});
    # 将字符"H"的符号数据放入字典中
    symbolData.put("H", new int[]{0,512,17,17,17,17,17,512     });
    # 将字符"M"的符号数据放入字典中
    symbolData.put("M", new int[]{0,512,7,13,25,13,7,512       });
    # 将字符"?"的符号数据放入字典中
    symbolData.put("?", new int[]{0,5,3,2,354,18,11,5          });
    # 将字符"U"的符号数据放入字典中
    symbolData.put("U", new int[]{0,128,129,257,257,257,129,128});
    # 将字符"R"的符号数据放入字典中
    symbolData.put("R", new int[]{0,512,18,18,50,82,146,271    });
    # 将字符"P"的符号数据放入字典中
    symbolData.put("P", new int[]{0,512,18,18,18,18,18,15      });
    # 将字符"Q"的符号数据放入字典中
    symbolData.put("Q", new int[]{0,125,131,258,258,322,131,381});
    # 将字符"Y"的符号数据放入字典中
    symbolData.put("Y", new int[]{0,8,9,17,481,17,9,8          });
    # 将字符"V"的符号数据放入字典中
    symbolData.put("V", new int[]{0,64,65,129,257,129,65,64    });
    # 将字符"X"的符号数据放入字典中
    symbolData.put("X", new int[]{0,388,69,41,17,41,69,388     });
    # 将字符"Z"的符号数据放入字典中
    symbolData.put("Z", new int[]{0,386,322,290,274,266,262,260});
    # 将字符"I"的符号数据放入字典中
    symbolData.put("I", new int[]{0,258,258,258,512,258,258,258});
    # 将字符"C"的符号数据放入字典中
    symbolData.put("C", new int[]{0,125,131,258,258,258,131,69 });
    # 将字符"J"的符号数据放入字典中
    symbolData.put("J", new int[]{0,65,129,257,257,257,129,128 });
    # 将字符"1"的符号数据放入字典中
    symbolData.put("1", new int[]{0,0,0,261,259,512,257,257    });
    # 将字符"2"的符号数据放入字典中
    symbolData.put("2", new int[]{0,261,387,322,290,274,267,261});
    # 将字符"*"的符号数据放入字典中
    symbolData.put("*", new int[]{0,69,41,17,512,17,41,69      });
    # 将字符"3"的符号数据放入字典中
    symbolData.put("3", new int[]{0,66,130,258,274,266,150,100 });
    # 将字符"4"的符号数据放入字典中
    symbolData.put("4", new int[]{0,33,49,41,37,35,512,33      });
    # 将字符"5"的符号数据放入字典中
    symbolData.put("5", new int[]{0,160,274,274,274,274,274,226});
    # 将字符"6"的符号数据放入字典中
    symbolData.put("6", new int[]{0,194,291,293,297,305,289,193});
    # 将字符"7"的符号数据放入字典中
    symbolData.put("7", new int[]{0,258,130,66,34,18,10,8      });
    # 将字符"8"的符号数据放入字典中
    symbolData.put("8", new int[]{0,69,171,274,274,274,171,69  });
    # 将字符"9"的符号数据放入字典中
    symbolData.put("9", new int[]{0,263,138,74,42,26,10,7      });
    # 将字符"="的符号数据放入字典中
    symbolData.put("=", new int[]{0,41,41,41,41,41,41,41       });
    // 将符号和对应的数据存入symbolData中
    symbolData.put("!", new int[]{0,1,1,1,384,1,1,1            });
    symbolData.put("0", new int[]{0,57,69,131,258,131,69,57    });
    symbolData.put(".", new int[]{0,1,1,129,449,129,1,1        });

    // 打印提示信息并获取用户输入的水平值
    System.out.print("HORIZONTAL? ");
    horizontal = Integer.parseInt(scan.nextLine());

    // 打印提示信息并获取用户输入的垂直值
    System.out.print("VERTICAL? ");
    vertical = Integer.parseInt(scan.nextLine());

    // 打印提示信息并获取用户输入的居中值
    System.out.print("CENTERED? ");
    centerResponse = scan.nextLine().toUpperCase();

    // 初始化centerFlag为0
    centerFlag = 0;

    // 按字典顺序比较用户输入的居中值，如果大于"P"，则将centerFlag设为1
    if (centerResponse.compareTo("P") > 0) {
      centerFlag = 1;
    }

    // 打印提示信息并获取用户输入的字符值
    System.out.print("CHARACTER (TYPE 'ALL' IF YOU WANT CHARACTER BEING PRINTED)? ");
    characters = scan.nextLine().toUpperCase();

    // 打印提示信息并获取用户输入的语句值
    System.out.print("STATEMENT? ");
    statement = scan.nextLine().toUpperCase();

    // 初始化打印设置
    System.out.print("SET PAGE? ");
    setPage = scan.nextLine();

    // 开始循环遍历语句中的字母
    }  // 结束循环遍历语句中的字母

    // 为横幅添加额外的长度
    for (index = 1; index <= 75; index++) {
      System.out.println("");
    }

  }  // play方法结束

  // 主方法
  public static void main(String[] args) {

    // 创建Banner对象并调用play方法
    Banner game = new Banner();
    game.play();

  }  // main方法结束
}  // 类 Banner 结束
```