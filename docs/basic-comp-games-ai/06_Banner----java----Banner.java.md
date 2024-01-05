# `06_Banner\java\Banner.java`

```
import java.util.Scanner;  # 导入 Scanner 类，用于用户输入
import java.util.HashMap;  # 导入 HashMap 类，用于创建哈希映射
import java.util.Map;  # 导入 Map 接口，用于操作键值对
import java.lang.Math;  # 导入 Math 类，用于数学运算

/**
 * Game of Banner
 * <p>
 * Based on the BASIC game of Banner here
 * https://github.com/coding-horror/basic-computer-games/blob/main/06%20Banner/banner.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's BASIC game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 *
 * Converted from BASIC to Java by Darren Cardenas.
 */

public class Banner {

  private final Scanner scan;  // 用于用户输入的 Scanner 对象
  public Banner() {
    // 创建一个新的 Scanner 对象，用于从标准输入读取数据
    scan = new Scanner(System.in);
  }  // End of constructor Banner

  public void play() {
    // 初始化各种变量
    int bitIndex = 0; // 用于跟踪比特索引
    int centerFlag = 0; // 用于标记中心位置
    int dataIndex = 0; // 用于跟踪数据索引
    int hIndex = 0; // 用于跟踪水平索引
    int horizontal = 0; // 用于存储水平值
    int index = 0; // 用于跟踪索引
    int letterIndex = 0; // 用于跟踪字母索引
    int tempVal = 0; // 用于存储临时值
    int vertical = 0; // 用于存储垂直值
    int vIndex = 0; // 用于跟踪垂直索引
    int writeIndex = 0; // 用于跟踪写入索引
    int[] writerMap = new int[10];  // 创建一个长度为10的整型数组writerMap
    int[] writeLimit = new int[10];  // 创建一个长度为10的整型数组writeLimit

    String centerResponse = "";  // 创建一个空字符串centerResponse
    String characters = "";  // 创建一个空字符串characters
    String letter = "";  // 创建一个空字符串letter
    String lineContent = "";  // 创建一个空字符串lineContent
    String setPage = "";  // 创建一个空字符串setPage
    String statement = "";  // 创建一个空字符串statement
    String token = "";  // 创建一个空字符串token  // 打印token

    Map<String, int[]> symbolData = new HashMap<String, int[]>();  // 创建一个HashMap对象symbolData，键为String类型，值为整型数组
    symbolData.put(" ", new int[]{0,0,0,0,0,0,0,0});  // 向symbolData中添加键为" "，值为包含8个整数的整型数组
    symbolData.put("A", new int[]{0,505,37,35,34,35,37,505});  // 向symbolData中添加键为"A"，值为包含8个整数的整型数组
    symbolData.put("G", new int[]{0,125,131,258,258,290,163,101});  // 向symbolData中添加键为"G"，值为包含8个整数的整型数组
    symbolData.put("E", new int[]{0,512,274,274,274,274,258,258});  // 向symbolData中添加键为"E"，值为包含8个整数的整型数组
    symbolData.put("T", new int[]{0,2,2,2,512,2,2,2});  // 向symbolData中添加键为"T"，值为包含8个整数的整型数组
    symbolData.put("W", new int[]{0,256,257,129,65,129,257,256});  // 向symbolData中添加键为"W"，值为包含8个整数的整型数组
    symbolData.put("L", new int[]{0,512,257,257,257,257,257,257});  // 向symbolData中添加键为"L"，值为包含8个整数的整型数组
# 将字符"S"的数据存入symbolData字典，数据为一个整数数组
symbolData.put("S", new int[]{0,69,139,274,274,274,163,69  });

# 将字符"O"的数据存入symbolData字典，数据为一个整数数组
symbolData.put("O", new int[]{0,125,131,258,258,258,131,125});

# 将字符"N"的数据存入symbolData字典，数据为一个整数数组
symbolData.put("N", new int[]{0,512,7,9,17,33,193,512      });

# 将字符"F"的数据存入symbolData字典，数据为一个整数数组
symbolData.put("F", new int[]{0,512,18,18,18,18,2,2        });

# 将字符"K"的数据存入symbolData字典，数据为一个整数数组
symbolData.put("K", new int[]{0,512,17,17,41,69,131,258    });

# 将字符"B"的数据存入symbolData字典，数据为一个整数数组
symbolData.put("B", new int[]{0,512,274,274,274,274,274,239});

# 将字符"D"的数据存入symbolData字典，数据为一个整数数组
symbolData.put("D", new int[]{0,512,258,258,258,258,131,125});

# 将字符"H"的数据存入symbolData字典，数据为一个整数数组
symbolData.put("H", new int[]{0,512,17,17,17,17,17,512     });

# 将字符"M"的数据存入symbolData字典，数据为一个整数数组
symbolData.put("M", new int[]{0,512,7,13,25,13,7,512       });

# 将字符"?"的数据存入symbolData字典，数据为一个整数数组
symbolData.put("?", new int[]{0,5,3,2,354,18,11,5          });

# 将字符"U"的数据存入symbolData字典，数据为一个整数数组
symbolData.put("U", new int[]{0,128,129,257,257,257,129,128});

# 将字符"R"的数据存入symbolData字典，数据为一个整数数组
symbolData.put("R", new int[]{0,512,18,18,50,82,146,271    });

# 将字符"P"的数据存入symbolData字典，数据为一个整数数组
symbolData.put("P", new int[]{0,512,18,18,18,18,18,15      });

# 将字符"Q"的数据存入symbolData字典，数据为一个整数数组
symbolData.put("Q", new int[]{0,125,131,258,258,322,131,381});

# 将字符"Y"的数据存入symbolData字典，数据为一个整数数组
symbolData.put("Y", new int[]{0,8,9,17,481,17,9,8          });

# 将字符"V"的数据存入symbolData字典，数据为一个整数数组
symbolData.put("V", new int[]{0,64,65,129,257,129,65,64    });

# 将字符"X"的数据存入symbolData字典，数据为一个整数数组
symbolData.put("X", new int[]{0,388,69,41,17,41,69,388     });

# 将字符"Z"的数据存入symbolData字典，数据为一个整数数组
symbolData.put("Z", new int[]{0,386,322,290,274,266,262,260});

# 将字符"I"的数据存入symbolData字典，数据为一个整数数组
symbolData.put("I", new int[]{0,258,258,258,512,258,258,258});

# 将字符"C"的数据存入symbolData字典，数据为一个整数数组
symbolData.put("C", new int[]{0,125,131,258,258,258,131,69 });
    // 将符号和对应的数据存入 symbolData 中
    symbolData.put("J", new int[]{0,65,129,257,257,257,129,128 });
    symbolData.put("1", new int[]{0,0,0,261,259,512,257,257    });
    symbolData.put("2", new int[]{0,261,387,322,290,274,267,261});
    symbolData.put("*", new int[]{0,69,41,17,512,17,41,69      });
    symbolData.put("3", new int[]{0,66,130,258,274,266,150,100 });
    symbolData.put("4", new int[]{0,33,49,41,37,35,512,33      });
    symbolData.put("5", new int[]{0,160,274,274,274,274,274,226});
    symbolData.put("6", new int[]{0,194,291,293,297,305,289,193});
    symbolData.put("7", new int[]{0,258,130,66,34,18,10,8      });
    symbolData.put("8", new int[]{0,69,171,274,274,274,171,69  });
    symbolData.put("9", new int[]{0,263,138,74,42,26,10,7      });
    symbolData.put("=", new int[]{0,41,41,41,41,41,41,41       });
    symbolData.put("!", new int[]{0,1,1,1,384,1,1,1            });
    symbolData.put("0", new int[]{0,57,69,131,258,131,69,57    });
    symbolData.put(".", new int[]{0,1,1,129,449,129,1,1        });

    // 打印提示信息并获取用户输入的水平值
    System.out.print("HORIZONTAL? ");
    horizontal = Integer.parseInt(scan.nextLine());

    // 打印提示信息并获取用户输入的垂直值
    System.out.print("VERTICAL? ");
    // 从输入中读取一个整数
    vertical = Integer.parseInt(scan.nextLine());

    // 打印提示信息并从输入中读取用户的回答，转换为大写形式
    System.out.print("CENTERED? ");
    centerResponse = scan.nextLine().toUpperCase();

    // 初始化centerFlag变量为0
    centerFlag = 0;

    // 使用词典顺序比较用户回答和字符串"P"，如果用户回答大于"P"，则将centerFlag设为1
    if (centerResponse.compareTo("P") > 0) {
      centerFlag = 1;
    }

    // 打印提示信息并从输入中读取用户的回答，转换为大写形式
    System.out.print("CHARACTER (TYPE 'ALL' IF YOU WANT CHARACTER BEING PRINTED)? ");
    characters = scan.nextLine().toUpperCase();

    // 打印提示信息并从输入中读取用户的回答，转换为大写形式
    System.out.print("STATEMENT? ");
    statement = scan.nextLine().toUpperCase();

    // 打印提示信息
    System.out.print("SET PAGE? ");
    setPage = scan.nextLine();  // 从输入中获取下一页的内容

    // 开始循环遍历语句中的每个字母
    for (letterIndex = 1; letterIndex <= statement.length(); letterIndex++) {

      // 提取一个字母
      letter = String.valueOf(statement.charAt(letterIndex - 1));

      // 开始循环遍历所有符号数据
      for (String symbolString: symbolData.keySet()) {

        // 开始处理字母
        if (letter.equals(" ")) {  // 如果字母是空格
          for (index = 1; index <= (7 * horizontal); index++) {  // 打印空行
            System.out.println("");
          }
          break;  // 跳出循环

        } else if (letter.equals(symbolString)) {  // 如果字母与符号数据中的某个符号匹配
          token = characters;  // 将token设置为字符
          if (characters.equals("ALL")) {  // 如果字符变量等于"ALL"
            token = symbolString;  // 将token设置为symbolString
          }

          for (dataIndex = 1; dataIndex <= 7; dataIndex++) {  // 遍历dataIndex从1到7的循环

            // 避免覆盖符号数据
            tempVal = symbolData.get(symbolString)[dataIndex];  // 从symbolData中获取symbolString对应的数据，并将其赋值给tempVal

            for (bitIndex = 8; bitIndex >= 0; bitIndex--) {  // 遍历bitIndex从8到0的循环

              if (Math.pow(2, bitIndex) < tempVal) {  // 如果2的bitIndex次方小于tempVal
                writerMap[9 - bitIndex] = 1;  // 将writerMap中索引为9-bitIndex的位置设置为1
                tempVal -= Math.pow(2, bitIndex);  // tempVal减去2的bitIndex次方

                if (tempVal == 1) {  // 如果tempVal等于1
                  writeLimit[dataIndex] = 9 - bitIndex;  // 将writeLimit中索引为dataIndex的位置设置为9-bitIndex
                  break;  // 跳出循环
                }
            } else {
                // 如果 writerMap[writeIndex] 不等于 0，则在 lineContent 中添加 token 中对应位置的字符
                lineContent += token.charAt(index - 1);
            }
        }
        // 添加换行符
        lineContent += "\n";
    }
}
// 将 lineContent 添加到 result 中
result += lineContent;
// 更新 dataIndex
dataIndex++;
// 如果 dataIndex 大于 token.length()，则重置为 1
if (dataIndex > token.length()) {
    dataIndex = 1;
}
// 如果 centerFlag 为 1，则更新 centerFlag 为 0；否则更新为 1
centerFlag = 1 - centerFlag;
```
                    }
                  }

                } else {

                  for (vIndex = 1; vIndex <= vertical; vIndex++) {
                    lineContent += token;
                  }
                }

              }  // End of writeIndex loop

              System.out.println(lineContent);

            }  // End of hIndex loop

          }  // End of dataIndex loop

          // Add padding between letters
          for (index = 1; index <= 2 * horizontal; index++) {
```

这段代码看起来是一个嵌套的循环结构，但是语言不太清晰，需要进一步了解上下文才能准确地解释每个语句的作用。
    // 打印空行
    System.out.println("");
  }

}  // 结束字母处理

}  // 结束循环遍历所有符号数据

}  // 结束循环遍历语句字母

// 为横幅添加额外长度
for (index = 1; index <= 75; index++) {
  // 打印空行
  System.out.println("");
}

}  // 方法play结束

public static void main(String[] args) {
  // 创建Banner对象
  Banner game = new Banner();
  // 调用play方法
  game.play();
}
  }  // 方法 main 的结束

}  // 类 Banner 的结束
```