# `basic-computer-games\63_Name\javascript\name.js`

```
// 定义一个打印函数，将字符串添加到输出元素中
function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 定义一个输入函数，返回一个 Promise 对象
function input()
{
    var input_element;
    var input_str;

    return new Promise(function (resolve) {
                       // 创建一个输入框元素
                       input_element = document.createElement("INPUT");

                       // 打印提示符
                       print("? ");
                       // 设置输入框类型和长度
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       // 将输入框添加到输出元素中
                       document.getElementById("output").appendChild(input_element);
                       // 让输入框获得焦点
                       input_element.focus();
                       input_str = undefined;
                       // 监听输入框的键盘事件
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      // 当按下回车键时，获取输入框的值
                                                      input_str = input_element.value;
                                                      // 从输出元素中移除输入框
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 打印输入的字符串
                                                      print(input_str);
                                                      // 打印换行符
                                                      print("\n");
                                                      // 解析 Promise 对象
                                                      resolve(input_str);
                                                      }
                                                      });
                       });
}

// 定义一个生成指定数量空格的函数
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

var str;
var b;

// 主程序
async function main()
{
    // 打印名称
    print(tab(34) + "NAME\n");
    // 打印创意计算的地址
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    // 打印多个换行符
    print("\n");
    print("\n");
    print("\n");
    // 打印问候语
    print("HELLO.\n");
    // 打印计算机的名称
    print("MY NAME IS CREATIVE COMPUTER.\n");
    // 打印提示信息，要求输入名字
    print("WHAT'S YOUR NAME (FIRST AND LAST)");
    // 获取输入的名字
    str = await input();
    // 获取名字的长度
    l = str.length;
    // 打印感谢信息
    print("\n");
    print("THANK YOU, ");
}
    # 逆序打印字符串中的字符
    for (i = l; i >= 1; i--)
        print(str[i - 1]);
    # 打印句号
    print(".\n");
    # 打印提示信息
    print("OOPS!  I GUESS I GOT IT BACKWARDS.  A SMART\n");
    print("COMPUTER LIKE ME SHOULDN'T MAKE A MISTAKE LIKE THAT!\n");
    print("\n");
    # 打印提示信息
    print("BUT I JUST NOTICED YOUR LETTERS ARE OUT OF ORDER.\n");
    print("LET'S PUT THEM IN ORDER LIKE THIS: ");
    # 创建空数组
    b = [];
    # 将字符串中每个字符的 ASCII 码存入数组 b
    for (i = 1; i <= l; i++)
        b[i - 1] = str.charCodeAt(i - 1);
    # 对数组 b 进行排序
    b.sort();
    # 将排序后的字符打印出来
    for (i = 1; i <= l; i++)
        print(String.fromCharCode(b[i - 1]));
    print("\n");
    print("\n");
    # 提示用户是否喜欢排序后的字符顺序
    print("DON'T YOU LIKE THAT BETTER");
    # 等待用户输入
    ds = await input();
    # 根据用户输入进行不同的输出
    if (ds == "YES") {
        print("\n");
        print("I KNEW YOU'D AGREE!!\n");
    } else {
        print("\n");
        print("I'M SORRY YOU DON'T LIKE IT THAT WAY.\n");
    }
    print("\n");
    # 打印结束语
    print("I REALLY ENJOYED MEETING YOU " + str + ".\n");
    print("HAVE A NICE DAY!\n");
# 结束当前的函数定义
}

# 调用名为main的函数
main();
```