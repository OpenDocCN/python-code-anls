# `basic-computer-games\64_Nicomachus\javascript\nicomachus.js`

```
// NICOMACHUS
//
// Converted from BASIC to Javascript by Oscar Toledo G. (nanochess)
//

// 定义打印函数，将字符串添加到输出元素中
function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 定义输入函数，返回一个 Promise 对象
function input()
{
    var input_element;
    var input_str;

    return new Promise(function (resolve) {
                       // 创建一个输入框元素
                       input_element = document.createElement("INPUT");

                       // 打印提示符
                       print("? ");
                       // 设置输入框类型为文本，长度为50
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       // 将输入框添加到输出元素中
                       document.getElementById("output").appendChild(input_element);
                       // 让输入框获得焦点
                       input_element.focus();
                       input_str = undefined;
                       // 监听键盘按下事件
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      // 获取输入框的值
                                                      input_str = input_element.value;
                                                      // 移除输入框
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 打印输入的值
                                                      print(input_str);
                                                      print("\n");
                                                      // 解析 Promise 对象
                                                      resolve(input_str);
                                                      }
                                                      });
                       });
}

// 定义制表符函数，返回指定数量的空格字符串
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
    // 打印标题
    print(tab(33) + "NICOMA\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("BOOMERANG PUZZLE FROM ARITHMETICA OF NICOMACHUS -- A.D. 90!\n");
    # 进入无限循环，直到用户中断程序
    while (1) {
        # 打印空行
        print("\n");
        # 打印提示信息
        print("PLEASE THINK OF A NUMBER BETWEEN 1 AND 100.\n");
        # 打印提示信息
        print("YOUR NUMBER DIVIDED BY 3 HAS A REMAINDER OF");
        # 将用户输入的字符串转换为整数
        a = parseInt(await input());
        # 打印提示信息
        print("YOUR NUMBER DIVIDED BY 5 HAS A REMAINDER OF");
        # 将用户输入的字符串转换为整数
        b = parseInt(await input());
        # 打印提示信息
        print("YOUR NUMBER DIVIDED BY 7 HAS A REMAINDER OF");
        # 将用户输入的字符串转换为整数
        c = parseInt(await input());
        # 打印空行
        print("\n");
        # 打印提示信息
        print("LET ME THINK A MOMENT...\n");
        # 打印空行
        print("\n");
        # 根据用户输入的数值进行计算
        d = 70 * a + 21 * b + 15 * c;
        # 循环直到 d 小于等于 105
        while (d > 105)
            d -= 105;
        # 打印结果
        print("YOUR NUMBER WAS " + d + ", RIGHT");
        # 进入内部无限循环，直到用户中断程序
        while (1) {
            # 获取用户输入的字符串
            str = await input();
            # 打印空行
            print("\n");
            # 判断用户输入的字符串
            if (str == "YES") {
                # 打印提示信息
                print("HOW ABOUT THAT!!\n");
                # 退出内部循环
                break;
            } else if (str == "NO") {
                # 打印提示信息
                print("I FEEL YOUR ARITHMETIC IS IN ERROR.\n");
                # 退出内部循环
                break;
            } else {
                # 打印提示信息
                print("EH?  I DON'T UNDERSTAND '" + str + "'  TRY 'YES' OR 'NO'.\n");
            }
        }
        # 打印空行
        print("\n");
        # 打印提示信息
        print("LET'S TRY ANOTHER.\n");
    }
# 调用名为main的函数
main();
```