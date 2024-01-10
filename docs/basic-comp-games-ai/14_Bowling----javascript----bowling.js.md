# `basic-computer-games\14_Bowling\javascript\bowling.js`

```
// 定义一个打印函数，将字符串输出到指定的元素上
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

                       // 输出提示符
                       print("? ");
                       // 设置输入框类型和长度
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       // 将输入框添加到指定元素上
                       document.getElementById("output").appendChild(input_element);
                       // 让输入框获得焦点
                       input_element.focus();
                       input_str = undefined;
                       // 监听键盘事件，当按下回车键时，获取输入值并移除输入框
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      input_str = input_element.value;
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 打印输入值
                                                      print(input_str);
                                                      print("\n");
                                                      // 解析输入值并返回
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

// 主程序
async function main()
{
    // 打印标题
    print(tab(34) + "BOWL\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    // 初始化数组
    c = [];
    a = [];
    for (i = 0; i <= 15; i++)
        c[i] = 0;
    // 打印欢迎信息
    print("WELCOME TO THE ALLEY\n");
    print("BRING YOUR FRIENDS\n");
    print("OKAY LET'S FIRST GET ACQUAINTED\n");
    print("\n");
    print("THE INSTRUCTIONS (Y/N)\n");
}
    # 从输入中获取字符串
    str = await input();
    # 如果字符串的第一个字符是 "Y"，则执行以下代码块
    if (str.substr(0, 1) == "Y") {
        # 打印保龄球游戏的介绍信息
        print("THE GAME OF BOWLING TAKES MIND AND SKILL.DURING THE GAME\n");
        print("THE COMPUTER WILL KEEP SCORE.YOU MAY COMPETE WITH\n");
        print("OTHER PLAYERS[UP TO FOUR].YOU WILL BE PLAYING TEN FRAMES\n");
        print("ON THE PIN DIAGRAM 'O' MEANS THE PIN IS DOWN...'+' MEANS THE\n");
        print("PIN IS STANDING.AFTER THE GAME THE COMPUTER WILL SHOW YOUR\n");
        print("SCORES .\n");
    }
    # 打印提示信息，询问有多少人参与游戏
    print("FIRST OF ALL...HOW MANY ARE PLAYING");
    # 从输入中获取整数，并转换为整型
    r = parseInt(await input());
    # 结束代码块
    }
# 结束当前的函数定义
}

# 调用名为main的函数
main();
```