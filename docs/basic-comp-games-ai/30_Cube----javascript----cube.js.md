# `basic-computer-games\30_Cube\javascript\cube.js`

```py
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
                                                      // 当按下回车键时，获取输入框的值
                                                      input_str = input_element.value;
                                                      // 从输出元素中移除输入框
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 打印输入的字符串
                                                      print(input_str);
                                                      // 打印换行符
                                                      print("\n");
                                                      // 解析输入的字符串并传递给 Promise 对象
                                                      resolve(input_str);
                                                      }
                                                      });
                       });
}

// 定义一个制表符函数，返回指定数量的空格字符串
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
    print(tab(33) + "CUBE\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    // 打印提示信息，等待用户输入并将结果转换为整数
    print("DO YOU WANT TO SEE THE INSTRUCTIONS? (YES--1,NO--0)");
    b7 = parseInt(await input());
}
    # 如果 b7 不等于 0，则执行以下代码块
    if (b7 != 0) {
        # 打印游戏规则和提示信息
        print("THIS IS A GAME IN WHICH YOU WILL BE PLAYING AGAINST THE\n");
        print("RANDOM DECISION OF THE COMPUTER. THE FIELD OF PLAY IS A\n");
        print("CUBE OF SIDE 3. ANY OF THE 27 LOCATIONS CAN BE DESIGNATED\n");
        print("BY INPUTING THREE NUMBERS SUCH AS 2,3,1. AT THE START,\n");
        print("YOU ARE AUTOMATICALLY AT LOCATION 1,1,1. THE OBJECT OF\n");
        print("THE GAME IS TO GET TO LOCATION 3,3,3. ONE MINOR DETAIL:\n");
        print("THE COMPUTER WILL PICK, AT RANDOM, 5 LOCATIONS AT WHICH\n");
        print("IT WILL PLANT LAND MINES. IF YOU HIT ONE OF THESE LOCATIONS\n");
        print("YOU LOSE. ONE OTHER DETAIL: YOU MAY MOVE ONLY ONE SPACE \n");
        print("IN ONE DIRECTION EACH MOVE. FOR  EXAMPLE: FROM 1,1,2 YOU\n");
        print("MAY MOVE TO 2,1,2 OR 1,1,3. YOU MAY NOT CHANGE\n");
        print("TWO OF THE NUMBERS ON THE SAME MOVE. IF YOU MAKE AN ILLEGAL\n");
        print("MOVE, YOU LOSE AND THE COMPUTER TAKES THE MONEY YOU MAY\n");
        print("HAVE BET ON THAT ROUND.\n");
        print("\n");
        print("\n");
        print("ALL YES OR NO QUESTIONS WILL BE ANSWERED BY A 1 FOR YES\n");
        print("OR A 0 (ZERO) FOR NO.\n");
        print("\n");
        print("WHEN STATING THE AMOUNT OF A WAGER, PRINT ONLY THE NUMBER\n");
        print("OF DOLLARS (EXAMPLE: 250)  YOU ARE AUTOMATICALLY STARTED WITH\n");
        print("500 DOLLARS IN YOUR ACCOUNT.\n");
        print("\n");
        print("GOOD LUCK!\n");
    }
    # 初始化变量 a1 为 500
    a1 = 500;
    # 打印结束游戏的提示信息
    print("TOUGH LUCK!\n");
    print("\n");
    print("GOODBYE.\n");
# 结束当前的函数定义
}

# 调用名为main的函数
main();
```