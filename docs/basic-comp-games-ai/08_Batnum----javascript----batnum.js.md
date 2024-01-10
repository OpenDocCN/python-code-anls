# `basic-computer-games\08_Batnum\javascript\batnum.js`

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
                       // 监听键盘按下事件
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      // 获取输入的字符串
                                                      input_str = input_element.value;
                                                      // 移除输入框
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 打印输入的字符串
                                                      print(input_str);
                                                      print("\n");
                                                      // 解析输入的字符串
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
    // 打印游戏标题
    print(tab(33) + "BATNUM\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("THIS PROGRAM IS A 'BATTLE OF NUMBERS' GAME, WHERE THE\n");
    print("COMPUTER IS YOUR OPPONENT.\n");
    print("\n");
    print("THE GAME STARTS WITH AN ASSUMED PILE OF OBJECTS. YOU\n");
}
    # 打印提示信息，指示玩家交替从堆中移除物体
    print("AND YOUR OPPONENT ALTERNATELY REMOVE OBJECTS FROM THE PILE.\n");
    # 打印提示信息，定义胜利为取走最后一个物体或不取走
    print("WINNING IS DEFINED IN ADVANCE AS TAKING THE LAST OBJECT OR\n");
    # 打印提示信息，指示可以指定其他的开始条件
    print("NOT. YOU CAN ALSO SPECIFY SOME OTHER BEGINNING CONDITIONS.\n");
    # 打印提示信息，指示在游戏中不要使用零
    print("DON'T USE ZERO, HOWEVER, IN PLAYING THE GAME.\n");
    # 打印提示信息，指示输入负数以停止游戏
    print("ENTER A NEGATIVE NUMBER FOR NEW PILE SIZE TO STOP PLAYING.\n");
    # 打印空行
    print("\n");
    # 初始化变量 first_time 为 1
    first_time = 1;
    # 结束函数定义
    }
# 结束 main 函数的定义
}

# 调用 main 函数
main();
```