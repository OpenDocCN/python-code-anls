# `basic-computer-games\59_Lunar_LEM_Rocket\javascript\lem.js`

```py
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
                       // 监听键盘事件，当按下回车键时，获取输入值并解析 Promise
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      input_str = input_element.value;
                                                      document.getElementById("output").removeChild(input_element);
                                                      print(input_str);
                                                      print("\n");
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
    // 输出标题
    print(tab(34) + "LEM\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    // 初始化变量
    // ROCKT2 是一个模拟登月的交互式游戏，类似阿波罗计划
    // 完全没有机会参与
    zs = "GO";
    b1 = 1;
    // 输出提示信息
    print("\n");
    print("TOO BAD, THE SPACE PROGRAM HATES TO LOSE EXPERIENCED\n");
    print("ASTRONAUTS.\n");
}
# 调用名为main的函数
main();
```