# `basic-computer-games\02_Amazing\javascript\amazing.js`

```

// 定义一个打印函数，用于在页面上输出字符串
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

                       // 在页面上输出提示符
                       print("? ");
                       // 设置输入框类型和长度
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       // 将输入框添加到页面上
                       document.getElementById("output").appendChild(input_element);
                       // 让输入框获得焦点
                       input_element.focus();
                       input_str = undefined;
                       // 监听输入框的键盘事件
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      // 当按下回车键时，获取输入的字符串并移除输入框
                                                      input_str = input_element.value;
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 在页面上输出输入的字符串
                                                      print(input_str);
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

// 在页面上输出标题
print(tab(28) + "AMAZING PROGRAM\n");
print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
print("\n");
print("\n");
print("\n");
print("FOR EXAMPLE TYPE 10,10 AND PRESS ENTER\n");
print("\n");

// 主程序
async function main()
// 如果想要查看访问的单元格顺序
//    for (j = 1; j <= v2; j++) {
//        str = "I";
//        for (i = 1; i <= h; i++) {
//            str += w[i][j] + " ";
//        }
//        print(str + "\n");
//    }
}

// 调用主程序
main();

```