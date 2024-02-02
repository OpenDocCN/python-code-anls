# `basic-computer-games\42_Gunner\javascript\gunner.js`

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
                       // 监听键盘按下事件
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      // 获取输入的字符串
                                                      input_str = input_element.value;
                                                      // 移除输入框
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 输出输入的字符串
                                                      print(input_str);
                                                      print("\n");
                                                      // 解析输入的字符串并返回
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

// 输出标题
print(tab(30) + "GUNNER\n");
print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
print("\n");
print("\n");
print("\n");
print("YOU ARE THE OFFICER-IN-CHARGE, GIVING ORDERS TO A GUN\n");
print("CREW, TELLING THEM THE DEGREES OF ELEVATION YOU ESTIMATE\n");
print("WILL PLACE A PROJECTILE ON TARGET.  A HIT WITHIN 100 YARDS\n");
print("OF THE TARGET WILL DESTROY IT.\n");
print("\n");

// 主控制部分，使用 async 函数声明
async function main()
{
    // 在这里编写主要的程序逻辑
}
    # 打印一个空行
    print("\n");
    # 打印"OK.  RETURN TO BASE CAMP."
    print("OK.  RETURN TO BASE CAMP.\n");
# 结束 main 函数的定义
}

# 调用 main 函数
main();
```