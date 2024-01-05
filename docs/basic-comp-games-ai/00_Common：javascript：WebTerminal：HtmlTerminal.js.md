# `00_Common\javascript\WebTerminal\HtmlTerminal.js`

```
/**
 * @class HtmlTerminal
 * 
 * This class is a very basic implementation of a "terminal" in the browser.
 * It provides simple functions like "write" and an "input" Callback.
 * 
 * @license AGPL-2.0
 * @author Alexaner Wunschik <https://github.com/mojoaxel>
 */
class HtmlTerminal {

  /**
   * Input callback.
   * If the prompt is activated by calling the input function
   * a callback is defined. If this member is not set this means
   * the prompt is not active.
   * 
   * @private
   * @type {function}
   */
  inputCallback;
```

这段代码是一个类的定义，名为HtmlTerminal。它包含一个成员变量inputCallback，用于存储输入回调函数。如果prompt被调用，inputCallback会被定义，如果没有设置，表示prompt没有被激活。这个成员变量是私有的，类型为function。
  #inputCallback = undefined;  # 定义一个变量inputCallback，并初始化为undefined

  /**
   * A html element to show a "prompt".
   * 用于显示“提示”信息的HTML元素。
   * 
   * @private
   * @type {HTMLElement}
   */
  #$prompt;  # 声明一个私有属性，用于存储一个HTML元素

  /**
   * Constructor
   * Creates a basic terminal simulation on the provided HTMLElement.
   * 构造函数
   * 在提供的HTMLElement上创建一个基本的终端模拟。
   * 
   * @param {HTMLElement} $output - a dom element
   * @param {HTMLElement} $output - 一个DOM元素
   */
  constructor($output) {  # 构造函数，接受一个DOM元素作为参数
    // Store the output DOM element in a local variable.
    // 将输出的DOM元素存储在一个局部变量中。
    this.$output = $output;  # 将传入的DOM元素存储在实例的$output属性中
    // 清空终端。
    this.clear();

    // 将调用 "terminal" 添加到 $output 元素。
    this.$output.classList.add('terminal');

    // 创建一个提示元素。
    // 如果需要输入，将添加此元素。
    this.#$prompt = document.createElement("input");
    this.#$prompt.setAttribute("id", "prompt");
    this.#$prompt.setAttribute("type", "text");
    this.#$prompt.setAttribute("length", "50");
    this.#$prompt.addEventListener("keydown", this.#handleKey.bind(this));

    // 每次点击时强制将焦点放在提示上。
    // 这对移动设备的支持是必要的。
    document.body.addEventListener('click', () => {
      this.#$prompt.focus();
    });
  }
# 创建一个新的 HTMLElement，其中包含给定的文本内容。
# 然后将该元素作为新的“行”添加到 $output 中。
# 
# @private
# @memberof MinimalTerminal
# @param {String} text - 应在新“行”中显示的文本。
# @returns {HTMLElement} - 返回一个新的 DOM 元素 <pre class="line"></pre>
*/
newLine(text) {
  const $lineNode = document.createElement("pre");  # 创建一个 <pre> 元素节点
  $lineNode.classList.add("line");  # 为元素节点添加类名 "line"
  $lineNode.innerText = text;  # 设置元素节点的文本内容为传入的文本
  return $lineNode;  # 返回创建的元素节点
}
   * @private
   * @param {*} e 
   */
  #handleKey(e) {
    // 如果没有定义输入回调函数，则直接返回
    if (!this.#inputCallback) {
      return;
    }

    // 如果按下的键是回车键（keyCode 为 13）
    if (e.keyCode == 13) {
      // 获取输入框中的文本
      const text = this.#$prompt.value;
      // 清空输入框
      this.#$prompt.value = '';
      // 移除输入框
      this.#$prompt.remove();
      // 调用输入回调函数，并传入文本和换行符
      this.#inputCallback(text + '\n');
    }
  }

  /**
   * Clear the terminal.
   * Remove all lines.
  /**
   * 清空输出区域的内容
   * 
   * @public
   */
  clear() {
    this.$output.innerText = "";
  }

  /**
   * 创建一个新的 div 元素，并添加 HTML 内容
   * 
   * @public
   * @param {*} htmlContent - 要添加的 HTML 内容
   */
  inserHtml(htmlContent) {
    // 创建一个新的 div 元素
    const $htmlNode = document.createElement("div");
    // 将传入的 HTML 内容赋值给新创建的 div 元素
    $htmlNode.innerHTML = htmlContent;
    // 将新创建的 div 元素添加到输出区域
    this.$output.appendChild($htmlNode);
    // 滚动页面到底部
    document.body.scrollTo(0, document.body.scrollHeight);
  }
  /**
   * Write a text to the terminal.
   * By default there is no linebreak at the end of a new line
   * except the line ends with a "\n".
   * If the given text has multiple linebreaks, multiple lines are inserted.
   * 
   * @public
   * @param {string} text 
   */
  write(text) {
    if (!text || text.length <= 0) {
      // 如果文本为空，插入一个空行
      this.$output.appendChild(document.createElement("br"));
    } else if (text.endsWith("\n")) {
      // 如果文本以换行符结尾，插入一个新行节点和一个换行符
      const $lineNode = this.#newLine(text);
      this.$output.appendChild(this.#newLine(text));
      this.$output.appendChild(document.createElement("br"));
    } else if (text.includes("\n")) {
      // 如果文本包含换行符，根据换行符分割文本，插入多行节点
      const lines = text.split("\n");  // 将文本按照换行符分割成多行，并存储在数组中
      lines.forEach((line) => {  // 遍历每一行文本
        this.write(line);  // 调用write方法输出每一行文本
      });
    } else {
      // single line
      this.$output.appendChild(this.#newLine(text));  // 将单行文本添加到输出的DOM元素中
    }

    // scroll to the buttom of the page
    document.body.scrollTo(0, document.body.scrollHeight);  // 将页面滚动到底部
  }

  /**
   * Like "write" but with a newline at the end.
   * 
   * @public
   * @param {*} text  // 传入的文本参数
   */
  writeln(text) {  // 定义一个名为writeln的方法，用于输出文本并换行
    this.write(text + "\n");  // 将给定的文本内容写入到终端中，并在末尾添加换行符

  /**
   * Query from user input.
   * This is done by adding a input-element at the end of the terminal,
   * that showes a prompt and a blinking cursor.
   * If a key is pressed the input is added to the prompt element.
   * The input ends with a linebreak.
   * 
   * @public
   * @param {*} callback 
   */
  input(callback) {
    // show prompt with a blinking prompt
    this.#inputCallback = callback;  // 将传入的回调函数赋值给私有属性 inputCallback
    this.$output.appendChild(this.#$prompt);  // 将 prompt 元素添加到输出终端中
    this.#$prompt.focus();  // 设置 prompt 元素获得焦点
  }
}
bio = BytesIO(open(fname, 'rb').read())
```
这行代码创建了一个字节流对象`bio`，并使用`open`函数打开给定文件名`fname`，以二进制模式（'rb'）读取文件内容，并将其封装成字节流。

```python
zip = zipfile.ZipFile(bio, 'r')
```
这行代码使用字节流里面的内容创建了一个ZIP文件对象`zip`，以只读模式（'r'）打开。

```python
fdict = {n:zip.read(n) for n in zip.namelist()}
```
这行代码遍历ZIP对象所包含的文件名列表，然后使用`zip.read(n)`读取每个文件的数据，并将文件名和数据组成字典`fdict`。

```python
zip.close()
```
这行代码关闭了ZIP对象，释放了与之相关的资源。

```python
return fdict
```
这行代码返回了包含文件名到数据的字典`fdict`。
```