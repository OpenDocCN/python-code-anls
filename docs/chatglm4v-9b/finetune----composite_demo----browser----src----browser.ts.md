# `.\chatglm4-finetune\composite_demo\browser\src\browser.ts`

```py
import { JSDOM } from 'jsdom'; // 从 jsdom 库导入 JSDOM 类，用于创建 DOM 对象
import TurndownService from 'turndown'; // 从 turndown 库导入 TurndownService 类，用于将 HTML 转换为 Markdown

import config from './config'; // 导入配置文件中的配置
import { Message, ToolObservation } from './types'; // 导入 Message 和 ToolObservation 类型定义
import { logger, withTimeout } from './utils'; // 从 utils 模块导入 logger 和 withTimeout 工具函数

// 表示显示中的引用
interface Quote {
  text: string; // 引用的文本内容
  metadata: Metadata[]; // 引用的元数据数组
}

interface ActionResult {
  contentType: string; // 内容类型
  metadataList?: TetherQuoteMetadata[]; // 可选的元数据列表
  metadata?: any; // 可选的元数据
  roleMetadata: string; // 角色元数据
  message: string; // 消息内容
}

// 表示要在最终答案中标记的元数据
interface Metadata {
  type: string; // 元数据类型
  title: string; // 元数据标题
  url: string; // 元数据链接
  lines: string[]; // 相关行的数组
}

interface TetherQuoteExtra {
  cited_message_idx: number; // 引用的消息索引
  evidence_text: string; // 证据文本
}

interface TetherQuoteMetadata {
  type: string; // 元数据类型
  title: string; // 元数据标题
  url: string; // 元数据链接
  text: string; // 元数据文本
  pub_date?: string; // 可选的发布日期
  extra?: TetherQuoteExtra; // 可选的附加信息
}

interface Citation {
  citation_format_type: string; // 引用格式类型
  start_ix: number; // 开始索引
  end_ix: number; // 结束索引
  metadata?: TetherQuoteMetadata; // 可选的元数据
  invalid_reason?: string; // 可选的无效原因
}

interface PageState {
  aCounter: number; // 链接计数器
  imgCounter: number; // 图片计数器

  url: URL; // 当前页面的 URL 对象
  url_string: string; // 当前页面的 URL 字符串
  hostname: string; // 当前页面的主机名
  links: string[]; // 当前页面的链接数组
  links_meta: TetherQuoteMetadata[]; // 链接的元数据数组
  lines: string[]; // 当前页面的文本行
  line_source: Record<string, Metadata>; // 行的元数据，键为字符串表示的区间
  title?: string; // 可选的页面标题
}

interface BrowserState {
  pageStack: PageState[]; // 页面状态栈
  quoteCounter: number; // 引用计数器
  quotes: Record<string, Quote>; // 引用记录，键为引用 ID
}

// 移除密集链接的函数，接受一个文档和比率阈值
function removeDenseLinks(document: Document, ratioThreshold: number = 0.5) {
  // 移除导航元素
  const navs = document.querySelectorAll('nav'); // 查询所有导航元素
  navs.forEach(nav => { // 遍历每个导航元素
    if (nav.parentNode) { // 如果有父节点
      nav.parentNode.removeChild(nav); // 从父节点中移除导航元素
    }
  });

  // 查询列表、div、span、表格和段落元素
  const elements = document.querySelectorAll('ul, ol, div, span, nav, table, p'); // 查询相关元素
  elements.forEach(element => { // 遍历每个元素
    if (element === null) return; // 如果元素为 null，直接返回

    const children = Array.from(element.childNodes); // 将子节点转换为数组
    const links = element.querySelectorAll('a'); // 查询所有链接元素

    if (children.length <= 1) return; // 如果子节点数量小于等于1，直接返回

    const allText = element.textContent ? element.textContent.trim().replace(/\s+/g, '') : ''; // 获取元素文本内容并去除多余空格
    const linksText = Array.from(links) // 将链接文本合并成一个字符串
      .map(link => (link.textContent ? link.textContent.trim() : '')) // 处理每个链接的文本
      .join('') // 合并为单个字符串
      .replace(/\s+/g, ''); // 去除多余空格

    if (allText.length === 0 || linksText.length === 0) return; // 如果没有文本内容或链接文本，直接返回

    let ratio = linksText.length / allText.length; // 计算链接文本占总文本的比率
    if (ratio > ratioThreshold && element.parentNode) { // 如果比率超过阈值且有父节点
      element.parentNode.removeChild(element); // 从父节点中移除该元素
    }
  });
}

abstract class BaseBrowser {
  public static toolName = 'browser' as const; // 定义工具名称为 'browser'
  public description = 'BaseBrowser'; // 描述为 'BaseBrowser'

  private turndownService = new TurndownService({ // 初始化 TurndownService 实例
    headingStyle: 'atx', // 设置标题样式为 atx
  });

  private state: BrowserState; // 声明浏览器状态

  private transform(dom: JSDOM): string { // 转换函数，接收 JSDOM 对象，返回字符串
    let state = this.lastPageState(); // 获取最后一个页面状态
    state.aCounter = 0; // 重置链接计数器
    state.imgCounter = 0; // 重置图片计数器
    state.links = []; // 清空链接数组

    return this.turndownService.turndown(dom.window.document); // 将 DOM 文档转换为 Markdown
  }

  private formatPage(state: PageState): string { // 格式化页面函数，接收页面状态，返回字符串
    // 将状态中的行合并成一个字符串，以换行符分隔
    let formatted_lines = state.lines.join('\n');
    // 如果标题存在，则格式化标题并添加换行符，否则为空字符串
    let formatted_title = state.title ? `TITLE: ${state.title}\n\n` : '';
    // 定义可见范围的格式化字符串
    let formatted_range = `\nVisible: 0% - 100%`;
    // 将标题、行和可见范围合并成一个完整的消息字符串
    let formatted_message = formatted_title + formatted_lines + formatted_range;
    // 返回格式化后的消息字符串
    return formatted_message;
  }

  // 创建新的页面状态并返回
  private newPageState(): PageState {
    return {
      // 初始化计数器 aCounter 为 0
      aCounter: 0,
      // 初始化计数器 imgCounter 为 0
      imgCounter: 0,

      // 创建新的 URL 对象，指向空白页面
      url: new URL('about:blank'),
      // 初始化 URL 字符串为 'about:blank'
      url_string: 'about:blank',
      // 初始化主机名为空字符串
      hostname: '',
      // 初始化标题为空字符串
      title: '',
      // 初始化链接数组为空
      links: [],
      // 初始化链接元数据数组为空
      links_meta: [],
      // 初始化行数组为空
      lines: [],
      // 初始化行源对象为空
      line_source: {},
    };
  }

  // 推送新的页面状态到状态栈并返回该状态
  private pushPageState(): PageState {
    // 调用 newPageState 创建一个新的页面状态
    let state = this.newPageState();
    // 将新状态推入状态栈
    this.state.pageStack.push(state);
    // 返回新创建的页面状态
    return state;
  }

  // 获取状态栈中的最后一个页面状态
  private lastPageState(): PageState {
    // 如果状态栈为空，抛出错误
    if (this.state.pageStack.length === 0) {
      throw new Error('No page state');
    }
    // 返回状态栈中的最后一个页面状态
    return this.state.pageStack[this.state.pageStack.length - 1];
  }

  // 格式化错误 URL，限制其长度
  private formatErrorUrl(url: string): string {
    // 定义截断限制为 80 个字符
    let TRUNCATION_LIMIT = 80;
    // 如果 URL 长度小于等于限制，直接返回该 URL
    if (url.length <= TRUNCATION_LIMIT) {
      return url;
    }
    // 如果 URL 超过限制，截断并返回格式化的字符串
    return url.slice(0, TRUNCATION_LIMIT) + `... (URL truncated at ${TRUNCATION_LIMIT} chars)`;
  }

  // 定义一个包含异步搜索功能的对象
  protected functions = {
    // 异步搜索函数，接收查询字符串和最近几天的参数，默认值为 -1
    search: async (query: string, recency_days: number = -1) => {
      // 记录调试信息，显示正在搜索的内容
      logger.debug(`Searching for: ${query}`);
      // 创建 URL 查询参数对象，包含搜索查询
      const search = new URLSearchParams({ q: query });
      // 如果 recency_days 大于 0，添加相应的查询参数
      recency_days > 0 && search.append('recency_days', recency_days.toString());
      // 如果自定义配置 ID 存在，添加相应的查询参数
      if (config.CUSTOM_CONFIG_ID) {
    search.append('customconfig', config.CUSTOM_CONFIG_ID.toString());
    },
    # 定义一个打开 URL 的函数，接受一个字符串类型的 URL
        open_url: (url: string) => {
          # 记录调试信息，输出当前打开的 URL
          logger.debug(`Opening ${url}`);
    
          # 设置超时限制，并发起网络请求，获取响应文本
          return withTimeout(
            config.BROWSER_TIMEOUT,
            fetch(url).then(res => res.text()),
          )
            # 处理请求响应，提取返回值和耗时
            .then(async ({ value: res, time }) => {
              try {
                # 获取当前页面状态，并记录 URL 信息
                const state = this.pushPageState();
                state.url = new URL(url);  # 创建 URL 对象
                state.url_string = url;     # 存储原始 URL 字符串
                state.hostname = state.url.hostname;  # 提取主机名
    
                const html = res;  # 保存响应的 HTML 内容
                const dom = new JSDOM(html);  # 将 HTML 内容解析为 DOM 对象
                const title = dom.window.document.title;  # 获取页面标题
                const markdown = this.transform(dom);  # 转换 DOM 为 Markdown 格式
    
                state.title = title;  # 保存标题到状态
    
                # 移除第一行，因为它将作为标题
                const lines = markdown.split('\n');  # 按行分割 Markdown 内容
                lines.shift();  # 移除第一行
                # 移除后续的空行
                let i = 0;  
                while (i < lines.length - 1) {
                  if (lines[i].trim() === '' && lines[i + 1].trim() === '') {
                    lines.splice(i, 1);  # 删除连续的空行
                  } else {
                    i++;  # 移动到下一行
                  }
                }
    
                let page = lines.join('\n');  # 将处理后的行重新组合为字符串
    
                # 第一个换行符不是错误
                let text_result = `\nURL: ${url}\n${page}`;  # 创建结果字符串，包含 URL 和页面内容
                state.lines = text_result.split('\n');  # 将结果按行分割
    
                # 所有行只来自一个来源
                state.line_source = {};  # 初始化行来源对象
                state.line_source[`0-${state.lines.length - 1}`] = {
                  type: 'webpage',  # 设置行来源类型
                  title: title,     # 保存页面标题
                  url: url,        # 保存页面 URL
                  lines: state.lines,  # 保存行内容
                };
    
                let message = this.formatPage(state);  # 格式化页面状态为消息
    
                const returnContentType = 'browser_result';  # 定义返回内容类型
                return {
                  contentType: returnContentType,  # 返回内容类型
                  roleMetadata: returnContentType,  # 返回角色元数据
                  message,  # 返回格式化消息
                  metadataList: state.links_meta,  # 返回链接元数据
                };
              } catch (err) {
                # 捕获解析错误，抛出新的错误信息
                throw new Error(`parse error: ${err}`);
              }
            })
            # 捕获请求错误并进行处理
            .catch(err => {
              logger.error(err.message);  # 记录错误信息
              if (err.code === 'ECONNABORTED') {
                # 如果是超时错误，抛出超时信息
                throw new Error(`Timeout while loading page w/ URL: ${url}`);
              }
              # 否则抛出加载失败信息
              throw new Error(`Failed to load page w/ URL: ${url}`);
            });
        },
        },
      };
    
      # 构造函数初始化状态
      constructor() {
        this.state =  {
          pageStack: [],  # 页面栈初始化为空
          quotes: {},     # 初始化引用对象为空
          quoteCounter: 7,  # 初始化引用计数器
        };
    
        # 移除 turndown 服务中的 script 和 style 标签
        this.turndownService.remove('script');
        this.turndownService.remove('style');
    
        # 为 turndown 添加规则
    // 为 'reference' 类型的链接添加解析规则
    this.turndownService.addRule('reference', {
      // 过滤函数，判断节点是否为符合条件的链接
      filter: function (node, options: any): boolean {
        return (
          // 只有在使用内联样式时才返回 true
          options.linkStyle === 'inlined' &&
          // 节点必须是 'A' 标签
          node.nodeName === 'A' &&
          // 'href' 属性必须存在
          node.getAttribute('href') !== undefined
        );
      },

      // 替换函数，用于生成特定格式的链接
      replacement: (content, node, options): string => {
        // 获取当前页面状态的最新记录
        let state = this.state.pageStack[this.state.pageStack.length - 1];
        // 如果内容为空或节点没有 'getAttribute' 方法，则返回空字符串
        if (!content || !('getAttribute' in node)) return '';
        let href = undefined;
        try {
          // 确保节点具有 'getAttribute' 方法
          if ('getAttribute' in node) {
            // 从 'href' 属性中提取主机名
            const hostname = new URL(node.getAttribute('href')!).hostname;
            // 如果主机名与当前状态的主机名相同或不存在，则不附加主机名
            if (hostname === state.hostname || !hostname) {
              href = '';
            } else {
              // 否则，附加主机名
              href = '†' + hostname;
            }
          }
        } catch (e) {
          // 捕获异常以避免显示错误的链接
          href = '';
        }
        // 如果 href 仍然未定义，则返回空字符串
        if (href === undefined) return '';

        // 获取链接的完整 URL
        const url = node.getAttribute('href')!;
        // 查找当前链接在状态中的索引
        let linkId = state.links.findIndex(link => link === url);
        // 如果链接不存在，则为其分配新的 ID
        if (linkId === -1) {
          linkId = state.aCounter++;
          // logger.debug(`New link[${linkId}]: ${url}`);
          // 将新链接的元数据推入状态中
          state.links_meta.push({
            type: 'webpage',
            title: node.textContent!,
            url: href,
            text: node.textContent!,
          });
          // 将新链接添加到状态链接数组中
          state.links.push(url);
        }
        // 返回格式化的链接字符串
        return `【${linkId}†${node.textContent}${href}】`;
      },
    });
    // 为 'img' 标签添加解析规则
    this.turndownService.addRule('img', {
      // 过滤条件，指定过滤 'img' 标签
      filter: 'img',

      // 替换函数，用于生成特定格式的图像标记
      replacement: (content, node, options): string => {
        // 获取当前页面状态的最新记录
        let state = this.state.pageStack[this.state.pageStack.length - 1];
        // 返回格式化的图像标记字符串
        return `[Image ${state.imgCounter++}]`;
      },
    });
    // 为 'li' 标签添加解析规则，并调整缩进
    this.turndownService.addRule('list', {
      // 过滤条件，指定过滤 'li' 标签
      filter: 'li',

      // 替换函数，用于生成特定格式的列表项
      replacement: function (content, node, options) {
        // 清理内容的换行符
        content = content
          .replace(/^\n+/, '') // 移除开头的多余换行符
          .replace(/\n+$/, '\n') // 将结尾的多余换行符替换为一个换行符
          .replace(/\n/gm, '\n  '); // 在每行前添加缩进

        // 确定列表前缀符号
        let prefix = options.bulletListMarker + ' ';
        // 获取父节点，确保是列表
        const parent = node.parentNode! as Element;
        // 如果父节点是有序列表，计算索引
        if (parent.nodeName === 'OL') {
          const start = parent.getAttribute('start');
          const index = Array.prototype.indexOf.call(parent.children, node);
          // 根据列表的起始值调整前缀
          prefix = (start ? Number(start) + index : index + 1) + '.  ';
        }
        // 返回格式化的列表项字符串，处理换行
        return '  ' + prefix + content + (node.nextSibling && !/\n$/.test(content) ? '\n' : '');
      },
    });
    // 为 'strong' 和 'b' 标签添加解析规则，移除加粗效果
    this.turndownService.addRule('emph', {
      // 过滤条件，指定过滤 'strong' 和 'b' 标签
      filter: ['strong', 'b'],

      // 替换函数，返回原始内容
      replacement: function (content, node, options) {
        // 如果内容为空，则返回空字符串
        if (!content.trim()) return '';
        // 返回原始内容
        return content;
      },
    });
  }

  // 定义抽象方法，用于处理每一行内容并返回一个或多个 ActionResult
  abstract actionLine(content: string): Promise<ActionResult | ActionResult[]>;

  // 异步方法，处理传入的内容并返回 ToolObservation 数组
  async action(content: string): Promise<ToolObservation[]> {
    // 将内容按行分割成数组
    const lines = content.split('\n');
    // 初始化结果数组，用于存储 ActionResult
    let results: ActionResult[] = [];
    // 遍历每一行
    for (const line of lines) {
      // 记录当前处理的行信息
      logger.info(`Action line: ${line}`)
      try {
        // 调用 actionLine 方法处理当前行，并等待结果
        const lineActionResult = await this.actionLine(line);
        // 记录当前行的处理结果
        logger.debug(`Action line result: ${JSON.stringify(lineActionResult, null, 2)}`);
        // 检查结果是否为数组
        if (Array.isArray(lineActionResult)) {
          // 将数组结果合并到 results 中
          results = results.concat(lineActionResult);
        } else {
          // 将单个结果添加到 results 中
          results.push(lineActionResult);
        }
      } catch (err) {
        // 定义错误内容类型
        const returnContentType = 'system_error';
        // 将错误信息封装到结果中
        results.push({
          contentType: returnContentType,
          roleMetadata: returnContentType,
          message: `Error when executing command ${line}\n${err}`,
          metadata: {
            failedCommand: line,
          },
        });
      }
    }
    // 初始化观察结果数组
    const observations: ToolObservation[] = [];
    // 遍历每个 ActionResult 以生成 ToolObservation
    for (const result of results) {
      // 构建观察对象
      const observation: ToolObservation = {
        contentType: result.contentType,
        result: result.message,
        roleMetadata: result.roleMetadata,
        metadata: result.metadata ?? {},
      };

      // 如果结果中有 metadataList，将其添加到观察对象的 metadata 中
      if (result.metadataList) {
        observation.metadata.metadata_list = result.metadataList;
      }
      // 将观察对象添加到观察结果数组中
      observations.push(observation);
    }
    // 返回所有观察结果
    return observations;
  }

  // 后处理方法，用于处理消息和元数据
  postProcess(message: Message, metadata: any) {
    // 正则模式，用于匹配引用内容
    const quotePattern = /【(.+?)†(.*?)】/g;
    // 获取消息内容
    const content = message.content;
    // 初始化匹配变量
    let match;
    // 初始化引用数组
    let citations: Citation[] = [];
    // 定义引用格式类型
    const citation_format_type = 'tether_og';
    // 当匹配到引文模式时循环处理
    while ((match = quotePattern.exec(content))) {
      // 记录当前匹配的引文
      logger.debug(`Citation match: ${match[0]}`);
      // 获取匹配的起始索引
      const start_ix = match.index;
      // 获取匹配的结束索引
      const end_ix = match.index + match[0].length;

      // 初始化无效原因为 undefined
      let invalid_reason = undefined;
      // 声明元数据变量，类型为 TetherQuoteMetadata
      let metadata: TetherQuoteMetadata;
      // 尝试块，处理引文解析
      try {
        // 解析被引用消息的索引
        let cited_message_idx = parseInt(match[1]);
        // 获取证据文本
        let evidence_text = match[2];
        // 从状态中获取引用内容
        let quote = this.state.quotes[cited_message_idx.toString()];
        // 如果引用未定义，记录无效原因
        if (quote === undefined) {
          invalid_reason = `'Referenced message ${cited_message_idx} in citation 【${cited_message_idx}†${evidence_text}】 is not a quote or tether browsing display.'`;
          // 记录错误信息
          logger.error(`Triggered citation error with quote undefined: ${invalid_reason}`);
          // 将无效引文信息推入 citations 数组
          citations.push({
            citation_format_type,
            start_ix,
            end_ix,
            invalid_reason,
          });
        } else {
          // 定义额外信息
          let extra: TetherQuoteExtra = {
            cited_message_idx,
            evidence_text,
          };
          // 获取引用的元数据
          const quote_metadata = quote.metadata[0];
          // 构造引文元数据对象
          metadata = {
            type: 'webpage',
            title: quote_metadata.title,
            url: quote_metadata.url,
            text: quote_metadata.lines.join('\n'),
            extra,
          };
          // 将有效引文信息推入 citations 数组
          citations.push({
            citation_format_type,
            start_ix,
            end_ix,
            metadata,
          });
        }
      } catch (err) {
        // 记录异常信息
        logger.error(`Triggered citation error: ${err}`);
        // 记录无效原因为捕获的异常
        invalid_reason = `Citation Error: ${err}`;
        // 将无效引文信息推入 citations 数组
        citations.push({
          start_ix,
          end_ix,
          citation_format_type,
          invalid_reason,
        });
      }
    }
    // 将引文数组添加到元数据中
    metadata.citations = citations;
  }

  // 获取当前状态
  getState() {
    // 返回状态对象
    return this.state;
  }
} // 结束类或块的作用域

export class SimpleBrowser extends BaseBrowser { // 定义一个名为 SimpleBrowser 的类，继承自 BaseBrowser
  public description = 'SimpleBrowser'; // 声明一个公开属性 description，值为 'SimpleBrowser'

  constructor() { // 构造函数
    super(); // 调用父类的构造函数
  }

  async actionLine(content: string): Promise<ActionResult | ActionResult[]> { // 异步方法 actionLine，接受一个字符串参数 content，返回 ActionResult 或 ActionResult 数组
    const regex = /(\w+)\(([^)]*)\)/; // 正则表达式，用于匹配函数名和参数
    const matches = content.match(regex); // 在 content 中查找匹配项

    if (matches) { // 如果找到匹配项
      const functionName = matches[1]; // 提取函数名
      let args_string = matches[2]; // 提取参数字符串
      if (functionName === 'mclick') { // 如果函数名为 'mclick'
        args_string = args_string.trim().slice(1, -1); // 去除参数字符串的 '[' 和 ']'
      }

      const args = args_string.split(',').map(arg => arg.trim()); // 将参数字符串按逗号分割，并去除空格

      let result; // 声明结果变量
      switch (functionName) { // 根据函数名执行不同的逻辑
        case 'search': // 如果函数名为 'search'
          logger.debug(`SimpleBrowser action search ${args[0].slice(1, -1)}`); // 记录调试信息
          const recency_days = /(^|\D)(\d+)($|\D)/.exec(args[1])?.[2] as undefined | `${number}`; // 提取 recency_days 参数
          result = await this.functions.search( // 调用 functions 对象的 search 方法
            args[0].slice(1, -1), // 去除查询字符串的引号
            recency_days && Number(recency_days), // 如果 recency_days 存在，则转换为数字
          );
          break; // 结束 switch 语句
        case 'open_url': // 如果函数名为 'open_url'
          logger.debug(`SimpleBrowser action open_url ${args[0].slice(1, -1)}`); // 记录调试信息
          result = await this.functions.open_url(args[0].slice(1, -1)); // 调用 functions 对象的 open_url 方法
          break; // 结束 switch 语句
        case 'mclick': // 如果函数名为 'mclick'
          logger.debug(`SimpleBrowser action mclick ${args}`); // 记录调试信息
          result = await this.functions.mclick(args.map(x => parseInt(x))); // 调用 functions 对象的 mclick 方法，传入解析后的参数
          break; // 结束 switch 语句
        default: // 如果没有匹配的函数名
          throw new Error(`Parse Error: ${content}`); // 抛出解析错误
      }

      return result; // 返回结果
    } else { // 如果没有找到匹配项
      throw new Error('Parse Error'); // 抛出解析错误
    }
  }
}

if (require.main === module) { // 如果当前模块是主模块
  (async () => { // 定义并立即执行一个异步函数
    let browser = new SimpleBrowser(); // 实例化 SimpleBrowser 对象
    let demo = async (action: string) => { // 定义一个异步函数 demo，接受一个字符串参数 action
      logger.info(` ------ Begin of Action: ${action} ------`); // 记录操作开始信息
      let results = await browser.action(action); // 调用 browser 对象的 action 方法，获取结果
      for (const [idx, result] of results.entries()) { // 遍历结果数组
        logger.info(`[Result ${idx}] contentType: ${result.contentType}`); // 记录结果的 contentType
        logger.info(`[Result ${idx}] roleMetadata: ${result.roleMetadata}`); // 记录结果的 roleMetadata
        logger.info(`[Result ${idx}] result: ${result.result}`); // 记录结果
        logger.info(`[Result ${idx}] metadata: ${JSON.stringify(result.metadata, null, 2)}`); // 记录结果的 metadata
      }
      logger.info(` ------ End of Action: ${action} ------\n\n`); // 记录操作结束信息
    };

    await demo("search('Apple Latest News')"); // 执行搜索操作
    await demo('mclick([0, 1, 5, 6])'); // 执行 mclick 操作
    await demo('mclick([1, 999999])'); // 执行 mclick 操作，包含超出范围的索引
    await demo("open_url('https://chatglm.cn')"); // 执行打开 URL 操作
    await demo("search('zhipu latest News')"); // 执行搜索操作
    await demo('mclick([0, 1, 5, 6])'); // 再次执行 mclick 操作
  })(); // 结束立即执行的异步函数
}
```