# `.\DB-GPT-src\web\components\chat\chat-content\config.tsx`

```py
import { LinkOutlined, ReadOutlined, SyncOutlined } from '@ant-design/icons';
import ReactMarkdown from 'react-markdown';
import { Table, Image, Tag, Tabs, TabsProps, Popover } from 'antd';
import { Reference } from '@/types/chat';
import { AutoChart, BackEndChartType, getChartType } from '@/components/chart';
import { CodePreview } from './code-preview';
import { Datum } from '@antv/ava';
import rehypeRaw from 'rehype-raw';
import { IChunk } from '@/types/knowledge';
import AgentPlans from './agent-plans';
import AgentMessages from './agent-messages';
import VisConvertError from './vis-convert-error';
import VisChart from './vis-chart';
import VisDashboard from './vis-dashboard';
import VisPlugin from './vis-plugin';
import VisCode from './vis-code';
import { formatSql } from '@/utils';

type MarkdownComponent = Parameters<typeof ReactMarkdown>['0']['components'];

const customeTags: (keyof JSX.IntrinsicElements)[] = ['custom-view', 'chart-view', 'references', 'summary'];

/**
 * @description
 * Extracts custom tags from Markdown content for special rendering and returns both
 * the updated content and the matched tag values.
 */
function matchCustomeTagValues(context: string) {
  const matchValues = customeTags.reduce<string[]>((acc, tagName) => {
    // Regular expression to match the custom tags in the Markdown content
    const tagReg = new RegExp(`<${tagName}[^>]*\/?>`, 'gi');
    // Replace matched tags with an empty string and collect them in 'acc'
    context = context.replace(tagReg, (matchVal) => {
      acc.push(matchVal);
      return '';
    });
    return acc;
  }, []);
  return { context, matchValues };
}

const basicComponents: MarkdownComponent = {
  /**
   * @description
   * Renders code blocks in Markdown with special handling for specific languages
   * like 'agent-plans', 'agent-messages', 'vis-convert-error', and 'vis-dashboard'.
   */
  code({ inline, node, className, children, style, ...props }) {
    const content = String(children);
    const { context, matchValues } = matchCustomeTagValues(content);
    // Determine the language of the code block
    const lang = className?.replace('language-', '') || 'javascript';

    // Special handling for 'agent-plans' language
    if (lang === 'agent-plans') {
      try {
        // Parse JSON content specific to 'agent-plans' component
        const data = JSON.parse(content) as Parameters<typeof AgentPlans>[0]['data'];
        // Render 'AgentPlans' component with parsed data
        return <AgentPlans data={data} />;
      } catch (e) {
        // Render generic code preview if parsing fails
        return <CodePreview language={lang} code={content} />;
      }
    }

    // Special handling for 'agent-messages' language
    if (lang === 'agent-messages') {
      try {
        // Parse JSON content specific to 'agent-messages' component
        const data = JSON.parse(content) as Parameters<typeof AgentMessages>[0]['data'];
        // Render 'AgentMessages' component with parsed data
        return <AgentMessages data={data} />;
      } catch (e) {
        // Render generic code preview if parsing fails
        return <CodePreview language={lang} code={content} />;
      }
    }

    // Special handling for 'vis-convert-error' language
    if (lang === 'vis-convert-error') {
      try {
        // Parse JSON content specific to 'vis-convert-error' component
        const data = JSON.parse(content) as Parameters<typeof VisConvertError>[0]['data'];
        // Render 'VisConvertError' component with parsed data
        return <VisConvertError data={data} />;
      } catch (e) {
        // Render generic code preview if parsing fails
        return <CodePreview language={lang} code={content} />;
      }
    }

    // Special handling for 'vis-dashboard' language
    if (lang === 'vis-dashboard') {
      try {
        // Parse JSON content specific to 'vis-dashboard' component
        const data = JSON.parse(content) as Parameters<typeof VisDashboard>[0]['data'];
        // Render 'VisDashboard' component with parsed data
        return <VisDashboard data={data} />;
      } catch (e) {
        // Render generic code preview if parsing fails
        return <CodePreview language={lang} code={content} />;
      }
    }

    // Default rendering for other languages with generic code preview
    return <CodePreview language={lang} code={content} />;
  }
};
    // 如果语言类型为 'vis-chart'
    if (lang === 'vis-chart') {
      try {
        // 尝试解析内容为 JSON，获取 VisChart 组件所需的 data
        const data = JSON.parse(content) as Parameters<typeof VisChart>[0]['data'];
        // 返回渲染 VisChart 组件的 JSX
        return <VisChart data={data} />;
      } catch (e) {
        // 解析失败时返回展示错误代码预览的 JSX
        return <CodePreview language={lang} code={content} />;
      }
    }

    // 如果语言类型为 'vis-plugin'
    if (lang === 'vis-plugin') {
      try {
        // 尝试解析内容为 JSON，获取 VisPlugin 组件所需的 data
        const data = JSON.parse(content) as Parameters<typeof VisPlugin>[0]['data'];
        // 返回渲染 VisPlugin 组件的 JSX
        return <VisPlugin data={data} />;
      } catch (e) {
        // 解析失败时返回展示错误代码预览的 JSX
        return <CodePreview language={lang} code={content} />;
      }
    }

    // 如果语言类型为 'vis-code'
    if (lang === 'vis-code') {
      try {
        // 尝试解析内容为 JSON，获取 VisCode 组件所需的 data
        const data = JSON.parse(content) as Parameters<typeof VisCode>[0]['data'];
        // 返回渲染 VisCode 组件的 JSX
        return <VisCode data={data} />;
      } catch (e) {
        // 解析失败时返回展示错误代码预览的 JSX
        return <CodePreview language={lang} code={content} />;
      }
    }

    // 如果以上条件均不满足，则渲染默认的内容
    return (
      <>
        {!inline ? (
          // 如果不是内联模式，则展示代码预览组件
          <CodePreview code={context} language={lang} />
        ) : (
          // 如果是内联模式，则展示带有样式的 <code> 元素
          <code {...props} style={style} className="p-1 mx-1 rounded bg-theme-light dark:bg-theme-dark text-sm">
            {children}
          </code>
        )}
        {/* 使用 ReactMarkdown 渲染 markdown 内容 */}
        <ReactMarkdown components={markdownComponents} rehypePlugins={[rehypeRaw]}>
          {matchValues.join('\n')}
        </ReactMarkdown>
      </>
    );
  },
  // 定义 <ul> 元素的渲染方法
  ul({ children }) {
    return <ul className="py-1">{children}</ul>;
  },
  // 定义 <ol> 元素的渲染方法
  ol({ children }) {
    return <ol className="py-1">{children}</ol>;
  },
  // 定义 <li> 元素的渲染方法
  li({ children, ordered }) {
    return <li className={`text-sm leading-7 ml-5 pl-2 text-gray-600 dark:text-gray-300 ${ordered ? 'list-decimal' : 'list-disc'}`}>{children}</li>;
  },
  // 定义 <table> 元素的渲染方法
  table({ children }) {
    return (
      <table className="my-2 rounded-tl-md rounded-tr-md max-w-full bg-white dark:bg-gray-800 text-sm rounded-lg overflow-hidden">{children}</table>
    );
  },
  // 定义 <thead> 元素的渲染方法
  thead({ children }) {
    return <thead className="bg-[#fafafa] dark:bg-black font-semibold">{children}</thead>;
  },
  // 定义 <th> 元素的渲染方法
  th({ children }) {
    return <th className="!text-left p-4">{children}</th>;
  },
  // 定义 <td> 元素的渲染方法
  td({ children }) {
    return <td className="p-4 border-t border-[#f0f0f0] dark:border-gray-700">{children}</td>;
  },
  // 定义 <h1> 元素的渲染方法
  h1({ children }) {
    return <h3 className="text-2xl font-bold my-4 border-b border-slate-300 pb-4">{children}</h3>;
  },
  // 定义 <h2> 元素的渲染方法
  h2({ children }) {
    return <h3 className="text-xl font-bold my-3">{children}</h3>;
  },
  // 定义 <h3> 元素的渲染方法
  h3({ children }) {
    return <h3 className="text-lg font-semibold my-2">{children}</h3>;
  },
  // 定义 <h4> 元素的渲染方法
  h4({ children }) {
    return <h3 className="text-base font-semibold my-1">{children}</h3>;
  },
  // 定义 <a> 元素的渲染方法
  a({ children, href }) {
    return (
      <div className="inline-block text-blue-600 dark:text-blue-400">
        <LinkOutlined className="mr-1" />
        <a href={href} target="_blank">
          {children}
        </a>
      </div>
    );
  },
  // 定义 <img> 元素的渲染方法
  img({ src, alt }) {
  return (
    // 返回一个包含图片组件的 div 元素
    <div>
      {/* 图片组件，用于显示指定的图片 */}
      <Image
        // CSS 类名，设置最小高度为 1rem，宽度和高度自适应，带有边框和圆角
        className="min-h-[1rem] max-w-full max-h-full border rounded"
        // 图片的源地址
        src={src}
        // 图片的替代文本
        alt={alt}
        // 图片加载时的占位元素，显示加载中的标签
        placeholder={
          <Tag icon={<SyncOutlined spin />} color="processing">
            Image Loading...
          </Tag>
        }
        // 图片加载失败时显示的备用图像
        fallback="/images/fallback.png"
      />
    </div>
  );
},

blockquote({ children }) {
  return (
    // 返回一个带有引用样式的 blockquote 元素
    <blockquote
      // 设置内边距和边框样式
      className="py-4 px-6 border-l-4 border-blue-600 rounded bg-white my-2 text-gray-500 dark:bg-slate-800 dark:text-gray-200 dark:border-white shadow-sm"
    >
      {children}
    </blockquote>
  );
},
};

const extraComponents: MarkdownComponent = {
  // 自定义组件处理 'chart-view'，接收内容和子组件
  'chart-view': function ({ content, children }) {
    // 定义数据对象，包括数据、类型和 SQL 查询
    let data: {
      data: Datum[]; // 数据数组
      type: BackEndChartType; // 后端图表类型
      sql: string; // SQL 查询字符串
    };
    try {
      // 尝试解析内容为 JSON 数据
      data = JSON.parse(content as string);
    } catch (e) {
      // 解析失败时，记录错误并使用默认值
      console.log(e, content);
      data = {
        type: 'response_table', // 默认类型为响应表
        sql: '', // 默认空 SQL 查询
        data: [], // 空数据数组
      };
    }

    // 根据数据是否存在，生成列信息数组
    const columns = data?.data?.[0]
      ? Object.keys(data?.data?.[0])?.map((item) => {
          return {
            title: item, // 列标题
            dataIndex: item, // 数据索引
            key: item, // 键值
          };
        })
      : [];

    // 定义图表项对象
    const ChartItem = {
      key: 'chart', // 键值为 'chart'
      label: 'Chart', // 标签为 'Chart'
      children: <AutoChart data={data?.data} chartType={getChartType(data?.type)} />, // 自动图表组件
    };
    // 定义 SQL 项对象
    const SqlItem = {
      key: 'sql', // 键值为 'sql'
      label: 'SQL', // 标签为 'SQL'
      children: <CodePreview code={formatSql(data?.sql, 'mysql')} language="sql" />, // SQL 代码预览组件
    };
    // 定义数据项对象
    const DataItem = {
      key: 'data', // 键值为 'data'
      label: 'Data', // 标签为 'Data'
      children: <Table dataSource={data?.data} columns={columns}  scroll={{x:true}} />, // 表格组件
    };
    // 根据数据类型确定选项卡数组
    const TabItems: TabsProps['items'] = data?.type === 'response_table' ? [DataItem, SqlItem] : [ChartItem, SqlItem, DataItem];

    // 返回最终的组件结构，包含选项卡和子组件
    return (
      <div>
        <Tabs defaultActiveKey={data?.type === 'response_table' ? 'data' : 'chart'} items={TabItems} size="small" />
        {children}
      </div>
    );
  },

  // 处理 'references' 自定义组件，接收标题、引用和子组件
  references: function ({ title, references, children }) {
    let referenceData;

    // 兼容低版本，从子组件中读取数据
    if (children) {
      try {
        referenceData = JSON.parse(children as string);
        title = referenceData.title; // 更新标题
        references = referenceData.references; // 更新引用
      } catch (error) {
        // 解析失败时记录错误并返回渲染错误信息
        console.log('parse references failed', error);
        return <p className="text-sm text-red-500">Render Reference Error!</p>;
      }
    } else {
      // 新版本，从标签属性中读取引用数据
      try {
        references = JSON.parse(references as string); // 解析引用
      } catch (error) {
        // 解析失败时记录错误并返回渲染错误信息
        console.log('parse references failed', error);
        return <p className="text-sm text-red-500">Render Reference Error!</p>;
      }
    }

    // 如果没有引用或引用长度小于 1，则返回空
    if (!references || references?.length < 1) {
      return null;
    }

    // 继续处理引用数据的渲染或其他逻辑
    return (
      <div className="border-t-[1px] border-gray-300 mt-3 py-2">
        {/* 包含一个段落和参考链接的容器 */}
        <p className="text-sm text-gray-500 dark:text-gray-400 mb-2">
          {/* 标题链接 */}
          <LinkOutlined className="mr-2" />
          {/* 标题文字 */}
          <span className="font-semibold">{title}</span>
        </p>
        {/* 显示参考文献列表 */}
        {references.map((reference: Reference, index: number) => (
          <div key={`file_${index}`} className="text-sm font-normal block ml-2 h-6 leading-6 overflow-hidden">
            {/* 每个参考文献的内容 */}
            <span className="inline-block w-6">[{index + 1}]</span>
            {/* 显示参考文献名称 */}
            <span className="mr-2 lg:mr-4 text-blue-400">{reference.name}</span>
            {/* 显示参考文献中的段落或者ID */}
            {reference?.chunks?.map((chunk: IChunk | number, index) => (
              <span key={`chunk_${index}`}>
                {/* 如果是对象，显示弹出框 */}
                {typeof chunk === 'object' ? (
                  <Popover
                    content={
                      <div className="max-w-4xl">
                        {/* 弹出框中的内容部分 */}
                        <p className="mt-2 font-bold mr-2 border-t border-gray-500 pt-2">Content:</p>
                        <p>{chunk?.content || 'No Content'}</p>
                        <p className="mt-2 font-bold mr-2 border-t border-gray-500 pt-2">MetaData:</p>
                        <p>{chunk?.meta_info || 'No MetaData'}</p>
                        <p className="mt-2 font-bold mr-2 border-t border-gray-500 pt-2">Score:</p>
                        <p>{chunk?.recall_score || ''}</p>
                      </div>
                    }
                    title="Chunk Information"
                  >
                    {/* 弹出框触发按钮 */}
                    <span className="cursor-pointer text-blue-500 ml-2" key={`chunk_content_${chunk?.id}`}>
                      {chunk?.id}
                    </span>
                  </Popover>
                ) : (
                  {/* 如果不是对象，直接显示ID */}
                  <span className="cursor-pointer text-blue-500 ml-2" key={`chunk_id_${chunk}`}>
                    {chunk}
                  </span>
                )}
                {/* 如果不是最后一个元素，显示逗号分隔符 */}
                {index < reference?.chunks.length - 1 && <span key={`chunk_comma_${index}`}>,</span>}
              </span>
            ))}
          </div>
        ))}
      </div>
    );
  },
  summary: function ({ children }) {
    return (
      <div>
        {/* 文档摘要的容器 */}
        <p className="mb-2">
          {/* 摘要图标 */}
          <ReadOutlined className="mr-2" />
          {/* 摘要标题 */}
          <span className="font-semibold">Document Summary</span>
        </p>
        {/* 显示摘要内容 */}
        <div>{children}</div>
      </div>
    );
  },
};

const markdownComponents = {
  ...basicComponents,
  ...extraComponents,
};

export default markdownComponents;



// 定义一个空对象字面量结尾的语句
};

// 定义一个常量 markdownComponents，它是由两个对象的属性扩展而来：basicComponents 和 extraComponents
const markdownComponents = {
  ...basicComponents,
  ...extraComponents,
};

// 导出 markdownComponents 对象作为模块的默认输出
export default markdownComponents;
```