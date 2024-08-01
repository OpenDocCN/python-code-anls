# `.\DB-GPT-src\web\components\chat\monaco-editor.tsx`

```py
import * as monaco from 'monaco-editor/esm/vs/editor/editor.api.js';
// 导入 Monaco Editor 的模块和 API

import Editor, { OnChange, loader } from '@monaco-editor/react';
// 导入 Monaco Editor 的 React 组件和相关回调

import classNames from 'classnames';
// 导入用于处理 CSS 类名的工具函数

import { useContext, useMemo } from 'react';
// 导入 React 的 useContext 和 useMemo 钩子

import { formatSql } from '@/utils';
// 导入格式化 SQL 的工具函数

import { getModelService } from './ob-editor/service';
// 导入获取模型服务的函数

import { useLatest } from 'ahooks';
// 导入用于获取最新值的自定义钩子

import { ChatContext } from '@/app/chat-context';
// 导入聊天上下文的 React Context

import { github, githubDark } from './ob-editor/theme';
// 导入 Monaco Editor 的两种主题

import { register } from './ob-editor/ob-plugin';
// 导入注册 Monaco 插件的函数

loader.config({ monaco });
// 配置 Monaco Editor 加载器，使用指定的 monaco 对象

export interface ISession {
  getTableList: (schemaName?: string) => Promise<string[]>;
  getTableColumns: (tableName: string) => Promise<{ columnName: string; columnType: string }[]>;
  getSchemaList: () => Promise<string[]>;
}
// 定义会话接口 ISession，包含获取表列表、表列信息和模式列表的异步方法

interface MonacoEditorProps {
  className?: string;
  value: string;
  language: string;
  onChange?: OnChange;
  thoughts?: string;
  session?: ISession;
}
// 定义 MonacoEditor 组件的属性接口，包括 CSS 类名、编辑器内容、语言、变更事件、批注和会话信息

let plugin = null;
// 定义插件变量，初始为 null

monaco.editor.defineTheme('github', github as any);
// 定义名为 'github' 的 Monaco 主题，使用导入的 github 主题配置

monaco.editor.defineTheme('githubDark', githubDark as any);
// 定义名为 'githubDark' 的 Monaco 主题，使用导入的 githubDark 主题配置

export default function MonacoEditor({ className, value, language = 'mysql', onChange, thoughts, session }: MonacoEditorProps) {
  // MonacoEditor 组件的默认导出函数，接收属性为 className、value、language、onChange、thoughts 和 session

  // 合并 value 和 thoughts 的编辑器内容
  const editorValue = useMemo(() => {
    if (language !== 'mysql') {
      return value;
    }
    if (thoughts && thoughts.length > 0) {
      return formatSql(`-- ${thoughts} \n${value}`);
    }
    return formatSql(value);
  }, [value, thoughts]);

  const sessionRef = useLatest(session);
  // 使用 useLatest 钩子获取最新的 session 对象的引用

  const context = useContext(ChatContext);
  // 使用 useContext 钩子获取聊天上下文的值

  async function pluginRegister(editor: monaco.editor.IStandaloneCodeEditor) {
    // 异步函数 pluginRegister，用于注册插件到 Monaco 编辑器

    const plugin = await register();
    // 调用 register 函数异步获取插件实例

    plugin.setModelOptions(
      editor.getModel()?.id || '',
      getModelService({
        modelId: editor.getModel()?.id || '',
        delimiter: ';',
      }, () => sessionRef.current || null)
    );
    // 调用插件的 setModelOptions 方法设置模型选项，包括模型 ID 和获取模型服务的回调函数
  }

  return (
    <Editor
      className={classNames(className)}
      onMount={pluginRegister}
      value={editorValue}
      defaultLanguage={language}
      onChange={onChange}
      theme={context?.mode !== "dark" ? "github" : "githubDark"}
      options={{
        minimap: {
          enabled: false,
        },
        wordWrap: 'on',
      }}
    />
  );
  // 返回 Monaco Editor 组件，设置类名、挂载事件、内容值、默认语言、变更事件、主题和选项
}
```