# `.\DB-GPT-src\web\components\chat\header\index.tsx`

```py
import { useContext } from 'react';  // 导入 useContext 函数，用于在函数组件中访问上下文对象
import ChatExcel from './chat-excel';  // 导入 ChatExcel 组件，用于处理 Excel 相关操作
import { ChatContext } from '@/app/chat-context';  // 导入 ChatContext 上下文，提供场景和刷新对话列表的功能
import ModeTab from '@/components/chat/mode-tab';  // 导入 ModeTab 组件，用于显示模式选项卡
import ModelSelector from '@/components/chat/header/model-selector';  // 导入 ModelSelector 组件，用于选择模型
import DBSelector from './db-selector';  // 导入 DBSelector 组件，用于选择数据库
import AgentSelector from './agent-selector';  // 导入 AgentSelector 组件，用于选择代理人

/**
 * chat header
 */
interface Props {
  refreshHistory?: () => Promise<void>;  // 刷新历史记录的回调函数，可选
  modelChange?: (val: string) => void;  // 模型更改的回调函数，可选
}

function Header({ refreshHistory, modelChange }: Props) {
  const { scene, refreshDialogList } = useContext(ChatContext);  // 使用 useContext 获取 ChatContext 上下文中的 scene 和 refreshDialogList

  return (
    <div className="w-full py-2 px-4 md:px-4 flex flex-wrap items-center justify-center gap-1 md:gap-4">
      {/* Models Selector */}
      <ModelSelector onChange={modelChange} />  // 渲染 ModelSelector 组件，传入 modelChange 回调函数

      {/* DB Selector */}
      <DBSelector />  // 渲染 DBSelector 组件，用于选择数据库

      {/* Excel Upload */}
      {scene === 'chat_excel' && (  // 如果场景为 'chat_excel'，则渲染 ChatExcel 组件
        <ChatExcel
          onComplete={() => {
            refreshDialogList?.();  // 执行刷新对话列表的操作，如果 refreshDialogList 存在的话
            refreshHistory?.();  // 执行刷新历史记录的操作，如果 refreshHistory 存在的话
          }}
        />
      )}

      {/* Agent Selector */}
      {scene === 'chat_agent' && <AgentSelector />}  // 如果场景为 'chat_agent'，则渲染 AgentSelector 组件

      <ModeTab />  // 渲染 ModeTab 组件，显示模式选项卡
    </div>
  );
}

export default Header;  // 导出 Header 组件作为默认导出
```