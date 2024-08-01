# `.\DB-GPT-src\web\components\common\completion-input.tsx`

```py
// 导入 SendOutlined 组件，用于渲染发送图标
import { SendOutlined } from '@ant-design/icons';
// 导入 Button 和 Input 组件，用于渲染按钮和输入框
import { Button, Input } from 'antd';
// 导入 React 相关的 hooks 和类型
import { PropsWithChildren, useContext, useEffect, useMemo, useRef, useState } from 'react';
// 导入 PromptBot 组件，用于提示框相关功能
import PromptBot from './prompt-bot';
// 导入 DocUpload 组件，用于文档上传功能
import DocUpload from '../chat/doc-upload';
// 导入 DocList 组件，用于文档列表展示功能
import DocList from '../chat/doc-list';
// 导入 IDocument 类型，用于文档对象的类型定义
import { IDocument } from '@/types/knowledge';
// 导入 ChatContext，用于获取聊天上下文相关信息
import { ChatContext } from '@/app/chat-context';
// 导入 apiInterceptors 和 getDocumentList 函数，用于处理 API 请求和获取文档列表
import { apiInterceptors, getDocumentList } from '@/client/api';

// 定义 TextAreaProps 类型，继承自 Input.TextArea 的参数类型，并排除 value、onPressEnter、onChange、onSubmit 属性
type TextAreaProps = Omit<Parameters<typeof Input.TextArea>[0], 'value' | 'onPressEnter' | 'onChange' | 'onSubmit'>;

// 定义 Props 接口，包含 loading 和 onSubmit 函数属性，以及可选的 handleFinish 函数属性
interface Props {
  loading?: boolean;
  onSubmit: (val: string) => void;
  handleFinish?: (val: boolean) => void;
}

// 定义 CompletionInput 组件，接收 PropsWithChildren<Props & TextAreaProps> 参数类型
function CompletionInput({ children, loading, onSubmit, handleFinish, ...props }: PropsWithChildren<Props & TextAreaProps>) {
  // 使用 useContext hook 获取 ChatContext 中的 dbParam 和 scene 属性
  const { dbParam, scene } = useContext(ChatContext);

  // 定义 useState hook，用于管理用户输入的状态
  const [userInput, setUserInput] = useState('');
  // 使用 useMemo hook 根据 scene 属性值决定是否显示上传组件
  const showUpload = useMemo(() => scene === 'chat_knowledge', [scene]);
  // 定义 useState hook，用于管理文档列表的状态
  const [documents, setDocuments] = useState<IDocument[]>([]);
  // 使用 useRef hook 创建 uploadCountRef 变量，用于记录上传计数
  const uploadCountRef = useRef(0);

  // 使用 useEffect hook 根据 dbParam 变化触发 fetchDocuments 函数
  useEffect(() => {
    showUpload && fetchDocuments();
  }, [dbParam]);

  // 定义异步函数 fetchDocuments，用于获取文档列表
  async function fetchDocuments() {
    // 如果 dbParam 不存在，直接返回 null
    if (!dbParam) {
      return null;
    }
    // 发起 API 请求获取文档列表，并更新文档状态
    const [_, data] = await apiInterceptors(
      getDocumentList(dbParam, {
        page: 1,
        page_size: uploadCountRef.current,
      }),
    );
    setDocuments(data?.data!);
  }

  // 定义 onUploadFinish 函数，用于处理上传完成事件
  const onUploadFinish = async () => {
    // 增加上传计数
    uploadCountRef.current += 1;
    // 执行 fetchDocuments 函数更新文档列表
    await fetchDocuments();
  };

  // 返回组件
  return (
    <div className="flex-1 relative">
      {/* 占据剩余空间的弹性布局容器 */}
      <DocList documents={documents} dbParam={dbParam} />
      {/* 显示文档列表，传递文档和数据库参数 */}
      {showUpload && <DocUpload handleFinish={handleFinish} onUploadFinish={onUploadFinish} className="absolute z-10 top-2 left-2" />}
      {/* 如果显示上传组件，则渲染上传组件 */}
      <Input.TextArea
        className={`flex-1 ${showUpload ? 'pl-10' : ''} pr-10`}
        // 文本输入框，根据是否显示上传组件调整左侧填充样式
        size="large"
        value={userInput}
        // 输入框的值为用户输入内容
        autoSize={{ minRows: 1, maxRows: 4 }}
        // 自动调整高度，最小1行，最大4行
        {...props}
        // 传递其他所有属性给输入框组件
        onPressEnter={(e) => {
          if (!userInput.trim()) return;
          if (e.keyCode === 13) {
            if (e.shiftKey) {
              e.preventDefault()
              setUserInput((state) => state + '\n');
              return;
            }
            onSubmit(userInput);
            setTimeout(() => {
              setUserInput('');
            }, 0);
          }
        }}
        // 处理回车键事件，如果按下 Enter 键且不为空白，则提交输入内容
        onChange={(e) => {
          if (typeof props.maxLength === 'number') {
            setUserInput(e.target.value.substring(0, props.maxLength));
            return;
          }
          setUserInput(e.target.value);
        }}
        // 处理输入内容变化事件，根据 maxLength 截断输入内容
      />
      <Button
        className="ml-2 flex items-center justify-center absolute right-0 bottom-0"
        // 按钮，绝对定位在右下角，用于提交用户输入
        size="large"
        type="text"
        loading={loading}
        icon={<SendOutlined />}
        onClick={() => {
          onSubmit(userInput);
        }}
        // 点击按钮提交用户输入
      />
      <PromptBot
        submit={(prompt) => {
          setUserInput(userInput + prompt);
        }}
        // 交互式提示机器人组件，处理用户输入并提交
      />
      {children}
      {/* 渲染子组件 */}
    </div>
}

export default CompletionInput;
```