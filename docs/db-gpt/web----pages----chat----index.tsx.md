# `.\DB-GPT-src\web\pages\chat\index.tsx`

```py
import React, { useContext, useEffect } from 'react';
import { useRouter } from 'next/router';
import { ChatContext } from '@/app/chat-context';
import dynamic from 'next/dynamic';

const DbEditor = dynamic(() => import('@/components/chat/db-editor'), { ssr: false });
const ChatContainer = dynamic(() => import('@/components/chat/chat-container'), { ssr: false });

function Chat() {
  const {
    query: { id, scene },  // 从路由器获取查询参数中的id和scene
  } = useRouter();
  const { isContract, setIsContract, setIsMenuExpand } = useContext(ChatContext);  // 从ChatContext中获取状态和设置函数

  useEffect(() => {
    // 仅初始化执行，防止dashboard页面无法切换状态
    setIsMenuExpand(scene !== 'chat_dashboard');  // 根据scene参数判断是否展开菜单，并设置菜单展开状态

    // 路由变了要取消Editor模式，再进来是默认的Preview模式
    if (id && scene) {
      setIsContract(false);  // 如果id和scene存在，则设置非合约模式
    }
  }, [id, scene]);  // useEffect依赖于id和scene参数的变化

  // 根据isContract状态决定显示DbEditor组件或ChatContainer组件
  return <>{isContract ? <DbEditor /> : <ChatContainer />}</>;
}

export default Chat;
```