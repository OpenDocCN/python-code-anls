# `.\DB-GPT-src\web\components\chat\header\model-selector.tsx`

```py
/**
 * multi-models selector
 */

// 导入所需模块和组件
import { ChatContext } from '@/app/chat-context';  // 导入聊天上下文
import { Select } from 'antd';  // 导入 Ant Design 的 Select 组件
import { MODEL_ICON_MAP } from '@/utils/constants';  // 导入模型图标映射常量
import Image from 'next/image';  // 导入 Next.js 的 Image 组件
import { useContext } from 'react';  // 导入 React 的 useContext 钩子
import { useTranslation } from 'react-i18next';  // 导入 React i18n 的 useTranslation 钩子

// 定义 Props 接口，包含可选的 onChange 回调函数
interface Props {
  onChange?: (model: string) => void;  // 当模型变化时的回调函数
}

// 默认模型图标的 URL
const DEFAULT_ICON_URL = '/models/huggingface.svg';

// 渲染模型图标的函数，接受模型名称和可选的宽高参数
export function renderModelIcon(model?: string, props?: { width: number; height: number }) {
  const { width, height } = props || {};

  // 如果模型名称为空，则返回 null
  if (!model) return null;

  // 返回 Image 组件，显示模型对应的图标
  return (
    <Image
      className="rounded-full border border-gray-200 object-contain bg-white inline-block"
      width={width || 24}
      height={height || 24}
      src={MODEL_ICON_MAP[model]?.icon || DEFAULT_ICON_URL}  // 根据模型名称获取对应图标的 URL
      alt="llm"  // 图标的替代文本
    />
  );
}

// 模型选择器组件定义，接受 onChange 回调函数作为 Props
function ModelSelector({ onChange }: Props) {
  const { t } = useTranslation();  // 获取翻译函数 t
  const { modelList, model } = useContext(ChatContext);  // 使用聊天上下文中的 modelList 和 model

  // 如果模型列表为空或长度为 0，则返回 null，不显示选择器
  if (!modelList || modelList.length <= 0) {
    return null;
  }

  // 返回 Select 组件，显示模型选择器
  return (
    <Select
      value={model}  // 当前选中的模型
      placeholder={t('choose_model')}  // 选择器的占位符文本，使用翻译函数 t 获取
      className="w-52"  // 自定义样式类名，控制选择器的宽度
      onChange={(val) => {
        onChange?.(val);  // 当选择器的值变化时调用外部传入的 onChange 回调函数
      }}
    >
      {/* 遍历模型列表，为每个模型生成一个 Option 选项 */}
      {modelList.map((item) => (
        <Select.Option key={item}>
          <div className="flex items-center">
            {renderModelIcon(item)}  // 调用渲染模型图标的函数，显示模型图标
            <span className="ml-2">{MODEL_ICON_MAP[item]?.label || item}</span>  // 显示模型标签或默认为模型名称
          </div>
        </Select.Option>
      ))}
    </Select>
  );
}

export default ModelSelector;  // 导出模型选择器组件作为默认导出
```