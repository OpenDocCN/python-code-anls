# `.\DB-GPT-src\web\components\common\gpt-card.tsx`

```py
# 导入 React 相关组件和库
import React, { HtmlHTMLAttributes, PropsWithChildren, ReactNode, memo, useCallback, useMemo } from 'react';
# 导入 antd 组件中的 Tag 和 Tooltip
import { Tag, TagProps, Tooltip } from 'antd';
# 导入 classNames 库
import classNames from 'classnames';
# 导入 next/image 组件
import Image from 'next/image';

# 定义 Props 接口
interface Props {
  title: string;  # 标题
  desc?: string;  # 描述
  disabled?: boolean;  # 是否禁用
  tags?: (  # 标签数组
    | string
    | {
        text: ReactNode;  # 标签文本
        /** @default false */
        border?: boolean;  # 是否有边框
        /** @default default */
        color?: TagProps['color'];  # 标签颜色
      }
  )[];
  operations?: {  # 操作数组
    children: ReactNode;  # 操作子元素
    label?: string;  # 操作标签
    onClick?: () => void;  # 点击操作时的回调函数
  }[];
  icon?: ReactNode;  # 图标
  iconBorder?: boolean;  # 图标是否有边框
  onClick?: () => void;  # 点击卡片时的回调函数
}

# 定义 GPTCard 组件
function GPTCard({
  icon,
  iconBorder = true,
  title,
  desc,
  tags,
  children,
  disabled,
  operations,
  className,
  ...props
}: PropsWithChildren<HtmlHTMLAttributes<HTMLDivElement> & Props) {
  
  # 使用 useMemo 缓存图标节点
  const iconNode = useMemo(() => {
    if (!icon) return null;

    if (typeof icon === 'string') {
      return (
        <Image
          className={classNames('w-11 h-11 rounded-full mr-4 object-contain bg-white', {
            'border border-gray-200': iconBorder,
          })}
          width={48}
          height={48}
          src={icon}
          alt={title}
        />
      );
    }

    return icon;
  }, [icon]);

  # 使用 useMemo 缓存标签节点
  const tagNode = useMemo(() => {
    if (!tags || !tags.length) return null;
    return (
      <div className="flex items-center mt-1 flex-wrap">
        {tags.map((tag, index) => {
          if (typeof tag === 'string') {
            return (
              <Tag key={index} className="text-xs" bordered={false} color="default">
                {tag}
              </Tag>
            );
          }
          return (
            <Tag key={index} className="text-xs" bordered={tag.border ?? false} color={tag.color}>
              {tag.text}
            </Tag>
          );
        })}
      </div>
    );
  }, [tags]);

  # 返回卡片组件
  return (
    <div
      className={classNames(
        'group/card relative flex flex-col w-72 rounded justify-between text-black bg-white shadow-[0_8px_16px_-10px_rgba(100,100,100,.08)] hover:shadow-[0_14px_20px_-10px_rgba(100,100,100,.15)] dark:bg-[#232734] dark:text-white dark:hover:border-white transition-[transfrom_shadow] duration-300 hover:-translate-y-1 min-h-fit',
        {
          'grayscale cursor-no-drop': disabled,
          'cursor-pointer': !disabled && !!props.onClick,
        },
        className,
      )}
      {...props}
    >
      <div className="p-4">
        <div className="flex items-center">
          {iconNode}
          <div className="flex flex-col">
            <h2 className="text-sm font-semibold">{title}</h2>
            {tagNode}
          </div>
        </div>
        {desc && (
          <Tooltip title={desc}>
            <p className="mt-2 text-sm text-gray-500 font-normal line-clamp-2">{desc}</p>
          </Tooltip>
        )}
      </div>
      <div>
        {children}
        {operations && !!operations.length && (
          <div className="flex flex-wrap items-center justify-center border-t border-solid border-gray-100 dark:border-theme-dark">
            {operations.map((item, index) => (
              <Tooltip key={`operation-${index}`} title={item.label}>
                {/* 操作按钮容器 */}
                <div
                  className="relative flex flex-1 items-center justify-center h-11 text-gray-400 hover:text-blue-500 transition-colors duration-300 cursor-pointer"
                  onClick={(e) => {
                    e.stopPropagation();  // 阻止事件冒泡，以防止触发父元素的事件处理
                    item.onClick?.();  // 执行操作按钮的点击事件，如果定义了的话
                  }}
                >
                  {item.children}  {/* 显示操作按钮的内容 */}
                  {/* 如果不是最后一个操作按钮，则显示一个竖线 */}
                  {index < operations.length - 1 && <div className="w-[1px] h-6 absolute top-2 right-0 bg-gray-100 rounded dark:bg-theme-dark" />}
                </div>
              </Tooltip>
            ))}
          </div>
        )}
      </div>
    </div>
}

export default memo(GPTCard);
```