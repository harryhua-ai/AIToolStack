import React from 'react';
import './ShortcutHelper.css';

interface ShortcutItem {
  key: string;
  description: string;
}

const shortcuts: ShortcutItem[] = [
  { key: '1-9', description: '快速切换类别' },
  { key: 'R', description: '矩形框工具' },
  { key: 'P', description: '多边形工具' },
  { key: 'V', description: '选择/移动工具' },
  { key: 'K', description: '关键点工具' },
  { key: 'A / ←', description: '上一张图像' },
  { key: 'D / →', description: '下一张图像' },
  { key: 'Space + 拖拽', description: '平移画布' },
  { key: 'H', description: '隐藏/显示标注' },
  { key: 'Del / Backspace', description: '删除选中标注' },
  { key: 'Ctrl+Z', description: '撤销' },
  { key: 'Ctrl+Shift+Z', description: '重做' },
  { key: 'Ctrl+S', description: '手动保存' },
  { key: 'Esc', description: '取消当前操作' },
  { key: 'Enter', description: '完成多边形绘制' },
];

export const ShortcutHelper: React.FC = () => {
  return (
    <div className="shortcut-helper">
      <div className="shortcut-helper-content">
        {shortcuts.map((item, index) => (
          <div key={index} className="shortcut-item">
            <span className="shortcut-key">{item.key}</span>
            <span className="shortcut-description">{item.description}</span>
          </div>
        ))}
      </div>
    </div>
  );
};
