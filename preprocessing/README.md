# Preprocessing 文件夹说明

这个文件夹用于存放YOLO训练数据生成过程中的中间文件。

## 📁 文件夹结构

```
preprocessing/
├── raw_images/          # 原始截图文件
├── annotations/         # JSON标注文件  
├── visualizations/      # 可视化检查文件
└── README.md           # 本说明文件
```

## 🔄 工作流程

1. **截图处理** (选项2)
   - 处理screenshots文件夹中的截图
   - 生成检测结果并保存到各个子文件夹

2. **手动筛选**
   - 查看 `visualizations/` 文件夹中的可视化结果
   - 删除标注错误的可视化文件

3. **转换训练数据** (选项3)
   - 根据剩余的可视化文件转换为YOLO格式
   - 转换后删除对应的原始文件

## 📋 文件命名规则

- **原始图像**: `capture_YYYYMMDD_HHMMSS_mmm_NNNN.jpg`
- **标注文件**: `ann_capture_YYYYMMDD_HHMMSS_mmm_NNNN.json`
- **可视化文件**: `vis_capture_YYYYMMDD_HHMMSS_mmm_NNNN.jpg`

## ⚠️ 注意事项

- 手动筛选时只需要删除 `visualizations/` 中的错误文件
- 程序会自动找到对应的原始文件和标注文件
- 转换完成后会自动清理已转换的文件，但保留文件夹结构
