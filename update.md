# 在wch的代码基础上进行了一点改进，使能分别使用三种窗函数进行处理

## 主要功能和wch的提交相同，不再做概述

## 使用示例
```bash
# 1. 特征提取

# 矩形窗
bash feat_ext.sh rect
# 汉宁窗
bash feat_ext.sh hanning
# 海明窗
bash feat_ext.sh hamming

# 2. 特征分析可视化

# 矩形窗
bash feat_anal.sh rect
# 汉宁窗
bash feat_anal.sh hanning
# 海明窗
bash feat_anal.sh hamming
# !!! 注意需要将脚本中的最后一个参数 --plots_root "/home/wdai/dw/DSP_exp1/plots"路径换成自己的路径
```