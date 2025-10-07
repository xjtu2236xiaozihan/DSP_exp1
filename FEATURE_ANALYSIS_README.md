# 语音特征分析系统

## 🎯 功能概述

这是一个完整的语音特征分析系统，用于提取和分析语音信号的时域特征，进行特征归一化和可视化分析。

## 📋 主要功能

### 1. 特征提取 (`feature_extraction.py`)
- **基础时域特征**: 短时能量、短时平均幅度、短时过零率
- **高级时域特征**: 均方根、峰值因子
- **频谱特征**: 频谱质心、频谱带宽
- **加窗处理**: 使用汉明窗减少频谱泄漏，提高分析精度

### 2. 特征分析可视化 (`feature_analysis.py`)
- **特征分布图**: 特征值的直方图分布
- **相关性矩阵**: 特征间的相关性热图
- **特征分析报告**: 详细的统计分析和特征重要性排序

## 🚀 快速开始

### 分步运行
```bash
# 1. 特征提取
python feature_extraction.py

# 2. 特征分析可视化
python feature_analysis.py
```

## 📁 目录结构

```
DSP_exp1/
├── feature_extraction.py          # 特征提取
├── feature_analysis.py           # 特征分析可视化
├── FEATURE_ANALYSIS_README.md     # 说明文档
└── dataset/
    ├── audio_processed/          # 处理后的音频文件
    └── features/                # 生成的特征文件
        └── audio_features.csv   # 特征CSV文件
```

## 📊 输出文件说明

### CSV特征文件
- **audio_features.csv**: 核心特征数据
  - 包含基础时域特征、高级时域特征、频谱特征
  - 每行代表一个音频文件的统计特征
  - 包含均值、标准差等统计信息

### 可视化图表
- **feature_distribution.png**: 特征分布直方图
- **feature_correlation.png**: 特征相关性热图

## 🔧 配置参数

### 特征提取参数
```python
FRAME_LENGTH = 0.025    # 帧长 25ms
FRAME_SHIFT = 0.010     # 帧移 10ms
SAMPLE_RATE = 16000     # 采样率 16kHz
```

### 特征组合权重
```python
weights = {
    'energy': 0.4,      # 短时能量权重
    'magnitude': 0.4,   # 短时平均幅度权重
    'zcr': 0.1,         # 过零率权重（降低）
    'autocorr': 0.05,   # 自相关权重
    'amp_diff': 0.05    # 幅度差分权重
}
```

## 📈 特征分析结果

### 噪声影响分析
系统会自动分析噪声对各个特征的影响：

1. **特征均值变化**: 量化噪声对特征中心值的影响
2. **特征方差变化**: 分析噪声对特征分布的影响
3. **特征重要性排序**: 基于噪声敏感性排序特征重要性

### 特征归一化
- **Z-score标准化**: 将特征标准化为均值0、方差1
- **特征组合**: 根据权重组合多个特征
- **降维处理**: 减少过零率等噪声敏感特征的权重

## 🛠️ 依赖库

```bash
pip install numpy pandas matplotlib seaborn scipy librosa pydub
```

### 库说明
- **numpy**: 数值计算
- **pandas**: 数据处理
- **matplotlib/seaborn**: 数据可视化
- **scipy**: 科学计算
- **librosa**: 音频处理
- **pydub**: 音频格式转换

## 📋 使用示例

### 1. 基础特征提取
```python
from feature_extraction import FeatureExtractor

extractor = FeatureExtractor()
features = extractor.extract_features("audio.wav")
```

### 2. 特征归一化
```python
normalized_features = extractor.normalize_features(features)
combined_feature = extractor.combine_features(features)
```

### 3. 噪声影响分析
```python
from feature_analysis import FeatureAnalyzer

analyzer = FeatureAnalyzer("audio_features.csv")
analyzer.plot_feature_comparison()
analyzer.generate_summary_report()
```

## 🔍 特征说明

### 时域特征
- **短时能量**: 反映语音信号的强度变化
- **短时平均幅度**: 衡量信号的平均幅度
- **短时过零率**: 指示语音的清浊音特性
- **短时自相关**: 反映信号的周期性
- **幅度差分**: 衡量信号的动态变化

### 频域特征
- **MFCC**: 模拟人耳听觉特性的特征
- **频谱质心**: 频谱的重心位置
- **频谱带宽**: 频谱的分散程度
- **频谱滚降点**: 85%能量对应的频率点
- **频谱平坦度**: 频谱的平坦程度

### 高级特征
- **偏度**: 信号分布的对称性
- **峰度**: 信号分布的尖锐程度
- **峰值因子**: 峰值与有效值的比值
- **波形因子**: 有效值与平均值的比值
- **脉冲因子**: 峰值与平均值的比值

## ⚠️ 注意事项

1. **音频格式**: 支持WAV、MP3、M4A、FLAC、AAC等格式
2. **采样率**: 建议使用16kHz采样率
3. **文件大小**: 大文件处理时间较长，建议分批处理
4. **内存使用**: 高级特征提取需要较多内存
5. **依赖库**: 确保所有依赖库正确安装

## 🐛 常见问题

### Q: 运行出错 "No module named 'librosa'"
A: 安装librosa库：`pip install librosa`

### Q: 音频文件读取失败
A: 检查音频文件格式，确保文件未损坏

### Q: 特征提取速度慢
A: 可以调整帧长和帧移参数，或减少处理的文件数量

### Q: 内存不足
A: 分批处理音频文件，或减少同时处理的文件数量

## 📞 技术支持

如有问题，请检查：
1. 依赖库是否正确安装
2. 音频文件是否存在且格式正确
3. 输出目录是否有写入权限
4. Python版本是否兼容（建议Python 3.7+）
