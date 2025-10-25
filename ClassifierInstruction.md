## 分类器使用说明

使用前需提前生成特征csv文件，文件路径与feature_extraction.py输出路径一致。

### 使用示例
```bash
# 1. 窗口类型选择（默认使用海明窗）
# 矩形窗
python classifier.py --window rect --classifier svm
# 汉宁窗
python classifier.py --window hanning --classifier svm
# 海明窗
python classifier.py --window hamming --classifier svm

# 2. 分类器类型选择（默认使用SVM）
# svm
python classifier.py --window hamming --classifier svm
# 随机森林
python classifier.py --window hamming --classifier random_forest
# K近邻
python classifier.py --window hamming --classifier knn
# 决策树
python classifier.py --window hamming --classifier dt
# 朴素贝叶斯
python classifier.py --window hamming --classifier nb
# 线性判别
python classifier.py --window hamming --classifier lda

# 3. 进行超参数调优
python classifier.py --window hamming --classifier svm --tune

# 4. 比较所有分类器
python classifier.py --window hamming --compare