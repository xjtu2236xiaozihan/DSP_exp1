#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
孤立字语音识别分类器
基于VAD预处理和特征提取的结果进行训练和预测
"""

import pandas as pd
import numpy as np
import os
import joblib
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class SpeechClassifier:
    """孤立字语音识别分类器"""

    def __init__(self, features_dir=None, window_type="hamming"):
        """
        初始化分类器

        Args:
            features_dir: 特征文件目录
            window_type: 窗口类型，用于自动查找对应的特征文件
        """
        self.current_dir = os.path.dirname(os.path.abspath(__file__))

        if features_dir is None:
            # 自动构建特征文件路径，与feature_extraction.py输出一致
            self.features_dir = os.path.join(
                self.current_dir,
                "dataset",
                f"features_{window_type}"
            )
        else:
            self.features_dir = features_dir

        self.window_type = window_type
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.feature_columns = None
        self.performance = {}

        print(f"分类器初始化完成 - 窗口类型: {window_type}")
        print(f"特征目录: {self.features_dir}")

    def find_features_file(self):
        """查找特征文件"""
        if not os.path.exists(self.features_dir):
            raise FileNotFoundError(f"特征目录不存在: {self.features_dir}")

        # 查找CSV文件（与feature_extraction.py输出格式一致）
        csv_files = [f for f in os.listdir(self.features_dir)
                     if f.endswith('.csv') and self.window_type in f]

        if not csv_files:
            # 如果没有找到带窗口名的文件，尝试查找任何CSV文件
            csv_files = [f for f in os.listdir(self.features_dir) if f.endswith('.csv')]

        if not csv_files:
            raise FileNotFoundError(f"在 {self.features_dir} 中未找到特征CSV文件")

        # 选择第一个找到的CSV文件
        features_file = os.path.join(self.features_dir, csv_files[0])
        print(f"找到特征文件: {features_file}")
        return features_file

    def load_and_prepare_data(self, test_size=0.2, random_state=42):
        """
        加载特征数据并准备训练集和测试集
        """
        # 查找并加载特征文件
        features_file = self.find_features_file()
        df = pd.read_csv(features_file)

        print(f"原始数据形状: {df.shape}")
        print(f"数据列: {df.columns.tolist()}")

        # 从文件名提取标签（与feature_extraction.py输出一致）
        def extract_label(filename):
            # 支持多种文件名格式
            if '_' in filename:
                # 格式: "audio_noise_5_1_vad_processed.wav" -> "5"
                parts = filename.split('_')
                # 返回第一个非特殊部分
                for part in parts:
                    if part not in ['audio','noise', 'vad', 'processed', 'wav']:
                        return part
            return filename.split('.')[0]  # 兜底

        df['label'] = df['file_name'].apply(extract_label)

        # 显示标签分布
        print("\n标签分布:")
        print(df['label'].value_counts())

        # 构建特征矩阵，仅采用时域特征（平均能量、能量标准差、平均幅度、幅度标准差、平均过零率、过零率标准差）
        self.feature_columns = ['energy_mean', 'energy_std', 'magnitude_mean', 'magnitude_std', 'zcr_mean', 'zcr_std']

        X = df[self.feature_columns].values
        y = df['label'].values

        print(f"\n特征矩阵形状: {X.shape}")
        print(f"标签向量形状: {y.shape}")
        print(f"使用的特征: {self.feature_columns}")

        # 编码标签
        y_encoded = self.label_encoder.fit_transform(y)
        print("标签编码映射:")
        for i, label in enumerate(self.label_encoder.classes_):
            print(f"  {label} -> {i}")

        # 分割数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded,
            test_size=test_size,
            random_state=random_state,
            stratify=y_encoded
        )

        # 数据标准化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print(f"\n数据分割完成:")
        print(f"训练集: {X_train_scaled.shape}, {y_train.shape}")
        print(f"测试集: {X_test_scaled.shape}, {y_test.shape}")

        return X_train_scaled, X_test_scaled, y_train, y_test, df

    def train_svm(self, X_train, y_train, kernel='rbf', C=1.0, gamma='scale'):
        """训练SVM分类器"""
        print(f"\n训练SVM分类器 (kernel={kernel}, C={C}, gamma={gamma})...")

        self.model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            random_state=42,
            probability=True
        )

        self.model.fit(X_train, y_train)
        print("SVM训练完成")
        return self.model

    def train_random_forest(self, X_train, y_train, n_estimators=100, max_depth=None):
        """训练随机森林分类器"""
        print(f"\n训练随机森林分类器 (n_estimators={n_estimators})...")

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )

        self.model.fit(X_train, y_train)
        print("随机森林训练完成")
        return self.model

    def train_knn(self, X_train, y_train, n_neighbors=5):
        """训练K近邻分类器"""
        print(f"\n训练KNN分类器 (n_neighbors={n_neighbors})...")

        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.model.fit(X_train, y_train)
        print("KNN训练完成")
        return self.model

    def train_dt(self, X_train, y_train, max_depth=5):
        """训练决策树分类器"""
        print(f"\n训练决策树分类器 (max_depth={max_depth})...")

        self.model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        self.model.fit(X_train, y_train)
        print("决策树训练完成")
        return self.model

    def train_nb(self, X_train, y_train):
        """训练朴素贝叶斯分类器"""
        print(f"\n训练朴素贝叶斯分类器...")

        self.model = GaussianNB()
        self.model.fit(X_train, y_train)
        print("朴素贝叶斯训练完成")
        return self.model

    def train_lda(self, X_train, y_train, n_neighbors=5):
        """训练K近邻分类器"""
        print(f"\n训练LDA分类器...")

        self.model = LinearDiscriminantAnalysis()
        self.model.fit(X_train, y_train)
        print("LDA训练完成")
        return self.model

    def evaluate_model(self, X_test, y_test):
        """评估模型性能"""
        if self.model is None:
            raise ValueError("模型未训练，请先调用训练方法")

        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test) if hasattr(self.model, 'predict_proba') else None

        accuracy = accuracy_score(y_test, y_pred)

        print(f"\n=== 模型评估结果 ===")
        print(f"准确率: {accuracy:.4f}")

        # 详细分类报告
        print("\n详细分类报告:")
        print(classification_report(y_test, y_pred,
                                    target_names=self.label_encoder.classes_))

        # 保存性能指标
        self.performance = {
            'accuracy': accuracy,
            'predictions': y_pred,
            'true_labels': y_test
        }

        return y_pred, y_pred_proba, accuracy

    def plot_confusion_matrix(self, y_test, y_pred, title="混淆矩阵"):
        """绘制混淆矩阵"""
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_)
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title(f'{title} - {self.window_type}窗口 (准确率: {self.performance["accuracy"]:.4f})')
        plt.tight_layout()

        # 保存图片
        output_dir = os.path.join(self.current_dir, "results")
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'confusion_matrix_{self.window_type}.png'),
                    dpi=300, bbox_inches='tight')
        plt.show()

    def compare_classifiers(self, X_train, X_test, y_train, y_test):
        """比较多种分类器性能"""
        classifiers = {
            'SVM(线性)': SVC(kernel='linear', random_state=42, probability=True),
            'SVM(RBF)': SVC(kernel='rbf', random_state=42, probability=True),
            '随机森林': RandomForestClassifier(n_estimators=100, random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            '决策树': DecisionTreeClassifier(max_depth=5, random_state=42),
            '朴素贝叶斯': GaussianNB(),
            'LDA': LinearDiscriminantAnalysis()
        }

        results = {}

        print("=== 分类器性能比较 ===")
        for name, clf in classifiers.items():
            print(f"\n训练 {name}...")

            # 训练模型
            clf.fit(X_train, y_train)

            # 预测
            y_pred = clf.predict(X_test)

            # 评估
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = {
                'classifier': clf,
                'accuracy': accuracy,
                'predictions': y_pred
            }

            print(f"{name} 准确率: {accuracy:.4f}")

        # 绘制比较图
        self._plot_classifier_comparison(results)

        return results

    def _plot_classifier_comparison(self, results):
        """绘制分类器比较图"""
        names = list(results.keys())
        accuracies = [results[name]['accuracy'] for name in names]

        plt.figure(figsize=(12, 6))
        bars = plt.bar(names, accuracies, color='skyblue')
        plt.xticks(rotation=45)
        plt.ylabel('准确率')
        plt.title(f'分类器性能比较 - {self.window_type}窗口')
        plt.ylim(0, 1.0)

        # 在柱状图上显示数值
        for bar, accuracy in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{accuracy:.4f}', ha='center', va='bottom')

        plt.tight_layout()

        # 保存图片
        output_dir = os.path.join(self.current_dir, "results")
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'classifier_comparison_{self.window_type}.png'),
                    dpi=300, bbox_inches='tight')
        plt.show()

    def hyperparameter_tuning(self, X_train, y_train, classifier_type='svm'):
        """超参数调优"""
        print(f"\n=== {classifier_type.upper()} 超参数调优 ===")

        if classifier_type == 'svm':
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': [1, 0.1, 0.01, 0.001],
                'kernel': ['rbf', 'linear']
            }
            base_model = SVC(random_state=42, probability=True)
        elif classifier_type == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
            base_model = RandomForestClassifier(random_state=42)
        else:
            raise ValueError("不支持的分类器类型")

        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)

        print(f"最佳参数: {grid_search.best_params_}")
        print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")

        self.model = grid_search.best_estimator_
        return grid_search.best_estimator_, grid_search.best_params_

    def save_model(self, model_path=None):
        """保存训练好的模型"""
        if self.model is None:
            raise ValueError("没有训练好的模型可以保存")

        if model_path is None:
            model_dir = os.path.join(self.current_dir, "models")
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f'speech_classifier_{self.window_type}.pkl')

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns,
            'window_type': self.window_type,
            'performance': self.performance
        }

        joblib.dump(model_data, model_path)

        # 保存模型信息
        model_info = {
            'window_type': self.window_type,
            'feature_columns': self.feature_columns,
            'classes': self.label_encoder.classes_.tolist(),
            'accuracy': float(self.performance.get('accuracy', 0)),
            'model_type': type(self.model).__name__
        }

        info_path = model_path.replace('.pkl', '_info.json')
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)

        print(f"模型已保存到: {model_path}")
        print(f"模型信息已保存到: {info_path}")

    def load_model(self, model_path):
        """加载训练好的模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        model_data = joblib.load(model_path)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_columns = model_data['feature_columns']
        self.window_type = model_data.get('window_type', 'unknown')
        self.performance = model_data.get('performance', {})

        print(f"模型加载成功 - 窗口类型: {self.window_type}")
        print(f"准确率: {self.performance.get('accuracy', '未知'):.4f}")

    def predict(self, features):
        """预测新样本"""
        if self.model is None:
            raise ValueError("请先加载或训练模型")

        # 标准化特征
        features_scaled = self.scaler.transform([features])

        # 预测
        prediction_encoded = self.model.predict(features_scaled)[0]
        prediction_label = self.label_encoder.inverse_transform([prediction_encoded])[0]

        # 获取概率（如果支持）
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features_scaled)[0]
            prob_dict = {}
            for i, prob in enumerate(probabilities):
                class_name = self.label_encoder.inverse_transform([i])[0]
                prob_dict[class_name] = float(prob)
        else:
            prob_dict = None

        return prediction_label, prob_dict

    def analyze_feature_importance(self, feature_names=None):
        """分析特征重要性（仅适用于树模型和线性模型）"""
        if self.model is None:
            raise ValueError("模型未训练")

        if feature_names is None:
            feature_names = self.feature_columns

        # 随机森林的特征重要性
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]

            plt.figure(figsize=(10, 8))
            plt.title("特征重要性")
            plt.barh(range(len(indices)), importances[indices], color='lightcoral')
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.xlabel('相对重要性')
            plt.gca().invert_yaxis()
            plt.tight_layout()

            # 保存图片
            output_dir = os.path.join(self.current_dir, "results")
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f'feature_importance_{self.window_type}.png'),
                        dpi=300, bbox_inches='tight')
            plt.show()

            print("特征重要性排序:")
            for i in range(min(10, len(indices))):
                print(f"{i + 1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

            return importances
        else:
            print("该模型不支持特征重要性分析")
            return None


def main():
    """主函数：完整的分类流程"""
    import argparse

    parser = argparse.ArgumentParser(description="孤立字语音识别分类器")
    parser.add_argument("--window", type=str, default="hamming",
                        choices=["rect", "hamming", "hanning"],
                        help="窗口类型（与特征提取一致）")
    parser.add_argument("--classifier", type=str, default="svm",
                        choices=["svm", "random_forest", "knn","dt","nb","lda"],
                        help="分类器类型")
    parser.add_argument("--tune", action="store_true",
                        help="是否进行超参数调优")
    parser.add_argument("--compare", action="store_true",
                        help="是否比较所有分类器")

    args = parser.parse_args()

    print("=== 孤立字语音识别分类器 ===")
    print(f"窗口类型: {args.window}")
    print(f"分类器: {args.classifier}")

    # 初始化分类器
    classifier = SpeechClassifier(window_type=args.window)

    try:
        # 加载和准备数据
        X_train, X_test, y_train, y_test, df = classifier.load_and_prepare_data()

        # 训练分类器
        if args.classifier == "svm":
            if args.tune:
                classifier.hyperparameter_tuning(X_train, y_train, 'svm')
            else:
                classifier.train_svm(X_train, y_train)
        elif args.classifier == "random_forest":
            if args.tune:
                classifier.hyperparameter_tuning(X_train, y_train, 'random_forest')
            else:
                classifier.train_random_forest(X_train, y_train)
        elif args.classifier == "knn":
            classifier.train_knn(X_train, y_train)
        elif args.classifier == "dt":
            classifier.train_dt(X_train, y_train)
        elif args.classifier == "nb":
            classifier.train_nb(X_train, y_train)
        elif args.classifier == "lda":
            classifier.train_lda(X_train, y_train)

        # 评估模型
        y_pred, y_pred_proba, accuracy = classifier.evaluate_model(X_test, y_test)

        # 绘制混淆矩阵
        classifier.plot_confusion_matrix(y_test, y_pred)

        # 比较所有分类器（可选）
        if args.compare:
            classifier.compare_classifiers(X_train, X_test, y_train, y_test)

        # 分析特征重要性
        classifier.analyze_feature_importance()

        # 保存模型
        classifier.save_model()

        print("\n=== 流程完成 ===")
        print(f"最终准确率: {accuracy:.4f}")

    except Exception as e:
        print(f"处理过程中出错: {e}")
        raise


if __name__ == "__main__":
    main()