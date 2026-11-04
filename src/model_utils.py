import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score

def split_dataset(df: pd.DataFrame,
                  test_size: float = 0.2,
                  random_state: int = 42):
    """
    拆分特征与标签，分层抽样划分训练集、测试集
    :param df: 包含Churn标签的数据集
    :param test_size: 测试集比例
    :param random_state: 随机种子
    :return: X_train, X_test, y_train, y_test
    """
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
        shuffle=True
    )
    return X_train, X_test, y_train, y_test

def get_metrics(y_true, y_pred, y_prob):
    """
    模型评估指标：AUC、Recall、Precision、F1
    :param y_true: 真实标签
    :param y_pred: 预测类别（0/1）
    :param y_prob: 正类预测概率
    :return: dict 指标字典，保留4位小数
    """
    auc = roc_auc_score(y_true, y_prob)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return {
        "AUC": round(auc, 4),
        "Recall": round(recall, 4),
        "Precision": round(precision, 4),
        "F1": round(f1, 4)
    }