import pandas as pd
import numpy as np

def load_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    处理 TotalCharges 字段, 删除ID, 标签映射
    """
    df = df.copy()

    # TotalCharges空值填充0,转为浮点数类型
    df["TotalCharges"] = df["TotalCharges"].replace(" ","0")
    df["TotalCharges"] = df["TotalCharges"].astype(float)

    # 删除无关ID
    df = df.drop(columns=["customerID"])

    # 标签映射 Yes/No → 1/0
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


def binary_col_encode(df: pd.DataFrame, cols):
    """
    二分类特征：映射为0/1
    """
    df = df.copy()
    map_dict = {
        "Male": 1, "Female": 0,
        "Yes": 1, "No": 0
    }
    for col in cols:
        df[col] = df[col].map(map_dict)
    return df

def preprocess_encode(df: pd.DataFrame, drop_first: bool, multi_cat_cols: list[str] | None = None) -> pd.DataFrame:
    """
    统一编码函数：支持预测建模 / 因果推断两种模式
    :param df: 原始输入DataFrame
    :param drop_first: True=预测分支（消除虚拟变量陷阱）；False=因果分支（保留全部哑变量）
    :param multi_cat_cols: 需要额外追加的无序多分类特征列表；不传则只使用项目默认列表
    :return: 编码后DataFrame
    """
    df = df.copy()

    # 默认多分类特征列表
    default_multi_cat = [
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaymentMethod"
    ]

    if multi_cat_cols is not None:
        # 默认列表 + 追加传入的列，同时去重（防止重复传入）
        cols = list(dict.fromkeys(default_multi_cat + multi_cat_cols))
    else:
        cols = default_multi_cat.copy()

    df = pd.get_dummies(
        df,
        columns=cols,
        drop_first=drop_first,
        dtype=int
    )
    return df

def build_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    构造3个业务衍生特征：avg_monthly_spend、user_life_stage、charge_ratio
    """
    df = df.copy()

    # 1. 平均月消费
    df["avg_monthly_spend"] = np.where(
        df["tenure"] == 0,
        df["MonthlyCharges"],
        df["TotalCharges"] / df["tenure"]
    )

    # 2. 用户生命周期三分类
    max_tenure = df["tenure"].max() 
    bins = [-1, 6, 24, max_tenure]
    df["user_life_stage"] = pd.cut(
        df["tenure"],
        bins=bins,
        labels=["new_user", "mid_user", "long_user"],
        right=True
    )

    # 3. 单月消费占比charge_ratio
    df["charge_ratio"] = df["MonthlyCharges"] / df["TotalCharges"].replace(0, np.nan)
    df["charge_ratio"] = df["charge_ratio"].fillna(1.0)

    return df