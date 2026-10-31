# churn-causal-analysis
Customer Churn Prediction with Causal Inference-用户流失预测与因果推断增强分析

> 基于电信用户数据集，在传统流失预测模型之上增加因果推断，评估运营干预策略效果

## 项目目标
1. 构建用户流失二分类预测模型，识别高流失风险客户；
2. 使用因果推断（PSM倾向性得分匹配），评估优惠活动对留存的ATE平均处理效应；
3. 输出可落地的精细化运营策略，区分 [容易挽留] 和 [干预无效] 用户；
## 数据集
公开 Kaggle 电信数据集 Telco Customer Churn
路径：data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv

- 样本总量：7043条用户数据；
- 特征数量：21个；
- 目标变量：`Churn`，标记用户是否流失（Yes=流失，No=留存）；
- 字段覆盖：用户基础信息、合约类型、宽带服务、增值业务、支付方式、月费、总费用等；

## 技术栈
Python,Pandas,Seaborn,Scikit-learn,XGBoost,Statsmodels,DoWhy
## 环境复现
```bash
# 创建并激活环境
conda create -n churn-env python=3.10
conda activate churn-env
# 安装依赖
pip install -r requirements.txt
```

## 1. 数据分布

- 据数据描述分布以及不同合约类型、宽带类型、付费方式下的流失率可视化可知，**流失用户占比 26.54%**，**按月合约**用户流失率显著高于一年 / 两年合约，**光纤宽带**用户流失率高于 DSL 和无宽带用户，**电子支票支付**用户流失明显高于自动扣款（银行卡 / 信用卡）；