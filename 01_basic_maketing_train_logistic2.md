
LASSO回帰
回帰係数が0に近いものがよりゼロに近いようにしてくれる

# 統計演習

https://archive.ics.uci.edu/ml/datasets/bank+marketing  
のデータを利用

## サマリ

### ターゲットのユーザー像(ペルソナ)

契約成否でデータを比較（Excelで）  
以下の像が浮かび上がった

* 若者か高齢者（学生か退職者）で
* 前回電話したときに携帯でかけていて
* それも3月、4月、10月のいずれかだった人

### ROIを最大化させるための予測モデル

* 基本的なアプローチ
 1. 以下のCategoryを掛け合わせた新しいCategoryを作成
    * age
    * job
    * contact
    * month
 2. logistic回帰でモデルを作成
 3. モデル作成時に利用したデータを利用して予測結果を取得
 4. 予測結果からモデルを評価（AUC）
 5. 良いモデルだったら、ROIが最大となるようなしきい値を算出（電話1回500円、契約成功LTVは2000円）
 6. テストデータを上記のモデルで評価する

* ほかは以下を使っている
    * const
    * campaign
    * pdays
    * previous
    * emp.var.rate
    * nr.employed
    * job
    * marital
    * education
    * housing
    * loan
    * contact
    * poutcome
    * range_age (10歳おきのCategory)


* テストデータの予測結果を算出し、訓練データで算出したしきい値0.27を適用すると
  -   1,488,500 円の利益がでる
* その時のConfusionMatrix

|        |          | Predict  |          |
|--------|----------|----------|----------|
|        |          | Positive | Negative |
| Actual | Positive | 1544      | 349     |
|        | Negative | 1655      | 572    |

* 適合率: 48.3%
  -  ランダムに電話した場合は46%の確率で契約取れるはずなので、少しは上がったと考えられる
  - 実はTestModel2の方が50.5%で4つを組み合わせた場合より適合率が高いので、アタックリストとしてはTestModel2の方が良い
  - 更にTestModel5のほうが53.4%と試した中では一番高かった
    - 単純比較でも傾向が出ていることがわかるパラメータを組み合わせるよりは、傾向が出ていないパラメータを組み合わせた方が、新たな傾向を見つけるチャンスになり良い気がする

# データ補完パート

## ライブラリの読み込み


```python
import pandas as pd
import pandas.io.sql as psql
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pymysql
import random
import copy
import statsmodels.api as sm
random.seed(0)
import pylab as pl
from sklearn.metrics import roc_curve, auc as calc_auc
from sklearn.metrics import confusion_matrix

%matplotlib inline
```

    /Users/isapro/.pyenv/versions/anaconda3-5.1.0/lib/python3.6/site-packages/statsmodels/compat/pandas.py:56: FutureWarning: The pandas.core.datetools module is deprecated and will be removed in a future version. Please use the pandas.tseries module instead.
      from pandas.core import datetools


## 関数定義

### 欠損値カウント処理


```python
# 指定されたDataFrameのうちObjectタイプになっているカラムについて、欠損値名（指定しなければunknown）でチェックする
def count_nan_value_at_object(data, nan_value = "unknown"):
    for obj in data.select_dtypes(include=object).columns:
        num = data[data[obj] == nan_value][obj].count()
        print('column %s\tnum %d' % (obj, num))
```

### 欠損値補完処理


```python
# 指定されたDataFrameのうち、対象として選択したカラム名の欠損値（指定しなければunknown)を他の値の出現割合に応じて保管する
# 破壊的メソッド
def comp_missing_category_value(data, target, nan_value = "unknown"):
    if data.job.dtype != "object":
        return

    # 値が入っているところだけを取り出す（出現頻度に応じた補完をするため）
    not_unknown_df = data[data[target] != nan_value][target]
    not_unknown_df.reset_index(drop=True,inplace=True)
    
    # 値が入っている割合に従い、欠損値を補正する
    data.loc[data[target] == nan_value, target] = pd.Categorical(not_unknown_df[random.randint(0,not_unknown_df.count())])

```

### 不要カラム削除処理


```python
def drop_columns_as_start_with(data, drop_list):
    X = data.copy()
    for c in drop_list:
        for t in [s for s in X if s.startswith(c)]:
            X.drop(t, axis=1, inplace=True)
    return X
```

### CPRProt作成処理


```python
def CPRProtFigure(X, result):
    total = len(X.columns)
    ncol = 3
    nrow = int(total / ncol) + 1
    fig = plt.figure(figsize=(20, nrow*4))
    plt.subplots_adjust(wspace=0.3, hspace=0.5)
    for i in range(0,total):
        ax = fig.add_subplot(nrow,ncol,i+1)
        result.plot_partial_residuals(X.columns[i], ax)

    return plt
```

### ROCプロット作成処理


```python
def ROCPlot(Y, prY):
    # 偽陽性、真陽性、しきい値の組み合わせを取得する
    fpr, tpr, thresholds = roc_curve(Y, prY)
    # AUCを計算する
    roc_auc = calc_auc(fpr, tpr)
   
    # ROC曲線用の処理
    i = np.arange(len(tpr)) # index for df
    roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
    roc.loc[(roc.tf-0).abs().argsort()[:1]]

    # 描画
    fig = plt.figure(figsize=(6, 6))
    plt.plot(roc['fpr'],roc['tpr'])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.ylim([0,1])
    plt.xlim([0,1])
    plt.plot([[0,0],[1,1]], c='r')
    plt.title('ROC')

    return plt, roc_auc
```


```python
def TestModel1(data):
     # 新しい変数の作成
    new_category = ["job","range_age","education"]
    for job in data["job"].unique():
        for range_age in data["range_age"].unique():
            for edu in data["education"].unique():
                data.loc[(data["job"] == job) & (data["range_age"] == range_age) & (data["education"] == edu), "personal"] = f'{job}+{range_age}+{edu}'
    for cat in new_category:
        data.drop(cat, axis=1, inplace=True)
    return new_category

```


```python
def TestModel2(data):
     # 新しい変数の作成
    new_category = ["contact","education"]
    for k1 in data["contact"].unique():
        for k2 in data["education"].unique():
                data.loc[(data["contact"] == k1) & (data["education"] == k2), "personal"] = f'{k1}+{k2}'
    for cat in new_category:
        data.drop(cat, axis=1, inplace=True)                
    return new_category
```


```python
def TestModel3(data):
     # 新しい変数の作成
    new_category = ["contact","month","range_age","job"]
    for k1 in data["contact"].unique():
        for k2 in data["month"].unique():
            for k3 in data["range_age"].unique():
                for k4 in data["job"].unique():
                    data.loc[\
                        (data["contact"] == k1) &\
                        (data["month"] == k2) &\
                        (data["range_age"] == k3) &\
                        (data["job"] == k4), "personal"] = f'{k1}+{k2}+{k3}+{k4}'
    for cat in new_category:
        data.drop(cat, axis=1, inplace=True)                
    return new_category
```


```python
def TestModel4(data):
     # 新しい変数の作成
    new_category = ["range_age","job"]
    for k1 in data["range_age"].unique():
        for k2 in data["job"].unique():
                data.loc[\
                    (data["range_age"] == k1) &\
                    (data["job"] == k2), "personal"] = f'{k1}+{k2}'
    for cat in new_category:
        data.drop(cat, axis=1, inplace=True)                
    return new_category
```


```python
def TestModel5(data):
     # 新しい変数の作成
    new_category = ["marital","education","day_of_week"]
    for k1 in data["marital"].unique():
        for k2 in data["education"].unique():
            for k3 in data["day_of_week"].unique():
                data.loc[\
                    (data["marital"] == k1) &\
                    (data["education"] == k2) &\
                    (data["day_of_week"] == k3), "personal"] = f'{k1}+{k2}+{k3}'
    for cat in new_category:
        data.drop(cat, axis=1, inplace=True)                
    return new_category
```

## MAIN関数


```python
def BANKMarketingDataCrensing(data):
    # 変数定義
    comp_list = ["job","marital","education","default","housing","loan"]
    dummy_list = ["job","marital","education","default","housing","loan","contact","month","day_of_week","poutcome","range_age","personal"]
#     dummy_list = ["job","marital","education","default","housing","loan","contact","month","day_of_week","poutcome","range_age"]

    # データの補完
    for target in comp_list:
        comp_missing_category_value(data, target, nan_value = "unknown")
    
    # ageのビンズ化(10歳ずつのカテゴリ)
    age_labels = [ "{0}_{1}".format(i, i + 9) for i in range(0, 100, 10) ]
    data.loc[:, "range_age"] = pd.cut(data.loc[:,'age'], np.arange(0, 101, 10), labels = age_labels)
    
    # 新しい変数の作成
#     used_array =  TestModel1(data)
#     used_array = TestModel2(data)
#     used_array = TestModel3(data)
#     used_array = TestModel4(data)
    used_array = TestModel5(data)
    for v in used_array:
        dummy_list.remove(v)
    
    # ダミー化
    analyze_data = data.copy()
    analyze_data = pd.get_dummies(analyze_data, columns=dummy_list, drop_first=True)
    
    # 切片の追加
    analyze_data = sm.add_constant(analyze_data)
    
    return analyze_data

def BANKMarketingCreateTrainData(data):
    # 変数定義
    drop_list = ['contract', 'duration','age','month','default','euribor3m','cons.','day_of','y']

    # dataの整形
    analyze_data = BANKMarketingDataCrensing(data)

    # 目的変数の作成
    analyze_data['contract'] = data['y'].apply(lambda x : 1 if x == "yes" else 0)
    
    # 目的変数と説明変数を抜き出して作成。説明変数からは不要なカラムを削除
    analyze_data_Y = analyze_data['contract']
    analyze_data_X = drop_columns_as_start_with(analyze_data, drop_list)

    # 分析用データの返却
    return analyze_data_X, analyze_data_Y

def BANKMarketingCreateTestData(data):
    # 変数定義
    drop_list = ['contract', 'duration','age','month','default','euribor3m','cons.','day_of','y']

    # dataの整形
    analyze_data = BANKMarketingDataCrensing(data)
    
    # 説明変数からは不要なカラムを削除
    analyze_data_X = drop_columns_as_start_with(analyze_data, drop_list)

    # 分析用データの返却
    return analyze_data_X

def BANKMarketingGLM(X, Y):
    # GLMの実行
    #モデルの構築
    model = sm.GLM(Y.astype(float), X, family=sm.families.Binomial())
    #fitに回帰した結果が入っているので、これをresに代入する
    res = model.fit()

    return res, X, Y

def BANKMarketingPredictProfit(Y, prY):
    # 変数定義
    thresholds = np.arange(0.01, 1.01, 0.01)
    profits = []
    max_t = 0
    max_p = 0
    max_tn = 0
    max_fp = 0
    max_fn = 0
    max_tp = 0

    # 最大値を探す
    for t in thresholds:
        # しきい値に対して0,1に変換する
        prY2 = prY.map(lambda x: 1 if x > t else 0)
        
        # 混同行列を作成
        tn, fp, fn, tp = confusion_matrix(Y, prY2).ravel()
        
        # 評価結果と最大値の更新
        y = 2000 * tp - 500 * (tp + fp)
        profits.append(y)
        if(max_t==0 or max_y < y):
            max_t = t
            max_y = y
            max_tn = tn
            max_fp = fp
            max_fn = fn
            max_tp = tp

    # 評価結果の描画
    fig, ax = plt.subplots()
    plt.plot(thresholds, profits)
    plt.scatter(max_t, max_y, marker='o', color='r', s=50)
    plt.xlabel('thresholds')
    plt.ylabel('profits')
    plt.title('Profit and loss graph')

    return plt, max_y, max_t, max_tp, max_fn, max_fp, max_tn

```

### 一発で予測結果を取得する関数


```python
def ippatu(train, test, t):
    # 整形
    trainX, trainY = BANKMarketingCreateTrainData(train)
    testX, testY = BANKMarketingCreateTrainData(test)

    # trainデータにあって、testデータに無いカラムは0で埋める
    for c in trainX.columns:
        if(c not in testX.columns):
            testX[c] = 0

    # testデータにあって、trainデータに無いカラムは捨て去る
    for c in testX.columns:
        if(c not in trainX.columns):
            testX.drop(c, axis=1, inplace=True)

    # 予測モデル
    res, trainX, trainY = BANKMarketingGLM(trainX,trainY)
    
    # 予測結果の取得
    test_predY = res.predict(testX)

    # しきい値を使って0,1に直す
    test_predY2 = test_predY.map(lambda x: 1 if x > t else 0)

    # 混同行列を作成
    tn, fp, fn, tp = confusion_matrix(testY, test_predY2).ravel()

    # 利益を算出
    y = 2000 * tp - 500 * (tp + fp)

    # 最適化された予測結果の格納
    temp = test.copy()
    temp['predict'] = test_predY2
    
    return temp, y, tp, fn, fp, tn
```

## Main処理


```python
# データの読み込み
data = pd.read_csv('bank_marketing_train.csv')
# データの整形
X, Y = BANKMarketingCreateTrainData(data)
# 予測結果の取得
res, X, Y = BANKMarketingGLM(X,Y)
# 予測結果の表示
res.summary()
```




<table class="simpletable">
<caption>Generalized Linear Model Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>      <td>contract</td>     <th>  No. Observations:  </th>  <td> 37068</td> 
</tr>
<tr>
  <th>Model:</th>                 <td>GLM</td>       <th>  Df Residuals:      </th>  <td> 36945</td> 
</tr>
<tr>
  <th>Model Family:</th>       <td>Binomial</td>     <th>  Df Model:          </th>  <td>   122</td> 
</tr>
<tr>
  <th>Link Function:</th>        <td>logit</td>      <th>  Scale:             </th>    <td>1.0</td>  
</tr>
<tr>
  <th>Method:</th>               <td>IRLS</td>       <th>  Log-Likelihood:    </th> <td> -8957.2</td>
</tr>
<tr>
  <th>Date:</th>           <td>Fri, 12 Apr 2019</td> <th>  Deviance:          </th> <td>  17914.</td>
</tr>
<tr>
  <th>Time:</th>               <td>00:06:19</td>     <th>  Pearson chi2:      </th> <td>3.82e+04</td>
</tr>
<tr>
  <th>No. Iterations:</th>        <td>22</td>        <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
                      <td></td>                         <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>                                     <td>  -34.0692</td> <td> 7817.942</td> <td>   -0.004</td> <td> 0.997</td> <td>-1.54e+04</td> <td> 1.53e+04</td>
</tr>
<tr>
  <th>campaign</th>                                  <td>   -0.0429</td> <td>    0.010</td> <td>   -4.314</td> <td> 0.000</td> <td>   -0.062</td> <td>   -0.023</td>
</tr>
<tr>
  <th>pdays</th>                                     <td>   -0.0010</td> <td>    0.001</td> <td>   -1.598</td> <td> 0.110</td> <td>   -0.002</td> <td>    0.000</td>
</tr>
<tr>
  <th>previous</th>                                  <td>    0.1467</td> <td>    0.225</td> <td>    0.653</td> <td> 0.514</td> <td>   -0.294</td> <td>    0.587</td>
</tr>
<tr>
  <th>emp.var.rate</th>                              <td>   -0.6297</td> <td>    0.068</td> <td>   -9.211</td> <td> 0.000</td> <td>   -0.764</td> <td>   -0.496</td>
</tr>
<tr>
  <th>nr.employed</th>                               <td>    0.0065</td> <td>    0.002</td> <td>    3.813</td> <td> 0.000</td> <td>    0.003</td> <td>    0.010</td>
</tr>
<tr>
  <th>job_blue-collar</th>                           <td>   -0.1383</td> <td>    0.076</td> <td>   -1.808</td> <td> 0.071</td> <td>   -0.288</td> <td>    0.012</td>
</tr>
<tr>
  <th>job_entrepreneur</th>                          <td>   -0.0759</td> <td>    0.115</td> <td>   -0.661</td> <td> 0.509</td> <td>   -0.301</td> <td>    0.149</td>
</tr>
<tr>
  <th>job_housemaid</th>                             <td>   -0.1470</td> <td>    0.161</td> <td>   -0.914</td> <td> 0.361</td> <td>   -0.462</td> <td>    0.168</td>
</tr>
<tr>
  <th>job_management</th>                            <td>   -0.1376</td> <td>    0.088</td> <td>   -1.570</td> <td> 0.117</td> <td>   -0.309</td> <td>    0.034</td>
</tr>
<tr>
  <th>job_retired</th>                               <td>    0.2324</td> <td>    0.133</td> <td>    1.745</td> <td> 0.081</td> <td>   -0.029</td> <td>    0.493</td>
</tr>
<tr>
  <th>job_self-employed</th>                         <td>   -0.0437</td> <td>    0.114</td> <td>   -0.383</td> <td> 0.702</td> <td>   -0.267</td> <td>    0.180</td>
</tr>
<tr>
  <th>job_services</th>                              <td>   -0.1561</td> <td>    0.084</td> <td>   -1.860</td> <td> 0.063</td> <td>   -0.321</td> <td>    0.008</td>
</tr>
<tr>
  <th>job_student</th>                               <td>    0.3085</td> <td>    0.140</td> <td>    2.198</td> <td> 0.028</td> <td>    0.033</td> <td>    0.584</td>
</tr>
<tr>
  <th>job_technician</th>                            <td>   -0.0005</td> <td>    0.071</td> <td>   -0.007</td> <td> 0.994</td> <td>   -0.139</td> <td>    0.138</td>
</tr>
<tr>
  <th>job_unemployed</th>                            <td>    0.0128</td> <td>    0.141</td> <td>    0.091</td> <td> 0.928</td> <td>   -0.264</td> <td>    0.290</td>
</tr>
<tr>
  <th>housing_yes</th>                               <td>   -0.0506</td> <td>    0.041</td> <td>   -1.225</td> <td> 0.221</td> <td>   -0.132</td> <td>    0.030</td>
</tr>
<tr>
  <th>loan_yes</th>                                  <td>   -0.0346</td> <td>    0.054</td> <td>   -0.636</td> <td> 0.524</td> <td>   -0.141</td> <td>    0.072</td>
</tr>
<tr>
  <th>contact_telephone</th>                         <td>   -0.2408</td> <td>    0.058</td> <td>   -4.164</td> <td> 0.000</td> <td>   -0.354</td> <td>   -0.127</td>
</tr>
<tr>
  <th>poutcome_nonexistent</th>                      <td>    0.8844</td> <td>    0.249</td> <td>    3.559</td> <td> 0.000</td> <td>    0.397</td> <td>    1.371</td>
</tr>
<tr>
  <th>poutcome_success</th>                          <td>    1.0077</td> <td>    0.619</td> <td>    1.629</td> <td> 0.103</td> <td>   -0.205</td> <td>    2.220</td>
</tr>
<tr>
  <th>range_age_10_19</th>                           <td>   -0.7806</td> <td> 7817.938</td> <td>-9.98e-05</td> <td> 1.000</td> <td>-1.53e+04</td> <td> 1.53e+04</td>
</tr>
<tr>
  <th>range_age_20_29</th>                           <td>   -1.5302</td> <td> 7817.938</td> <td>   -0.000</td> <td> 1.000</td> <td>-1.53e+04</td> <td> 1.53e+04</td>
</tr>
<tr>
  <th>range_age_30_39</th>                           <td>   -1.7622</td> <td> 7817.938</td> <td>   -0.000</td> <td> 1.000</td> <td>-1.53e+04</td> <td> 1.53e+04</td>
</tr>
<tr>
  <th>range_age_40_49</th>                           <td>   -1.8071</td> <td> 7817.938</td> <td>   -0.000</td> <td> 1.000</td> <td>-1.53e+04</td> <td> 1.53e+04</td>
</tr>
<tr>
  <th>range_age_50_59</th>                           <td>   -1.6480</td> <td> 7817.938</td> <td>   -0.000</td> <td> 1.000</td> <td>-1.53e+04</td> <td> 1.53e+04</td>
</tr>
<tr>
  <th>range_age_60_69</th>                           <td>   -0.6249</td> <td> 7817.938</td> <td>-7.99e-05</td> <td> 1.000</td> <td>-1.53e+04</td> <td> 1.53e+04</td>
</tr>
<tr>
  <th>range_age_70_79</th>                           <td>   -0.8631</td> <td> 7817.938</td> <td>   -0.000</td> <td> 1.000</td> <td>-1.53e+04</td> <td> 1.53e+04</td>
</tr>
<tr>
  <th>range_age_80_89</th>                           <td>   -0.4608</td> <td> 7817.938</td> <td>-5.89e-05</td> <td> 1.000</td> <td>-1.53e+04</td> <td> 1.53e+04</td>
</tr>
<tr>
  <th>range_age_90_99</th>                           <td>  -24.5923</td> <td> 7.04e+04</td> <td>   -0.000</td> <td> 1.000</td> <td>-1.38e+05</td> <td> 1.38e+05</td>
</tr>
<tr>
  <th>personal_divorced+basic.4y+mon</th>            <td>   -0.0237</td> <td>    0.661</td> <td>   -0.036</td> <td> 0.971</td> <td>   -1.319</td> <td>    1.271</td>
</tr>
<tr>
  <th>personal_divorced+basic.4y+thu</th>            <td>    0.1233</td> <td>    0.676</td> <td>    0.182</td> <td> 0.855</td> <td>   -1.201</td> <td>    1.448</td>
</tr>
<tr>
  <th>personal_divorced+basic.4y+tue</th>            <td>    0.0617</td> <td>    0.644</td> <td>    0.096</td> <td> 0.924</td> <td>   -1.201</td> <td>    1.325</td>
</tr>
<tr>
  <th>personal_divorced+basic.4y+wed</th>            <td>    0.0393</td> <td>    0.652</td> <td>    0.060</td> <td> 0.952</td> <td>   -1.239</td> <td>    1.318</td>
</tr>
<tr>
  <th>personal_divorced+basic.6y+fri</th>            <td>  -20.6209</td> <td> 1.43e+04</td> <td>   -0.001</td> <td> 0.999</td> <td>-2.81e+04</td> <td> 2.81e+04</td>
</tr>
<tr>
  <th>personal_divorced+basic.6y+mon</th>            <td>   -0.7391</td> <td>    1.130</td> <td>   -0.654</td> <td> 0.513</td> <td>   -2.954</td> <td>    1.476</td>
</tr>
<tr>
  <th>personal_divorced+basic.6y+thu</th>            <td>    0.4156</td> <td>    0.777</td> <td>    0.535</td> <td> 0.593</td> <td>   -1.107</td> <td>    1.938</td>
</tr>
<tr>
  <th>personal_divorced+basic.6y+tue</th>            <td>    0.2322</td> <td>    0.781</td> <td>    0.297</td> <td> 0.766</td> <td>   -1.299</td> <td>    1.763</td>
</tr>
<tr>
  <th>personal_divorced+basic.6y+wed</th>            <td>   -0.4712</td> <td>    0.874</td> <td>   -0.539</td> <td> 0.590</td> <td>   -2.185</td> <td>    1.242</td>
</tr>
<tr>
  <th>personal_divorced+basic.9y+fri</th>            <td>    0.1016</td> <td>    0.624</td> <td>    0.163</td> <td> 0.871</td> <td>   -1.120</td> <td>    1.324</td>
</tr>
<tr>
  <th>personal_divorced+basic.9y+mon</th>            <td>   -0.8922</td> <td>    0.761</td> <td>   -1.172</td> <td> 0.241</td> <td>   -2.384</td> <td>    0.600</td>
</tr>
<tr>
  <th>personal_divorced+basic.9y+thu</th>            <td>    0.0949</td> <td>    0.623</td> <td>    0.152</td> <td> 0.879</td> <td>   -1.127</td> <td>    1.316</td>
</tr>
<tr>
  <th>personal_divorced+basic.9y+tue</th>            <td>   -0.0804</td> <td>    0.640</td> <td>   -0.126</td> <td> 0.900</td> <td>   -1.334</td> <td>    1.173</td>
</tr>
<tr>
  <th>personal_divorced+basic.9y+wed</th>            <td>   -0.7617</td> <td>    0.661</td> <td>   -1.152</td> <td> 0.249</td> <td>   -2.058</td> <td>    0.534</td>
</tr>
<tr>
  <th>personal_divorced+high.school+fri</th>         <td>   -0.1162</td> <td>    0.547</td> <td>   -0.213</td> <td> 0.832</td> <td>   -1.187</td> <td>    0.955</td>
</tr>
<tr>
  <th>personal_divorced+high.school+mon</th>         <td>   -0.4673</td> <td>    0.559</td> <td>   -0.836</td> <td> 0.403</td> <td>   -1.563</td> <td>    0.628</td>
</tr>
<tr>
  <th>personal_divorced+high.school+thu</th>         <td>   -0.0183</td> <td>    0.538</td> <td>   -0.034</td> <td> 0.973</td> <td>   -1.073</td> <td>    1.037</td>
</tr>
<tr>
  <th>personal_divorced+high.school+tue</th>         <td>   -0.1714</td> <td>    0.550</td> <td>   -0.312</td> <td> 0.755</td> <td>   -1.249</td> <td>    0.906</td>
</tr>
<tr>
  <th>personal_divorced+high.school+wed</th>         <td>   -0.0945</td> <td>    0.564</td> <td>   -0.168</td> <td> 0.867</td> <td>   -1.200</td> <td>    1.011</td>
</tr>
<tr>
  <th>personal_divorced+professional.course+fri</th> <td>   -0.1008</td> <td>    0.621</td> <td>   -0.162</td> <td> 0.871</td> <td>   -1.318</td> <td>    1.117</td>
</tr>
<tr>
  <th>personal_divorced+professional.course+mon</th> <td>   -0.0197</td> <td>    0.607</td> <td>   -0.033</td> <td> 0.974</td> <td>   -1.209</td> <td>    1.169</td>
</tr>
<tr>
  <th>personal_divorced+professional.course+thu</th> <td>    0.4150</td> <td>    0.578</td> <td>    0.718</td> <td> 0.473</td> <td>   -0.718</td> <td>    1.549</td>
</tr>
<tr>
  <th>personal_divorced+professional.course+tue</th> <td>   -0.2559</td> <td>    0.619</td> <td>   -0.414</td> <td> 0.679</td> <td>   -1.468</td> <td>    0.957</td>
</tr>
<tr>
  <th>personal_divorced+professional.course+wed</th> <td>   -0.0631</td> <td>    0.624</td> <td>   -0.101</td> <td> 0.919</td> <td>   -1.286</td> <td>    1.160</td>
</tr>
<tr>
  <th>personal_divorced+university.degree+fri</th>   <td>    0.0458</td> <td>    0.543</td> <td>    0.084</td> <td> 0.933</td> <td>   -1.019</td> <td>    1.110</td>
</tr>
<tr>
  <th>personal_divorced+university.degree+mon</th>   <td>    0.3157</td> <td>    0.519</td> <td>    0.608</td> <td> 0.543</td> <td>   -0.701</td> <td>    1.333</td>
</tr>
<tr>
  <th>personal_divorced+university.degree+thu</th>   <td>   -0.2652</td> <td>    0.556</td> <td>   -0.477</td> <td> 0.633</td> <td>   -1.355</td> <td>    0.825</td>
</tr>
<tr>
  <th>personal_divorced+university.degree+tue</th>   <td>   -0.3611</td> <td>    0.557</td> <td>   -0.648</td> <td> 0.517</td> <td>   -1.453</td> <td>    0.731</td>
</tr>
<tr>
  <th>personal_divorced+university.degree+wed</th>   <td>    0.4321</td> <td>    0.532</td> <td>    0.812</td> <td> 0.417</td> <td>   -0.611</td> <td>    1.475</td>
</tr>
<tr>
  <th>personal_married+basic.4y+fri</th>             <td>   -0.1064</td> <td>    0.517</td> <td>   -0.206</td> <td> 0.837</td> <td>   -1.121</td> <td>    0.908</td>
</tr>
<tr>
  <th>personal_married+basic.4y+mon</th>             <td>   -0.3463</td> <td>    0.512</td> <td>   -0.677</td> <td> 0.498</td> <td>   -1.349</td> <td>    0.656</td>
</tr>
<tr>
  <th>personal_married+basic.4y+thu</th>             <td>   -0.2877</td> <td>    0.513</td> <td>   -0.561</td> <td> 0.575</td> <td>   -1.293</td> <td>    0.718</td>
</tr>
<tr>
  <th>personal_married+basic.4y+tue</th>             <td>    0.0057</td> <td>    0.504</td> <td>    0.011</td> <td> 0.991</td> <td>   -0.983</td> <td>    0.994</td>
</tr>
<tr>
  <th>personal_married+basic.4y+wed</th>             <td>   -0.0473</td> <td>    0.508</td> <td>   -0.093</td> <td> 0.926</td> <td>   -1.044</td> <td>    0.949</td>
</tr>
<tr>
  <th>personal_married+basic.6y+fri</th>             <td>    0.1596</td> <td>    0.528</td> <td>    0.302</td> <td> 0.762</td> <td>   -0.875</td> <td>    1.194</td>
</tr>
<tr>
  <th>personal_married+basic.6y+mon</th>             <td>    0.0506</td> <td>    0.524</td> <td>    0.097</td> <td> 0.923</td> <td>   -0.977</td> <td>    1.078</td>
</tr>
<tr>
  <th>personal_married+basic.6y+thu</th>             <td>   -0.1602</td> <td>    0.528</td> <td>   -0.303</td> <td> 0.762</td> <td>   -1.196</td> <td>    0.875</td>
</tr>
<tr>
  <th>personal_married+basic.6y+tue</th>             <td>   -0.2038</td> <td>    0.537</td> <td>   -0.380</td> <td> 0.704</td> <td>   -1.256</td> <td>    0.848</td>
</tr>
<tr>
  <th>personal_married+basic.6y+wed</th>             <td>    0.3122</td> <td>    0.516</td> <td>    0.605</td> <td> 0.545</td> <td>   -0.700</td> <td>    1.324</td>
</tr>
<tr>
  <th>personal_married+basic.9y+fri</th>             <td>   -0.0352</td> <td>    0.499</td> <td>   -0.070</td> <td> 0.944</td> <td>   -1.014</td> <td>    0.943</td>
</tr>
<tr>
  <th>personal_married+basic.9y+mon</th>             <td>   -0.2330</td> <td>    0.501</td> <td>   -0.465</td> <td> 0.642</td> <td>   -1.216</td> <td>    0.750</td>
</tr>
<tr>
  <th>personal_married+basic.9y+thu</th>             <td>   -0.0327</td> <td>    0.498</td> <td>   -0.066</td> <td> 0.948</td> <td>   -1.008</td> <td>    0.943</td>
</tr>
<tr>
  <th>personal_married+basic.9y+tue</th>             <td>   -0.0783</td> <td>    0.501</td> <td>   -0.156</td> <td> 0.876</td> <td>   -1.060</td> <td>    0.903</td>
</tr>
<tr>
  <th>personal_married+basic.9y+wed</th>             <td>   -0.1971</td> <td>    0.502</td> <td>   -0.392</td> <td> 0.695</td> <td>   -1.182</td> <td>    0.787</td>
</tr>
<tr>
  <th>personal_married+high.school+fri</th>          <td>   -0.1628</td> <td>    0.494</td> <td>   -0.329</td> <td> 0.742</td> <td>   -1.131</td> <td>    0.806</td>
</tr>
<tr>
  <th>personal_married+high.school+mon</th>          <td>   -0.2929</td> <td>    0.495</td> <td>   -0.592</td> <td> 0.554</td> <td>   -1.263</td> <td>    0.677</td>
</tr>
<tr>
  <th>personal_married+high.school+thu</th>          <td>    0.1484</td> <td>    0.492</td> <td>    0.302</td> <td> 0.763</td> <td>   -0.816</td> <td>    1.113</td>
</tr>
<tr>
  <th>personal_married+high.school+tue</th>          <td>   -0.3530</td> <td>    0.498</td> <td>   -0.708</td> <td> 0.479</td> <td>   -1.330</td> <td>    0.624</td>
</tr>
<tr>
  <th>personal_married+high.school+wed</th>          <td>    0.0014</td> <td>    0.495</td> <td>    0.003</td> <td> 0.998</td> <td>   -0.969</td> <td>    0.971</td>
</tr>
<tr>
  <th>personal_married+illiterate+fri</th>           <td>  -20.4003</td> <td> 4.58e+04</td> <td>   -0.000</td> <td> 1.000</td> <td>-8.98e+04</td> <td> 8.97e+04</td>
</tr>
<tr>
  <th>personal_married+illiterate+mon</th>           <td>  -20.6992</td> <td> 7.95e+04</td> <td>   -0.000</td> <td> 1.000</td> <td>-1.56e+05</td> <td> 1.56e+05</td>
</tr>
<tr>
  <th>personal_married+illiterate+thu</th>           <td>    1.6013</td> <td>    1.018</td> <td>    1.573</td> <td> 0.116</td> <td>   -0.394</td> <td>    3.596</td>
</tr>
<tr>
  <th>personal_married+illiterate+tue</th>           <td>  -20.5489</td> <td> 3.97e+04</td> <td>   -0.001</td> <td> 1.000</td> <td>-7.78e+04</td> <td> 7.78e+04</td>
</tr>
<tr>
  <th>personal_married+professional.course+fri</th>  <td>   -0.1137</td> <td>    0.513</td> <td>   -0.222</td> <td> 0.824</td> <td>   -1.118</td> <td>    0.891</td>
</tr>
<tr>
  <th>personal_married+professional.course+mon</th>  <td>   -0.1748</td> <td>    0.512</td> <td>   -0.342</td> <td> 0.733</td> <td>   -1.178</td> <td>    0.829</td>
</tr>
<tr>
  <th>personal_married+professional.course+thu</th>  <td>    0.1068</td> <td>    0.502</td> <td>    0.213</td> <td> 0.831</td> <td>   -0.877</td> <td>    1.090</td>
</tr>
<tr>
  <th>personal_married+professional.course+tue</th>  <td>   -0.0386</td> <td>    0.512</td> <td>   -0.076</td> <td> 0.940</td> <td>   -1.041</td> <td>    0.964</td>
</tr>
<tr>
  <th>personal_married+professional.course+wed</th>  <td>   -0.0867</td> <td>    0.512</td> <td>   -0.169</td> <td> 0.866</td> <td>   -1.091</td> <td>    0.917</td>
</tr>
<tr>
  <th>personal_married+university.degree+fri</th>    <td>   -0.1411</td> <td>    0.496</td> <td>   -0.284</td> <td> 0.776</td> <td>   -1.113</td> <td>    0.831</td>
</tr>
<tr>
  <th>personal_married+university.degree+mon</th>    <td>    0.1265</td> <td>    0.491</td> <td>    0.258</td> <td> 0.797</td> <td>   -0.836</td> <td>    1.089</td>
</tr>
<tr>
  <th>personal_married+university.degree+thu</th>    <td>    0.0745</td> <td>    0.491</td> <td>    0.152</td> <td> 0.879</td> <td>   -0.888</td> <td>    1.037</td>
</tr>
<tr>
  <th>personal_married+university.degree+tue</th>    <td>    0.1608</td> <td>    0.493</td> <td>    0.326</td> <td> 0.744</td> <td>   -0.806</td> <td>    1.127</td>
</tr>
<tr>
  <th>personal_married+university.degree+wed</th>    <td>    0.2640</td> <td>    0.491</td> <td>    0.537</td> <td> 0.591</td> <td>   -0.699</td> <td>    1.227</td>
</tr>
<tr>
  <th>personal_single+basic.4y+fri</th>              <td>   -1.8290</td> <td>    1.117</td> <td>   -1.638</td> <td> 0.101</td> <td>   -4.018</td> <td>    0.360</td>
</tr>
<tr>
  <th>personal_single+basic.4y+mon</th>              <td>   -0.5457</td> <td>    0.768</td> <td>   -0.710</td> <td> 0.477</td> <td>   -2.051</td> <td>    0.960</td>
</tr>
<tr>
  <th>personal_single+basic.4y+thu</th>              <td>   -0.6846</td> <td>    0.672</td> <td>   -1.018</td> <td> 0.308</td> <td>   -2.002</td> <td>    0.633</td>
</tr>
<tr>
  <th>personal_single+basic.4y+tue</th>              <td>   -0.5607</td> <td>    0.679</td> <td>   -0.825</td> <td> 0.409</td> <td>   -1.892</td> <td>    0.771</td>
</tr>
<tr>
  <th>personal_single+basic.4y+wed</th>              <td>   -0.2173</td> <td>    0.679</td> <td>   -0.320</td> <td> 0.749</td> <td>   -1.549</td> <td>    1.114</td>
</tr>
<tr>
  <th>personal_single+basic.6y+fri</th>              <td>   -0.1164</td> <td>    0.669</td> <td>   -0.174</td> <td> 0.862</td> <td>   -1.428</td> <td>    1.195</td>
</tr>
<tr>
  <th>personal_single+basic.6y+mon</th>              <td>   -1.4020</td> <td>    1.164</td> <td>   -1.204</td> <td> 0.229</td> <td>   -3.684</td> <td>    0.880</td>
</tr>
<tr>
  <th>personal_single+basic.6y+thu</th>              <td>    0.2608</td> <td>    0.647</td> <td>    0.403</td> <td> 0.687</td> <td>   -1.008</td> <td>    1.530</td>
</tr>
<tr>
  <th>personal_single+basic.6y+tue</th>              <td>    0.8037</td> <td>    0.638</td> <td>    1.259</td> <td> 0.208</td> <td>   -0.448</td> <td>    2.055</td>
</tr>
<tr>
  <th>personal_single+basic.6y+wed</th>              <td>   -0.1306</td> <td>    0.677</td> <td>   -0.193</td> <td> 0.847</td> <td>   -1.458</td> <td>    1.196</td>
</tr>
<tr>
  <th>personal_single+basic.9y+fri</th>              <td>   -0.0098</td> <td>    0.542</td> <td>   -0.018</td> <td> 0.986</td> <td>   -1.073</td> <td>    1.053</td>
</tr>
<tr>
  <th>personal_single+basic.9y+mon</th>              <td>   -0.3577</td> <td>    0.544</td> <td>   -0.657</td> <td> 0.511</td> <td>   -1.424</td> <td>    0.709</td>
</tr>
<tr>
  <th>personal_single+basic.9y+thu</th>              <td>   -0.0374</td> <td>    0.529</td> <td>   -0.071</td> <td> 0.944</td> <td>   -1.074</td> <td>    0.999</td>
</tr>
<tr>
  <th>personal_single+basic.9y+tue</th>              <td>    0.0414</td> <td>    0.551</td> <td>    0.075</td> <td> 0.940</td> <td>   -1.039</td> <td>    1.122</td>
</tr>
<tr>
  <th>personal_single+basic.9y+wed</th>              <td>   -0.2541</td> <td>    0.545</td> <td>   -0.466</td> <td> 0.641</td> <td>   -1.323</td> <td>    0.815</td>
</tr>
<tr>
  <th>personal_single+high.school+fri</th>           <td>    0.1764</td> <td>    0.499</td> <td>    0.354</td> <td> 0.723</td> <td>   -0.801</td> <td>    1.154</td>
</tr>
<tr>
  <th>personal_single+high.school+mon</th>           <td>    0.0422</td> <td>    0.501</td> <td>    0.084</td> <td> 0.933</td> <td>   -0.939</td> <td>    1.023</td>
</tr>
<tr>
  <th>personal_single+high.school+thu</th>           <td>    0.0840</td> <td>    0.499</td> <td>    0.168</td> <td> 0.866</td> <td>   -0.894</td> <td>    1.062</td>
</tr>
<tr>
  <th>personal_single+high.school+tue</th>           <td>    0.0477</td> <td>    0.503</td> <td>    0.095</td> <td> 0.924</td> <td>   -0.938</td> <td>    1.033</td>
</tr>
<tr>
  <th>personal_single+high.school+wed</th>           <td>    0.3140</td> <td>    0.496</td> <td>    0.634</td> <td> 0.526</td> <td>   -0.657</td> <td>    1.285</td>
</tr>
<tr>
  <th>personal_single+illiterate+fri</th>            <td>  -21.7309</td> <td> 7.95e+04</td> <td>   -0.000</td> <td> 1.000</td> <td>-1.56e+05</td> <td> 1.56e+05</td>
</tr>
<tr>
  <th>personal_single+professional.course+fri</th>   <td>    0.1416</td> <td>    0.537</td> <td>    0.264</td> <td> 0.792</td> <td>   -0.911</td> <td>    1.195</td>
</tr>
<tr>
  <th>personal_single+professional.course+mon</th>   <td>   -0.2273</td> <td>    0.539</td> <td>   -0.422</td> <td> 0.673</td> <td>   -1.284</td> <td>    0.829</td>
</tr>
<tr>
  <th>personal_single+professional.course+thu</th>   <td>   -0.2723</td> <td>    0.539</td> <td>   -0.505</td> <td> 0.614</td> <td>   -1.329</td> <td>    0.785</td>
</tr>
<tr>
  <th>personal_single+professional.course+tue</th>   <td>    0.1015</td> <td>    0.531</td> <td>    0.191</td> <td> 0.848</td> <td>   -0.940</td> <td>    1.143</td>
</tr>
<tr>
  <th>personal_single+professional.course+wed</th>   <td>   -0.0236</td> <td>    0.534</td> <td>   -0.044</td> <td> 0.965</td> <td>   -1.071</td> <td>    1.024</td>
</tr>
<tr>
  <th>personal_single+university.degree+fri</th>     <td>    0.1428</td> <td>    0.499</td> <td>    0.286</td> <td> 0.775</td> <td>   -0.835</td> <td>    1.120</td>
</tr>
<tr>
  <th>personal_single+university.degree+mon</th>     <td>    0.0235</td> <td>    0.497</td> <td>    0.047</td> <td> 0.962</td> <td>   -0.950</td> <td>    0.997</td>
</tr>
<tr>
  <th>personal_single+university.degree+thu</th>     <td>    0.3760</td> <td>    0.492</td> <td>    0.764</td> <td> 0.445</td> <td>   -0.588</td> <td>    1.340</td>
</tr>
<tr>
  <th>personal_single+university.degree+tue</th>     <td>    0.2269</td> <td>    0.496</td> <td>    0.457</td> <td> 0.648</td> <td>   -0.746</td> <td>    1.200</td>
</tr>
<tr>
  <th>personal_single+university.degree+wed</th>     <td>    0.1784</td> <td>    0.497</td> <td>    0.359</td> <td> 0.720</td> <td>   -0.796</td> <td>    1.153</td>
</tr>
</table>



### AICの確認


```python
print("AIC: {0}".format(res.aic))
```

    AIC: 18160.375645013897


### CPR(Component Plus Residual) プロットで目的関数と説明変数の線形性を確認


```python
# plt = CPRProtFigure(X, res)
```

## 予測パート

### AUCを確認する


```python
# とりあえず、元データから予測する
predictData = X.copy()

# 予測の実施
prY = res.predict(predictData)
#  予測結果と実際の値よりAUCを計算する
plt, auc = ROCPlot(Y, prY)
print("AUC: {0}".format(auc))
```

    AUC: 0.6964248020628219



![png](output_32_1.png)


### 利益が最大化するポイントを算出


```python
plt, y, t, tp, fn, fp, tn = BANKMarketingPredictProfit(Y, prY)
print("Max Profit: {0}, Threshold: {1}".format(y, t))
print("ConfusionMatrix\n\t{0}\t\t{1}\n\t{2}\t\t{3}".format(tp, fn, fp, tn))
```

    Max Profit: 339500, Threshold: 0.21000000000000002
    ConfusionMatrix
    	510		2237
    	851		33470



![png](output_34_1.png)


## （おまけ、いや本題）一発で予測結果取得


```python
t = 0.21
train = pd.read_csv('bank_marketing_train.csv')
test = pd.read_csv('bank_marketing_test-1.csv')
temp, y, tp, fn, fp, tn = ippatu(train,test,t)
print("Max Profit: {0}, Threshold: {1}".format(y, t))
print("ConfusionMatrix\n\t{0}\t\t{1}\n\t{2}\t\t{3}".format(tp, fn, fp, tn))
result
```

    Max Profit: 888000, Threshold: 0.21
    ConfusionMatrix
    	835		1058
    	729		1498





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>housing</th>
      <th>loan</th>
      <th>day_of_week</th>
      <th>duration</th>
      <th>campaign</th>
      <th>pdays</th>
      <th>previous</th>
      <th>poutcome</th>
      <th>emp.var.rate</th>
      <th>cons.price.idx</th>
      <th>cons.conf.idx</th>
      <th>euribor3m</th>
      <th>nr.employed</th>
      <th>y</th>
      <th>personal</th>
      <th>predict</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>32</td>
      <td>single</td>
      <td>university.degree</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>wed</td>
      <td>210</td>
      <td>5</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>-2.9</td>
      <td>92.469</td>
      <td>-33.6</td>
      <td>1.029</td>
      <td>5076.2</td>
      <td>yes</td>
      <td>cellular+jul+30_39+admin.</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>59</td>
      <td>married</td>
      <td>high.school</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>wed</td>
      <td>286</td>
      <td>2</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>-2.9</td>
      <td>92.469</td>
      <td>-33.6</td>
      <td>1.029</td>
      <td>5076.2</td>
      <td>no</td>
      <td>cellular+jul+50_59+admin.</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>40</td>
      <td>married</td>
      <td>high.school</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>wed</td>
      <td>475</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>success</td>
      <td>-2.9</td>
      <td>92.469</td>
      <td>-33.6</td>
      <td>1.029</td>
      <td>5076.2</td>
      <td>no</td>
      <td>cellular+jul+30_39+technician</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>26</td>
      <td>single</td>
      <td>university.degree</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>wed</td>
      <td>153</td>
      <td>1</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>-2.9</td>
      <td>92.469</td>
      <td>-33.6</td>
      <td>1.029</td>
      <td>5076.2</td>
      <td>yes</td>
      <td>cellular+jul+20_29+admin.</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>36</td>
      <td>married</td>
      <td>professional.course</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>wed</td>
      <td>182</td>
      <td>3</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>-2.9</td>
      <td>92.469</td>
      <td>-33.6</td>
      <td>1.029</td>
      <td>5076.2</td>
      <td>no</td>
      <td>cellular+jul+30_39+technician</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>32</td>
      <td>single</td>
      <td>university.degree</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>wed</td>
      <td>97</td>
      <td>3</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>-2.9</td>
      <td>92.469</td>
      <td>-33.6</td>
      <td>1.029</td>
      <td>5076.2</td>
      <td>yes</td>
      <td>cellular+jul+30_39+admin.</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>38</td>
      <td>married</td>
      <td>high.school</td>
      <td>no</td>
      <td>yes</td>
      <td>yes</td>
      <td>wed</td>
      <td>64</td>
      <td>3</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>-2.9</td>
      <td>92.469</td>
      <td>-33.6</td>
      <td>1.029</td>
      <td>5076.2</td>
      <td>no</td>
      <td>telephone+jul+30_39+admin.</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>29</td>
      <td>single</td>
      <td>high.school</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>thu</td>
      <td>253</td>
      <td>2</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>-2.9</td>
      <td>92.469</td>
      <td>-33.6</td>
      <td>1.018</td>
      <td>5076.2</td>
      <td>yes</td>
      <td>cellular+jul+20_29+student</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>33</td>
      <td>single</td>
      <td>university.degree</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>thu</td>
      <td>359</td>
      <td>2</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>-2.9</td>
      <td>92.469</td>
      <td>-33.6</td>
      <td>1.018</td>
      <td>5076.2</td>
      <td>no</td>
      <td>cellular+jul+30_39+technician</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>32</td>
      <td>single</td>
      <td>university.degree</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>thu</td>
      <td>315</td>
      <td>4</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>-2.9</td>
      <td>92.469</td>
      <td>-33.6</td>
      <td>1.018</td>
      <td>5076.2</td>
      <td>no</td>
      <td>cellular+jul+30_39+admin.</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>23</td>
      <td>single</td>
      <td>high.school</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>fri</td>
      <td>104</td>
      <td>2</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>-2.9</td>
      <td>92.469</td>
      <td>-33.6</td>
      <td>1.007</td>
      <td>5076.2</td>
      <td>yes</td>
      <td>cellular+jul+20_29+blue-collar</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>32</td>
      <td>single</td>
      <td>university.degree</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>fri</td>
      <td>123</td>
      <td>4</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>-2.9</td>
      <td>92.469</td>
      <td>-33.6</td>
      <td>1.007</td>
      <td>5076.2</td>
      <td>no</td>
      <td>telephone+jul+30_39+admin.</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>23</td>
      <td>single</td>
      <td>professional.course</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>fri</td>
      <td>336</td>
      <td>4</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>-2.9</td>
      <td>92.469</td>
      <td>-33.6</td>
      <td>1.007</td>
      <td>5076.2</td>
      <td>yes</td>
      <td>telephone+jul+20_29+technician</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>29</td>
      <td>single</td>
      <td>university.degree</td>
      <td>no</td>
      <td>yes</td>
      <td>yes</td>
      <td>mon</td>
      <td>326</td>
      <td>3</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>-2.9</td>
      <td>92.469</td>
      <td>-33.6</td>
      <td>0.996</td>
      <td>5076.2</td>
      <td>no</td>
      <td>cellular+jul+20_29+technician</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>51</td>
      <td>divorced</td>
      <td>university.degree</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>wed</td>
      <td>219</td>
      <td>2</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>-2.9</td>
      <td>92.469</td>
      <td>-33.6</td>
      <td>0.979</td>
      <td>5076.2</td>
      <td>no</td>
      <td>telephone+jul+50_59+management</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>26</td>
      <td>single</td>
      <td>university.degree</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>wed</td>
      <td>226</td>
      <td>2</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>-2.9</td>
      <td>92.469</td>
      <td>-33.6</td>
      <td>0.979</td>
      <td>5076.2</td>
      <td>no</td>
      <td>cellular+jul+20_29+admin.</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>51</td>
      <td>divorced</td>
      <td>university.degree</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>wed</td>
      <td>166</td>
      <td>7</td>
      <td>999</td>
      <td>1</td>
      <td>failure</td>
      <td>-2.9</td>
      <td>92.469</td>
      <td>-33.6</td>
      <td>0.979</td>
      <td>5076.2</td>
      <td>no</td>
      <td>cellular+jul+50_59+management</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>45</td>
      <td>married</td>
      <td>university.degree</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>thu</td>
      <td>58</td>
      <td>3</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>-2.9</td>
      <td>92.469</td>
      <td>-33.6</td>
      <td>0.969</td>
      <td>5076.2</td>
      <td>no</td>
      <td>telephone+jul+40_49+admin.</td>
      <td>1</td>
    </tr>
    <tr>
      <th>18</th>
      <td>27</td>
      <td>single</td>
      <td>university.degree</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>mon</td>
      <td>125</td>
      <td>2</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>-2.9</td>
      <td>92.469</td>
      <td>-33.6</td>
      <td>0.944</td>
      <td>5076.2</td>
      <td>yes</td>
      <td>cellular+jul+20_29+management</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19</th>
      <td>37</td>
      <td>married</td>
      <td>high.school</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>mon</td>
      <td>91</td>
      <td>2</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>-2.9</td>
      <td>92.469</td>
      <td>-33.6</td>
      <td>0.944</td>
      <td>5076.2</td>
      <td>no</td>
      <td>cellular+jul+30_39+services</td>
      <td>1</td>
    </tr>
    <tr>
      <th>20</th>
      <td>27</td>
      <td>single</td>
      <td>university.degree</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>mon</td>
      <td>258</td>
      <td>3</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>-2.9</td>
      <td>92.469</td>
      <td>-33.6</td>
      <td>0.944</td>
      <td>5076.2</td>
      <td>no</td>
      <td>cellular+jul+20_29+technician</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21</th>
      <td>36</td>
      <td>married</td>
      <td>high.school</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>tue</td>
      <td>305</td>
      <td>7</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>-2.9</td>
      <td>92.469</td>
      <td>-33.6</td>
      <td>0.937</td>
      <td>5076.2</td>
      <td>no</td>
      <td>telephone+jul+30_39+admin.</td>
      <td>1</td>
    </tr>
    <tr>
      <th>22</th>
      <td>53</td>
      <td>married</td>
      <td>university.degree</td>
      <td>no</td>
      <td>yes</td>
      <td>yes</td>
      <td>tue</td>
      <td>465</td>
      <td>6</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>-2.9</td>
      <td>92.469</td>
      <td>-33.6</td>
      <td>0.937</td>
      <td>5076.2</td>
      <td>no</td>
      <td>telephone+jul+50_59+management</td>
      <td>1</td>
    </tr>
    <tr>
      <th>23</th>
      <td>35</td>
      <td>married</td>
      <td>high.school</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>wed</td>
      <td>1084</td>
      <td>4</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>-2.9</td>
      <td>92.469</td>
      <td>-33.6</td>
      <td>0.933</td>
      <td>5076.2</td>
      <td>yes</td>
      <td>cellular+jul+30_39+entrepreneur</td>
      <td>1</td>
    </tr>
    <tr>
      <th>24</th>
      <td>33</td>
      <td>divorced</td>
      <td>university.degree</td>
      <td>no</td>
      <td>no</td>
      <td>yes</td>
      <td>thu</td>
      <td>106</td>
      <td>3</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>-2.9</td>
      <td>92.469</td>
      <td>-33.6</td>
      <td>0.927</td>
      <td>5076.2</td>
      <td>no</td>
      <td>cellular+jul+30_39+admin.</td>
      <td>1</td>
    </tr>
    <tr>
      <th>25</th>
      <td>27</td>
      <td>single</td>
      <td>high.school</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>thu</td>
      <td>64</td>
      <td>3</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>-2.9</td>
      <td>92.469</td>
      <td>-33.6</td>
      <td>0.927</td>
      <td>5076.2</td>
      <td>no</td>
      <td>cellular+jul+20_29+student</td>
      <td>1</td>
    </tr>
    <tr>
      <th>26</th>
      <td>51</td>
      <td>married</td>
      <td>university.degree</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>fri</td>
      <td>139</td>
      <td>5</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>-2.9</td>
      <td>92.469</td>
      <td>-33.6</td>
      <td>0.921</td>
      <td>5076.2</td>
      <td>yes</td>
      <td>cellular+jul+50_59+admin.</td>
      <td>0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>51</td>
      <td>married</td>
      <td>university.degree</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>fri</td>
      <td>116</td>
      <td>3</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>-2.9</td>
      <td>92.469</td>
      <td>-33.6</td>
      <td>0.921</td>
      <td>5076.2</td>
      <td>no</td>
      <td>telephone+jul+50_59+admin.</td>
      <td>1</td>
    </tr>
    <tr>
      <th>28</th>
      <td>50</td>
      <td>married</td>
      <td>university.degree</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>mon</td>
      <td>506</td>
      <td>2</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>-2.9</td>
      <td>92.469</td>
      <td>-33.6</td>
      <td>0.914</td>
      <td>5076.2</td>
      <td>no</td>
      <td>cellular+jul+40_49+self-employed</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29</th>
      <td>38</td>
      <td>single</td>
      <td>university.degree</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>mon</td>
      <td>195</td>
      <td>2</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>-2.9</td>
      <td>92.469</td>
      <td>-33.6</td>
      <td>0.914</td>
      <td>5076.2</td>
      <td>no</td>
      <td>telephone+jul+30_39+admin.</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4090</th>
      <td>35</td>
      <td>divorced</td>
      <td>basic.4y</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>tue</td>
      <td>363</td>
      <td>1</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>-1.1</td>
      <td>94.767</td>
      <td>-50.8</td>
      <td>1.035</td>
      <td>4963.6</td>
      <td>yes</td>
      <td>cellular+nov+30_39+technician</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4091</th>
      <td>35</td>
      <td>divorced</td>
      <td>basic.4y</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>tue</td>
      <td>514</td>
      <td>1</td>
      <td>9</td>
      <td>4</td>
      <td>success</td>
      <td>-1.1</td>
      <td>94.767</td>
      <td>-50.8</td>
      <td>1.035</td>
      <td>4963.6</td>
      <td>yes</td>
      <td>cellular+nov+30_39+technician</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4092</th>
      <td>33</td>
      <td>married</td>
      <td>university.degree</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>tue</td>
      <td>843</td>
      <td>1</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>-1.1</td>
      <td>94.767</td>
      <td>-50.8</td>
      <td>1.035</td>
      <td>4963.6</td>
      <td>yes</td>
      <td>cellular+nov+30_39+admin.</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4093</th>
      <td>33</td>
      <td>married</td>
      <td>university.degree</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>tue</td>
      <td>510</td>
      <td>1</td>
      <td>999</td>
      <td>1</td>
      <td>failure</td>
      <td>-1.1</td>
      <td>94.767</td>
      <td>-50.8</td>
      <td>1.035</td>
      <td>4963.6</td>
      <td>no</td>
      <td>cellular+nov+30_39+admin.</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4094</th>
      <td>60</td>
      <td>married</td>
      <td>basic.4y</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>tue</td>
      <td>347</td>
      <td>2</td>
      <td>4</td>
      <td>1</td>
      <td>success</td>
      <td>-1.1</td>
      <td>94.767</td>
      <td>-50.8</td>
      <td>1.035</td>
      <td>4963.6</td>
      <td>no</td>
      <td>cellular+nov+50_59+blue-collar</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4095</th>
      <td>35</td>
      <td>divorced</td>
      <td>basic.4y</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>tue</td>
      <td>385</td>
      <td>3</td>
      <td>4</td>
      <td>2</td>
      <td>success</td>
      <td>-1.1</td>
      <td>94.767</td>
      <td>-50.8</td>
      <td>1.035</td>
      <td>4963.6</td>
      <td>yes</td>
      <td>cellular+nov+30_39+technician</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4096</th>
      <td>54</td>
      <td>married</td>
      <td>professional.course</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>tue</td>
      <td>1868</td>
      <td>2</td>
      <td>10</td>
      <td>1</td>
      <td>success</td>
      <td>-1.1</td>
      <td>94.767</td>
      <td>-50.8</td>
      <td>1.035</td>
      <td>4963.6</td>
      <td>yes</td>
      <td>cellular+nov+50_59+admin.</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4097</th>
      <td>38</td>
      <td>divorced</td>
      <td>university.degree</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>wed</td>
      <td>403</td>
      <td>2</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>-1.1</td>
      <td>94.767</td>
      <td>-50.8</td>
      <td>1.030</td>
      <td>4963.6</td>
      <td>yes</td>
      <td>cellular+nov+30_39+housemaid</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4098</th>
      <td>32</td>
      <td>married</td>
      <td>university.degree</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>wed</td>
      <td>651</td>
      <td>1</td>
      <td>999</td>
      <td>1</td>
      <td>failure</td>
      <td>-1.1</td>
      <td>94.767</td>
      <td>-50.8</td>
      <td>1.030</td>
      <td>4963.6</td>
      <td>yes</td>
      <td>telephone+nov+30_39+admin.</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4099</th>
      <td>32</td>
      <td>married</td>
      <td>university.degree</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>wed</td>
      <td>236</td>
      <td>3</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>-1.1</td>
      <td>94.767</td>
      <td>-50.8</td>
      <td>1.030</td>
      <td>4963.6</td>
      <td>no</td>
      <td>cellular+nov+30_39+admin.</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4100</th>
      <td>38</td>
      <td>married</td>
      <td>university.degree</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>wed</td>
      <td>144</td>
      <td>2</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>-1.1</td>
      <td>94.767</td>
      <td>-50.8</td>
      <td>1.030</td>
      <td>4963.6</td>
      <td>no</td>
      <td>cellular+nov+30_39+entrepreneur</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4101</th>
      <td>62</td>
      <td>married</td>
      <td>high.school</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>wed</td>
      <td>154</td>
      <td>5</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>-1.1</td>
      <td>94.767</td>
      <td>-50.8</td>
      <td>1.030</td>
      <td>4963.6</td>
      <td>no</td>
      <td>cellular+nov+60_69+services</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4102</th>
      <td>40</td>
      <td>divorced</td>
      <td>university.degree</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>wed</td>
      <td>293</td>
      <td>2</td>
      <td>999</td>
      <td>4</td>
      <td>failure</td>
      <td>-1.1</td>
      <td>94.767</td>
      <td>-50.8</td>
      <td>1.030</td>
      <td>4963.6</td>
      <td>no</td>
      <td>cellular+nov+30_39+management</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4103</th>
      <td>33</td>
      <td>married</td>
      <td>professional.course</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>thu</td>
      <td>112</td>
      <td>1</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>-1.1</td>
      <td>94.767</td>
      <td>-50.8</td>
      <td>1.031</td>
      <td>4963.6</td>
      <td>yes</td>
      <td>telephone+nov+30_39+student</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4104</th>
      <td>31</td>
      <td>single</td>
      <td>university.degree</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>thu</td>
      <td>353</td>
      <td>1</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>-1.1</td>
      <td>94.767</td>
      <td>-50.8</td>
      <td>1.031</td>
      <td>4963.6</td>
      <td>yes</td>
      <td>cellular+nov+30_39+admin.</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4105</th>
      <td>62</td>
      <td>married</td>
      <td>university.degree</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>thu</td>
      <td>329</td>
      <td>1</td>
      <td>999</td>
      <td>2</td>
      <td>failure</td>
      <td>-1.1</td>
      <td>94.767</td>
      <td>-50.8</td>
      <td>1.031</td>
      <td>4963.6</td>
      <td>yes</td>
      <td>cellular+nov+60_69+retired</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4106</th>
      <td>62</td>
      <td>married</td>
      <td>university.degree</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>thu</td>
      <td>208</td>
      <td>1</td>
      <td>1</td>
      <td>6</td>
      <td>success</td>
      <td>-1.1</td>
      <td>94.767</td>
      <td>-50.8</td>
      <td>1.031</td>
      <td>4963.6</td>
      <td>yes</td>
      <td>cellular+nov+60_69+retired</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4107</th>
      <td>34</td>
      <td>single</td>
      <td>university.degree</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>thu</td>
      <td>180</td>
      <td>1</td>
      <td>999</td>
      <td>2</td>
      <td>failure</td>
      <td>-1.1</td>
      <td>94.767</td>
      <td>-50.8</td>
      <td>1.031</td>
      <td>4963.6</td>
      <td>no</td>
      <td>cellular+nov+30_39+student</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4108</th>
      <td>38</td>
      <td>divorced</td>
      <td>high.school</td>
      <td>no</td>
      <td>yes</td>
      <td>yes</td>
      <td>thu</td>
      <td>360</td>
      <td>1</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>-1.1</td>
      <td>94.767</td>
      <td>-50.8</td>
      <td>1.031</td>
      <td>4963.6</td>
      <td>no</td>
      <td>cellular+nov+30_39+housemaid</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4109</th>
      <td>57</td>
      <td>married</td>
      <td>professional.course</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>thu</td>
      <td>124</td>
      <td>6</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>-1.1</td>
      <td>94.767</td>
      <td>-50.8</td>
      <td>1.031</td>
      <td>4963.6</td>
      <td>no</td>
      <td>cellular+nov+50_59+retired</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4110</th>
      <td>62</td>
      <td>married</td>
      <td>university.degree</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>thu</td>
      <td>483</td>
      <td>2</td>
      <td>6</td>
      <td>3</td>
      <td>success</td>
      <td>-1.1</td>
      <td>94.767</td>
      <td>-50.8</td>
      <td>1.031</td>
      <td>4963.6</td>
      <td>yes</td>
      <td>cellular+nov+60_69+retired</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4111</th>
      <td>64</td>
      <td>divorced</td>
      <td>professional.course</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>fri</td>
      <td>151</td>
      <td>3</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>-1.1</td>
      <td>94.767</td>
      <td>-50.8</td>
      <td>1.028</td>
      <td>4963.6</td>
      <td>no</td>
      <td>cellular+nov+60_69+retired</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4112</th>
      <td>36</td>
      <td>married</td>
      <td>university.degree</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>fri</td>
      <td>254</td>
      <td>2</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>-1.1</td>
      <td>94.767</td>
      <td>-50.8</td>
      <td>1.028</td>
      <td>4963.6</td>
      <td>no</td>
      <td>cellular+nov+30_39+admin.</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4113</th>
      <td>37</td>
      <td>married</td>
      <td>university.degree</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>fri</td>
      <td>281</td>
      <td>1</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>-1.1</td>
      <td>94.767</td>
      <td>-50.8</td>
      <td>1.028</td>
      <td>4963.6</td>
      <td>yes</td>
      <td>cellular+nov+30_39+admin.</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4114</th>
      <td>29</td>
      <td>single</td>
      <td>basic.4y</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>fri</td>
      <td>112</td>
      <td>1</td>
      <td>9</td>
      <td>1</td>
      <td>success</td>
      <td>-1.1</td>
      <td>94.767</td>
      <td>-50.8</td>
      <td>1.028</td>
      <td>4963.6</td>
      <td>no</td>
      <td>cellular+nov+20_29+unemployed</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4115</th>
      <td>73</td>
      <td>married</td>
      <td>professional.course</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>fri</td>
      <td>334</td>
      <td>1</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>-1.1</td>
      <td>94.767</td>
      <td>-50.8</td>
      <td>1.028</td>
      <td>4963.6</td>
      <td>yes</td>
      <td>cellular+nov+70_79+retired</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4116</th>
      <td>46</td>
      <td>married</td>
      <td>professional.course</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>fri</td>
      <td>383</td>
      <td>1</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>-1.1</td>
      <td>94.767</td>
      <td>-50.8</td>
      <td>1.028</td>
      <td>4963.6</td>
      <td>no</td>
      <td>cellular+nov+40_49+blue-collar</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4117</th>
      <td>56</td>
      <td>married</td>
      <td>university.degree</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>fri</td>
      <td>189</td>
      <td>2</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>-1.1</td>
      <td>94.767</td>
      <td>-50.8</td>
      <td>1.028</td>
      <td>4963.6</td>
      <td>no</td>
      <td>cellular+nov+50_59+retired</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4118</th>
      <td>44</td>
      <td>married</td>
      <td>professional.course</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>fri</td>
      <td>442</td>
      <td>1</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>-1.1</td>
      <td>94.767</td>
      <td>-50.8</td>
      <td>1.028</td>
      <td>4963.6</td>
      <td>yes</td>
      <td>cellular+nov+40_49+technician</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4119</th>
      <td>74</td>
      <td>married</td>
      <td>professional.course</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>fri</td>
      <td>239</td>
      <td>3</td>
      <td>999</td>
      <td>1</td>
      <td>failure</td>
      <td>-1.1</td>
      <td>94.767</td>
      <td>-50.8</td>
      <td>1.028</td>
      <td>4963.6</td>
      <td>no</td>
      <td>cellular+nov+70_79+retired</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>4120 rows × 20 columns</p>
</div>


