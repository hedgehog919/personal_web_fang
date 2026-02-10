# 論文資訊
作者: Fuli Feng, Xiangnan He, Xiang Wang, Cheng Luo, and Yiquun Liu

發表日期: 2019年4月11日

會議/期刊: ACM Transactions on Information Systems (TOIS), Volume 37, Issue 2, Article No. 27

引用數量: 464次

DOI: 10.1145/3307392

Link: [論文原文](https://dl.acm.org/doi/pdf/10.1145/3309547)、[論文程式碼](https://github.com/fulifeng/Temporal_Relational_Stock_Ranking)


# 研究動機
這篇論文指出傳統深度學習股票預測方法存在忽略股票間的關聯性的問題，將每支股票視為獨立的時間序列，忽略了同產業股票可能有相似走勢、供應鏈上下游公司股價可能相互影響等重要資訊。

另外，論文還指出現有方法通常將任務定義為二分類(漲跌)或是迴歸(股價)問題，存在**某方法的均方誤差較小，也可能選出報酬較低的股票** 的致命缺點，然而投資者真正關心的是哪支股票的預期報酬最高。

# 方法論
為解決這些問題，論文提出了 Relational Stock Ranking (RSR) 關聯性股票排名框架，並設計時間圖卷積模組 Temporal Graph Convolution (TGC) 來動態捕捉股票間的關係，並將股票預測重新定義為排序任務。
![image](/posts/temporal-relational-stock-ranking/framework.png)
這個框架首先將每支股票的歷史價格序列輸入 LSTM 網路，學習出代表該股票時序特徵的序列嵌入向量，接著以產業關係、Wiki 公司關係定義股票關聯圖，時間圖卷積模組就會以此關聯圖更新節點嵌入向量，使得股票包含關聯股票的時序資訊，最後，將序列嵌入向量與節點嵌入向量串接後，輸入全連接層產生每支股票的排序分數，依此排序推薦投資標的。

### 序列嵌入(Sequential Embedding)
在序列嵌入，論文使用 LSTM 網路處理每支股票的歷史價格序列，捕捉長期時序依賴關係，也代表圖上該節點的初始特徵向量。

輸入為股票 $i$ 在時間 $t$ 的歷史特徵矩陣 $\mathbf{X}_i^t \in \mathbb{R}^{S \times D}$，輸出為序列嵌入向量：

$$\mathbf{E}^t = \text{LSTM}(\mathbf{X}^t)$$

- $S$：歷史序列長度
- $D$：每個時間步的特徵維度
- $\mathbf{E}^t = [\mathbf{e}_1^t, \cdots, \mathbf{e}_N^t]^T \in \mathbb{R}^{N \times U}$：所有股票的序列嵌入矩陣
- $N$：股票數量
- $U$：LSTM 隱藏單元數

```python
# training/rank_lstm.py
lstm_cell = tf.contrib.rnn.BasicLSTMCell(
    self.parameters['unit']  # (U,)，LSTM 隱藏單元數量
)

initial_state = lstm_cell.zero_state(self.batch_size,
                                     dtype=tf.float32)  # (N, U)，初始化隱藏狀態為零向量

outputs, _ = tf.nn.dynamic_rnn(
    lstm_cell, feature, dtype=tf.float32,
    initial_state=initial_state  # (N, S, U)，對輸入序列進行 LSTM 運算
)

seq_emb = outputs[:, -1, :]  # (N, U)，取最後時間步的隱藏狀態作為序列嵌入 E^t
```

### 關係嵌入(Relational Embedding)
#### 定義股票關聯圖
在進行關係嵌入計算之前，需要定義關係編碼向量跟關係遮罩矩陣。
* 關係編碼張量 $\mathcal{A} \in \mathbb{R}^{N \times N \times K}$：第 $(i, j)$ 位置儲存股票 $j$ 到股票 $i$ 的多熱編碼關係向量 $\mathbf{a}_{ji} \in \mathbb{R}^K$，值為 0 或 1，表示對應的關係類型是否存在，例如 $\mathbf{a}_{ji} = [1, 0, 1, 0, \cdots]$ 表示股票 $j$ 與 $i$ 之間存在第 1 種和第 3 種關係。

* 關係遮罩矩陣 $\mathbf{M} \in \mathbb{R}^{N \times N}$：用於在 softmax 計算時遮蔽無關係的股票對，若股票 $j$ 與 $i$ 之間存在至少一種關係（$\text{sum}(\mathbf{a}_{ji}) > 0$），則 $M_{ij} = 0$ ; 若股票 $j$ 與 $i$ 之間不存在任何關係（$\text{sum}(\mathbf{a}_{ji}) = 0$），則 $M_{ij} = -10^9$。

```python
# training/load_data.py
def load_relation_data(relation_file):
    relation_encoding = np.load(relation_file)  # (N, N, K)，載入關係編碼張量
    print('relation encoding shape:', relation_encoding.shape)
    rel_shape = [relation_encoding.shape[0], relation_encoding.shape[1]]  # (N, N)
    mask_flags = np.equal(np.zeros(rel_shape, dtype=int),
                          np.sum(relation_encoding, axis=2))  # (N, N)，判斷每對股票是否無任何關係（關係向量總和為 0）
    mask = np.where(mask_flags, np.ones(rel_shape) * -1e9, np.zeros(rel_shape))  # (N, N)，無關係處設為 -1e9，有關係處設為 0
    return relation_encoding, mask
```


#### 定義時間圖卷積
傳統圖卷積使用固定的鄰接矩陣，無法捕捉股票關係強度隨時間變化的特性，而 TGC 透過將序列嵌入納入關係強度函數，來實現動態的關係建模，如公式11：

$$\bar{\mathbf{e}}_i^t = \sum_{\{j | \text{sum}(\mathbf{a}_{ji}) > 0\}} \frac{g(\mathbf{a}_{ji}, \mathbf{e}_i^t, \mathbf{e}_j^t)}{d_j} \mathbf{e}_j^t$$

- $\bar{\mathbf{e}}_i^t$：股票 $i$ 的關係嵌入向量
- $\mathbf{a}_{ji} \in \mathbb{R}^K$：股票 $j$ 到股票 $i$ 的多熱編碼關係向量，每個元素表示對應關係類型是否存在
- $K$：關係類型總數
- $\text{sum}(\mathbf{a}_{ji}) > 0$：表示股票 $j$ 與 $i$ 之間至少存在一種關係
- $g(\cdot)$：關係強度函數，輸出一個純量權重
- $d_j$：正規化因子，關係節點 $j$ 的鄰居數量越多時，它對任一鄰居節點 $i$ 的影響會被稀釋
- $\mathbf{e}_j^t$：股票 $j$ 的序列嵌入向量

在計算股票 $i$ 的關係嵌入之前，要先定義股票 $i$ 的鄰居節點的重要性(注意力分數)，也就是傳播資訊的權重。

#### 注意力機制
在卷積公式中，鄰居節點的重要性(注意力分數)由關係強度函數 $g$ 決定，這篇論文使用並比較了兩種關係強度函數，其中，顯式透過嵌入相似度乘上關係類型重要性衡量節點之間的注意力分數，如公式12：
##### 顯式建模 (Explicit Modeling)
$$g(\mathbf{a}_{ji}, \mathbf{e}_i^t, \mathbf{e}_j^t) = {\mathbf{e}_i^t}^T \mathbf{e}_j^t \times \phi(\mathbf{w}^T \mathbf{a}_{ji} + b)$$

- ${\mathbf{e}_i^t}^T \mathbf{e}_j^t$：兩股票嵌入的內積，衡量當前時間點的相似度
- $\phi$：激活函數（論文使用 leaky ReLU，斜率 0.2）
- $\mathbf{w} \in \mathbb{R}^K$：關係類型的可學習權重向量，每個元素對應一種關係類型的重要性
- $b$：偏置項
- $\phi(\mathbf{w}^T \mathbf{a}_{ji} + b)$：關係重要性分數，為關係向量的加權和經過非線性變換

內積衡量股票之間的嵌入相似度：

$${\mathbf{e}_i^t}^T \mathbf{e}_j^t = \sum_{u=1}^{U} e_{i,u}^t \cdot e_{j,u}^t = e_{i,1}^t \cdot e_{j,1}^t + e_{i,2}^t \cdot e_{j,2}^t + \cdots + e_{i,U}^t \cdot e_{j,U}^t$$
* $\mathbf{e}_i^t = [e_{i,1}^t, e_{i,2}^t, \cdots, e_{i,U}^t]$
* $\mathbf{e}_j^t = [e_{j,1}^t, e_{j,2}^t, \cdots, e_{j,U}^t]$

如果套用到所有股票，可以寫成公式：

$$\mathbf{E} \times \mathbf{E}^T = (N, U) \times (U, N) = (N, N)$$

在這個矩陣的計算結果中，第 $(i, j)$ 元素正好是 $\mathbf{e}_i^T \mathbf{e}_j$，即股票 $i$ 與 $j$ 的嵌入內積
```python
# training/relation_rank_lstm.py (顯式建模分支)
if self.inner_prod:
    print('inner product weight')
    inner_weight = tf.matmul(feature, feature, transpose_b=True)  # (N, U) × (U, N) = (N, N)，計算所有股票對的嵌入內積 e_i^T e_j
    weight = tf.multiply(inner_weight, rel_weight[:, :, -1])  # (N, N)，逐元素相乘：相似度 × 關係重要性
```


說完顯式接著介紹隱式，它透過通用關係強度函數的輸出衡量節點之間的注意力分數，如公式13：
##### 隱式建模 (Implicit Modeling)
$$g(\mathbf{a}_{ji}, \mathbf{e}_i^t, \mathbf{e}_j^t) = \phi(\mathbf{w}^T [{\mathbf{e}_i^t}^T, {\mathbf{e}_j^t}^T, \mathbf{a}_{ji}^T]^T + b)$$

- $[{\mathbf{e}_i^t}^T, {\mathbf{e}_j^t}^T, \mathbf{a}_{ji}^T]^T \in \mathbb{R}^{2U+K}$：將兩股票嵌入與關係向量串接成一個向量
- $\mathbf{w} \in \mathbb{R}^{2U+K}$：全連接層的可學習權重向量
- $b$：偏置項

如果展開可以寫成：
$$\mathbf{w}^T \begin{bmatrix} \mathbf{e}_i^t \\ \mathbf{e}_j^t \\ \mathbf{a}_{ji} \end{bmatrix} = \sum_{u=1}^{U} w_{1,u} e_{i,u}^t + \sum_{u=1}^{U} w_{2,u} e_{j,u}^t + \sum_{k=1}^{K} w_{3,k} a_{ji,k}$$

- $e_{i,u}^t$：股票 $i$ 嵌入向量的第 $u$ 個分量
- $e_{j,u}^t$：股票 $j$ 嵌入向量的第 $u$ 個分量
- $a_{ji,k}$：關係向量的第 $k$ 個分量（0 或 1，表示第 $k$ 種關係是否存在）


由展開式我們可以看到，隱式建模學習的是三個面向通用的關係強度(所有股票對共享 $\mathbf{w} \in \mathbb{R}^{2U+K}$)，而非特定股票對之間的重要性：
1. $[{w_1, w_2, \cdots, w_U}]$：**目標股票的哪些狀態特徵使其容易受影響**，學習嵌入向量中的哪些維度特徵會使一支股票更容易接收來自其他股票的資訊。例如，模型可能學到「當某個代表波動性的維度值較高時，該股票對外部資訊更敏感」。這個規則適用於所有股票。
2. $[{w_{U+1}, w_{U+2}, \cdots, w_{2U}}]$：**來源股票的哪些狀態特徵使其具有影響力**，學習嵌入向量中的哪些維度特徵會使一支股票對其他股票產生較強的影響。例如，模型可能學到「當某個代表市場領導地位的維度值較高時，該股票對其他股票的影響較大」。這個規則同樣適用於所有股票。
3. $[{w_{2U+1}, w_{2U+2}, \cdots, w_{2U+K}}]$：**哪些關係類型更重要**，學習不同關係類型（如「同產業」、「供應商-客戶」等）對股票預測的通用重要性。

```python
# training/relation_rank_lstm.py
relation = tf.constant(self.rel_encoding, dtype=tf.float32)  # (N, N, K)，股票關係的多熱編碼張量 A
rel_mask = tf.constant(self.rel_mask, dtype=tf.float32)  # (N, N)，關係遮罩矩陣，無關係處為 -1e9

rel_weight = tf.layers.dense(relation, units=1,
                             activation=leaky_relu)  # (N, N, 1)，對每對股票的關係向量做 φ(w^T a_ji + b)，學習關係類型重要性
...

else:
    print('sum weight')
    head_weight = tf.layers.dense(feature, units=1,
                                  activation=leaky_relu)  # (N, 1)，對每支股票的嵌入計算一個純量分數
    tail_weight = tf.layers.dense(feature, units=1,
                                  activation=leaky_relu)  # (N, 1)，使用另一組參數計算另一個純量分數
    weight = tf.add(
        tf.add(
            tf.matmul(head_weight, all_one, transpose_b=True),  # (N, 1) × (1, N) = (N, N)，將 head 分數廣播到每一列
            tf.matmul(all_one, tail_weight, transpose_b=True)  # (N, 1) × (1, N) = (N, N)，將 tail 分數廣播到每一行
        ), rel_weight[:, :, -1]  # (N, N)，加上關係權重矩陣
    )
```
**遮罩與 Softmax 正規化**：
因為注意力分數矩陣內有些股票對之間沒有關聯，因此論文採用遮罩以及正規化將這些股票之間的注意力分數趨近於0。
```python
# training/relation_rank_lstm.py
weight_masked = tf.nn.softmax(tf.add(rel_mask, weight), dim=0)  # (N, N)，套用遮罩後沿 dim=0 做 softmax
```

`tf.add(rel_mask, weight)` 的遮罩機制：
有關係的位置`rel_mask` 的值為 0，無關係的位置為 $-10^9$，在與計算出的注意力分數相加後，無關係的位置變成極大負數，經過 softmax 後，極大負數對應的機率趨近於 0，這樣確保只有存在關係的股票對才會有非零的傳播權重。

#### 嵌入傳播計算
有了各節點之間的注意力分數，就可以進行兩個二維張量的標準矩陣乘法，計算出的每個節點的關係嵌入是其所有鄰居股票嵌入的加權和，如股票 $i$ 的關係嵌入向量的計算方式：

$$\bar{\mathbf{e}}_i^t = \sum_{j=1}^{N} W_{ij} \cdot \mathbf{e}_j^t = [\sum_{j=1}^{N} W_{ij} \cdot e_{j,1}^t, \sum_{j=1}^{N} W_{ij} \cdot e_{j,2}^t, \cdots, \sum_{j=1}^{N} W_{ij} \cdot e_{j,U}^t]$$
- $W_{ij}$：softmax 正規化後的權重，表示股票 $j$ 對股票 $i$ 的影響權重
- $e_{j,u}^t$：股票 $j$ 的序列嵌入向量的第 $u$ 個分量

如果套用到所有股票，可以寫成：
$$\text{outputs} = \mathbf{W} \times \mathbf{E}$$
* $\mathbf{W} \in \mathbb{R}^{N \times N}$：正規化後的傳播權重矩陣
* $\mathbf{E} \in \mathbb{R}^{N \times U}$：所有股票的序列嵌入矩陣

結果會是 $(N, U)$，其中，第 $i$ 列 $\sum_j \text{weight}_{ij} \cdot \mathbf{e}_j$ 就是股票 $i$ 的關係嵌入。

```python
# training/relation_rank_lstm.py
outputs_proped = tf.matmul(weight_masked, feature)  # (N, N) × (N, U) = (N, U)，加權聚合得到關係嵌入
```


### 預測股價(Prediction)

將序列嵌入與關係嵌入串接後，輸入全連接層得到預測的排序分數。

```python
# training/relation_rank_lstm.py
if self.flat:
    print('one more hidden layer')
    outputs_concated = tf.layers.dense(
        tf.concat([feature, outputs_proped], axis=1),  # (N, 2U)，串接序列嵌入與關係嵌入
        units=self.parameters['unit'], activation=leaky_relu,
        kernel_initializer=tf.glorot_uniform_initializer()  # (N, U)，額外的隱藏層變換
    )
else:
    outputs_concated = tf.concat([feature, outputs_proped], axis=1)  # (N, 2U)，直接串接

prediction = tf.layers.dense(
    outputs_concated, units=1, activation=leaky_relu, name='reg_fc',
    kernel_initializer=tf.glorot_uniform_initializer()  # (N, 1)，輸出預測價格
)

return_ratio = tf.div(tf.subtract(prediction, base_price), base_price)  # (N, 1)，計算預測報酬率 r̂^{t+1}
```

### 損失函數(Loss Function)

損失函數結合逐點迴歸損失與成對排序損失：

$$l(\hat{\mathbf{r}}^{t+1}, \mathbf{r}^{t+1}) = \|\hat{\mathbf{r}}^{t+1} - \mathbf{r}^{t+1}\|^2 + \alpha \sum_{i=0}^{N} \sum_{j=0}^{N} \max(0, -(\hat{r}_i^{t+1} - \hat{r}_j^{t+1})(r_i^{t+1} - r_j^{t+1}))$$

- $\mathbf{r}^{t+1} = [r_1^{t+1}, \cdots, r_N^{t+1}]$：所有股票的真實報酬率向量
- $\hat{\mathbf{r}}^{t+1} = [\hat{r}_1^{t+1}, \cdots, \hat{r}_N^{t+1}]$：所有股票的預測報酬率向量
- $\|\cdot\|^2$：均方誤差，確保預測值接近真實值
- $\alpha$：平衡兩項損失的超參數
- $(\hat{r}_i^{t+1} - \hat{r}_j^{t+1})$：預測的相對排序
- $(r_i^{t+1} - r_j^{t+1})$：真實的相對排序
- $\max(0, -(\cdot)(\cdot))$：當預測排序與真實排序相反時產生正的懲罰

```python
# training/rank_lstm.py
reg_loss = tf.losses.mean_squared_error(
    ground_truth, return_ratio, weights=mask  # (1,)，計算迴歸損失 ||r̂ - r||^2
)

pre_pw_dif = tf.subtract(
    tf.matmul(return_ratio, all_one, transpose_b=True),
    tf.matmul(all_one, return_ratio, transpose_b=True)  # (N, N)，計算預測報酬率的成對差異矩陣，(i,j) 元素為 r̂_i - r̂_j
)

gt_pw_dif = tf.subtract(
    tf.matmul(all_one, ground_truth, transpose_b=True),
    tf.matmul(ground_truth, all_one, transpose_b=True)  # (N, N)，計算真實報酬率的成對差異矩陣，(i,j) 元素為 r_j - r_i
)

mask_pw = tf.matmul(mask, mask, transpose_b=True)  # (N, N)，成對遮罩矩陣，只有兩支股票都有效時才計算損失

rank_loss = tf.reduce_mean(
    tf.nn.relu(
        tf.multiply(
            tf.multiply(pre_pw_dif, gt_pw_dif),  # (N, N)，當 (r̂_i - r̂_j) 與 (r_j - r_i) 同號時為正，表示排序錯誤
            mask_pw  # (N, N)，套用遮罩
        )
    )  # (1,)，對所有正值取平均，即 max-margin 排序損失
)

loss = reg_loss + tf.cast(self.parameters['alpha'], tf.float32) * rank_loss  # (1,)，總損失 = 迴歸損失 + α × 排序損失
```

### 實驗設定
實驗對象：指數 NASDAQ、NYSE 內的股票，並且保留在 2013/01/02 至 2017/12/08 期間交易日數超過 98% 的交易日、期間內股價從未低於 5 美元（排除仙股）的股票，最終從 NASDAQ 選出 1,026 支，從 NYSE 選出 1,737 支。

特徵：正規化收盤價、5 日移動平均、10 日移動平均、20 日移動平均、30 日移動平均

目標變數：隔日報酬率 $r_i^{t+1} = (p_i^{t+1} - p_i^t) / p_i^t$

訓練/驗證/測試期：
- 訓練期：2013/01/02 - 2015/12/31（756 個交易日）
- 驗證期：2016/01/04 - 2016/12/30（252 個交易日）
- 測試期：2017/01/03 - 2017/12/08（237 個交易日）

股票關係資料：
- 產業關係：NASDAQ 有 112 種產業類型、NYSE 有 130 種產業類型
- Wiki 公司關係：從 Wikidata 提取的一階與二階公司關係，NASDAQ 有 42 種、NYSE 有 32 種

# 研究結果

### 排序方法優於迴歸方法
Rank_LSTM 在兩個市場的 IRR 上都有顯著提升，在 NASDAQ 達到 0.68、在 NYSE 達到 0.56，相較於基準方法有超過 14% 的改善，這驗證了將股票預測問題重新定義為排序任務的有效性，只有 MRR 指標在 NYSE 市場的表現弱於 SFM，這是因為結合逐點與成對損失會在預測絕對值準確度與相對排序準確度之間產生權衡。
![image](/posts/temporal-relational-stock-ranking/table5.png)
![image](/posts/temporal-relational-stock-ranking/figure5.png)
### 股票關係增強預測效果
在 NYSE 市場上，所有考慮股票關係的方法都優於不考慮關係的 Rank_LSTM，特別是 RSR_I 在使用產業關係時達到 1.06 的 IRR，使用 Wiki 關係時達到 0.79 的 IRR，RSR 方法優於 GCN 與 GBR，因為 TGC 能夠動態捕捉關係強度隨時間的變化，而傳統方法使用固定的圖結構，另外，產業關係在 NASDAQ 上的效果不如預期，這可能是因為 NASDAQ 市場較為波動，受短期因素主導，而產業關係反映的是長期相關性。
* MSE (Mean Square Error, 均方誤差)：衡量預測回報率與真實回報率之間數值差距的指標，數值越小代表預測的絕對值越精準 。
* MRR (Mean Reciprocal Rank, 平均倒數排名)：衡量模型將表現最好的股票排在清單首位的準確度，數值越接近 1 代表最強勢股被排在越前面 。
* IRR (Internal Rate of Return, 內部收益率)：衡量根據模型預測進行投資的實際獲利能力，是評估模型交易策略表現最關鍵的收益指標 。
![image](/posts/temporal-relational-stock-ranking/table6.png)
![image](/posts/temporal-relational-stock-ranking/figure6.png)
![image](/posts/temporal-relational-stock-ranking/table7.png)
![image](/posts/temporal-relational-stock-ranking/figure7.png)

### Top1 回測策略的表現理想
論文評估了 Top1、Top5、Top10 三種回測策略，在大多數情況下，Top1 策略的報酬率最高，Top10 最低，這符合預期——當排序準確時，選擇預期報酬最高的股票應能獲得最高實際報酬，與市場指數相比，RSR_I 的 Top1 策略在 NASDAQ 達到 1.19、NYSE 達到 1.06，顯著優於被動投資，然而，與理想策略（事後選出最佳股票）相比仍有差距，顯示股票預測方法仍有很大的改進空間。
![image](/posts/temporal-relational-stock-ranking/figure8.png)
![image](/posts/temporal-relational-stock-ranking/table10.png)






