// research.ts 設定研究經驗的資料來源，含 title、author、advisor、department、date、sections 等欄位
import { Research } from '@/types';

export const research: Research = {
  title: '動態因果網絡於股票報酬預測之建構與應用',
  author: '謝子尉',
  advisor: '邱信瑜',
  department: '國立中央大學 資訊工程學系',
  date: '2025 年 12 月',
  sections: [
    {
      id: 'abstract',
      title: '摘要(暫定)',
      content: `


**Keyword**: GraphSAGE, GRU, Stock Correlation, Portofolio Optimize, Explainable Deep Learning(暫定)
      `.trim()
    },
    {
      id: 'introduction',
      title: 'Introduction',
      content: `
#### 圖神經網絡應用於股票預測的理論基礎

儘管效率市場假說（Efficient Market Hypothesis, EMH）主張股價已完全反映所有可用資訊，價格變動應呈現不可預測的隨機漫步（Fama, 1965），行為金融學的實證研究卻揭示了市場中普遍存在的無效性現象。這些無效性源自投資者的認知偏誤、資訊擴散的時滯效應，以及市場的結構性摩擦（Malkiel, 2003）。上述因素共同形塑了資產間持久的跨期依賴關係，為運用圖方法建模股票關聯結構提供了堅實的經濟學理論基礎。

#### 現有圖結構建構方法的侷限

當前研究主要採用相關性指標（如 Pearson 相關係數、曼哈頓距離）建構股票關聯圖（Xiang et al., 2022; Wu et al., 2022; Qin & Zhan, 2024）。然而，此類方法僅能識別變數間的統計關聯，無法區分直接因果效應與由第三方中介產生的虛假關聯。即便透過圖注意力網絡（GAT）學習邊權重，所得注意力係數本質上仍反映同期共變機率，而非時序上的因果傳導關係。

此外，現有模型多以股價作為節點特徵輸入（Cheng et al., 2021; Feng et al., 2019; Kim et al., 2019），卻未能充分納入市場系統性因子與產業關聯的局部結構。GAT 的注意力機制受限於預定義的靜態鄰域，可能遺漏全局資訊，或因關係結構過時而引入雜訊。

#### 因果推論方法的優勢

針對上述侷限，條件格蘭傑因果指數（Conditional Granger Causality Index, CGCI）提供了更適切的解決方案。相較於傳統相關性指標，CGCI 能在控制其他變數影響的條件下，識別股票間的直接因果關係，有效排除中介效應所導致的虛假因果連結。Billio et al.（2012）已證實格蘭傑因果方法可有效捕捉金融機構間的相互依賴關係；Chen et al.（2024）則將隱式因果探索引入圖神經網絡，但僅採用無條件的格蘭傑因果檢定。本研究進一步採用 CGCI 建構動態因果網絡，期望更精確地刻畫股票報酬間的時序傳導機制。
      `.trim()
    },
    {
      id: 'model',
      title: '研究方法與模型(暫定)',
      content: `
### Methodology
運用統計因果分析方法，結合市場風險因子識別股票間的先後影響關係，以此結果作為圖神經網絡的輸入，並與其他替代結構進行預測與交易績效比較。

      `.trim()
    },
    {
      id: 'results',
      title: '結果分析',
      content: `
### Expected Result
* 分析高出度與高入度股票的特性，檢視後者是否具較高可預測性
* 比較因果圖與其他圖結構在預測準確率及策略績效上的表現
* 檢驗牛熊市期間圖結構密度與連接數的差異
      `.trim()
    },
    {
      id: 'references',
      title: 'References',
      content: `
Billio, M., Getmansky, M., Lo, A. W., & Pelizzon, L. (2012). Econometric measures of connectedness and systemic risk in the finance and insurance sectors. *Journal of Financial Economics*, 104(3), 535-559.

Chen, W., Wang, Z., & Zhang, J. (2024). Implicit-Causality-Exploration-Enabled Graph Neural Network for Stock Prediction. *IEEE Transactions on Neural Networks and Learning Systems*.

Cheng, D., Yang, F., Xiang, S., & Liu, J. (2021). Modeling the Momentum Spillover Effect for Stock Prediction via Attribute-Driven Graph Attention Networks. *Proceedings of the AAAI Conference on Artificial Intelligence*, 35(1), 55-62.

Fama, E. F. (1965). The behavior of stock-market prices. *The Journal of Business*, 38(1), 34-105.

Feng, F., He, X., Wang, X., Luo, C., Liu, Y., & Chua, T. S. (2019). Temporal Relational Ranking for Stock Prediction. *ACM Transactions on Information Systems*, 37(2), 1-30.

Kim, R., So, C. H., Jeong, M., Lee, S., Kim, J., & Kang, J. (2019). HATS: A Hierarchical Graph Attention Network for Stock Movement Prediction. *arXiv preprint arXiv:1908.07999*.

Malkiel, B. G. (2003). The efficient market hypothesis and its critics. *Journal of Economic Perspectives*, 17(1), 59-82.

Qin, Y., & Zhan, Y. (2024). Modeling the stock relation with graph network for overnight stock movement prediction. *Information Sciences*, 660, 120111.

Wu, S., Liu, Y., Zou, Z., & Weng, T. H. (2022). Inductive Representation Learning on Dynamic Stock Co-Movement Graphs for Stock Predictions. *PRICAI 2022: Trends in Artificial Intelligence*, 335-347.

Xiang, S., Cheng, D., Shang, C., Zhang, Y., & Liang, Y. (2022). Temporal and Heterogeneous Graph Neural Network for Financial Time Series Prediction. *Proceedings of the 31st ACM International Conference on Information and Knowledge Management*, 3584-3593.
      `.trim()
    }
  ]
};