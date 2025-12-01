# 114961062 - HW2：Traditional NLP Methods  
國立政治大學｜生成式人工智慧｜作業二  
作者：114961062（立軒）

---

## 📘 作業內容簡介

本專案依照課程要求，實作傳統 NLP 的三項核心任務：

- **A-1：文本相似度（TF-IDF Similarity）**  
- **A-2：規則式分類（Rule-based Classification）**  
- **A-3：抽取式摘要（Extractive Summarization）**

所有程式均以 Python 實作，並完成輸出檔案於 `results/` 資料夾。  
專案遵循「可再現」、「可讀性高」、「具備說明性」三原則實作。

---

# 🗂 專案架構

114961062_hw2/
│
├── traditional_methods.py # A-1 ～ A-3 主程式
├── data/
│ └── texts.csv # 作業用文本資料（5 筆示例）
│
├── results/
│ ├── tfidf_similarity_manual.csv
│ ├── tfidf_similarity_sklearn.csv
│ ├── rule_based_A2.csv
│ └── summary_A3.csv
│
└── README.md # 本檔案


---

# 🔧 環境需求

- Python 3.8+
- 套件：
  - `numpy`
  - `pandas`
  - `jieba`
  - `scikit-learn`

安裝方式：

```bash
pip install numpy pandas jieba scikit-learn
📌 A-1：文本相似度（TF-IDF Similarity）
本部分分為：

✔ 手刻 TF-IDF
使用 jieba 斷詞

計算 TF

計算 IDF

建立 TF-IDF 向量

使用餘弦相似度（Cosine Similarity）求文章間相似度

將結果輸出至：
results/tfidf_similarity_manual.csv

✔ sklearn 版本
使用 TfidfVectorizer

自訂 tokenizer=tokenize，保持中文斷詞一致性

高效產生 TF-IDF 向量

輸出：
results/tfidf_similarity_sklearn.csv

目的：
理解 TF-IDF 的底層運作與文本向量化方式。

📌 A-2：規則式分類（Rule-based Classification）
分類方法不依賴機器學習，而是以「人工定義關鍵字」作為規則。
流程：

從 data/texts.csv 讀取文本

依關鍵字比對決定類別

類別示例（可依作業需求調整）：

AI/ML：人工智慧、機器學習、深度學習

Sport/Health：運動、健康、跑步

Other：不符合以上規則者

輸出至：
results/rule_based_A2.csv

特色：

可解釋性高

簡單、可控

適合資料量小的任務

📌 A-3：抽取式摘要（Extractive Summarization）
採用傳統 NLP 摘要方式：

使用正規表示式切句

對每句做 TF-IDF

以句子的 TF-IDF 權重總和作為「重要度」

選擇權重最高的前 1～2 句當摘要（可依需求調整）

輸出至：
results/summary_A3.csv

摘要結果完全由原句抽取，不生成新句子。

▶️ 執行方式
在專案根目錄執行：

bash
複製程式碼
python traditional_methods.py
程式會依序執行：

A-1：TF-IDF 相似度

A-2：規則式分類

A-3：抽取式摘要

並將所有輸出寫入 results/ 之中。

📝 作者心得（可選）
本作業讓我重新練習 NLP 傳統方法，
包含中文斷詞、TF-IDF 的底層邏輯、餘弦相似度計算、
以及規則式分類與抽取式摘要的完整流程。

透過手刻 TF-IDF，我能更清楚理解文本向量化的重要性；
規則式分類則突顯了可解釋性的重要；
最後的摘要任務，則串接了句子切分、TF-IDF、權重排序等技能。

本專案完整呈現傳統 NLP 的三支主線運算流程。