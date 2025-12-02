# HW2：傳統 NLP 方法 vs. 現代生成式 AI 方法比較分析

本作業比較三大任務在「傳統方法 (Part A)」與「AI 方法 (Part B)」的差異，並以程式自動輸出結果 (Part C)。

---

# 📦 Part 0 — 環境安裝

本作業需使用 Python 3.9+。

請在終端機執行以下指令安裝所有必要套件：

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jieba stopwordsiso openai tqdm


設定 OpenAI API Key

請先到終端機輸入：setx OPENAI_API_KEY "你的_API_Key"

專案結構
hw2/
│
├── traditional_methods.py        # Part A：TF-IDF、規則式分類、統計式摘要
├── modern_methods.py             # Part B：GPT-3.5-turbo AI 方法（新版 openai SDK）
├── comparison.py                 # Part C：自動比較產生四個輸出檔案
├── comparison_bouns.py           # Part D：加分題
│
├── results/                      # 程式自動生成
│   ├── tfidf_similarity_matrix.png
│   ├── classification_results.csv
│   ├── summarization_comparison.txt
│   └── performance_metrics.json
│
└── README.md # 執行說明
│
└── report.md  # 分析報告(Markdown格式)
│
└── requirements.txt #使用套件版本

Part A — Traditional Methods (傳統方法)

此部分在 traditional_methods.py 實作：

A-1：TF-IDF + Cosine Similarity

使用 手動 TF-IDF 計算（符合作業要求的 calculate_tf、calculate_idf）

使用 scikit-learn 計算 TF-IDF 與相似度

測試資料為 6 則具正負情緒的法律句子

A-2：規則式分類（Rule-based Classification）

情感分類：正面 / 負面 / 中性

主題分類：民法、刑法、行政法、憲法、訴訟法

自訂正負面關鍵字，避免全判成中性

A-3：統計式摘要（Statistical Summarization）

使用簡易字頻打分內容句子

取前 30% 高分句子作為摘要

執行 Part A：
python traditional_methods.py

Part B — Modern AI Methods (GPT-3.5-turbo)

實作於 modern_methods.py，使用 OpenAI Python SDK v1（新版 API）。

B-1：AI 相似度計算

使用 GPT-3.5-turbo 評估兩段文字相似度（0–1）

B-2：AI 文本分類

回傳 JSON：
{
  "sentiment": "正面",
  "topic": "行政法",
  "confidence": 0.87
}

B-3：AI 自動摘要

產生條理清楚的法律文本摘要

可控制大致字數（例如 80 字）

執行 Part B：
python modern_methods.py

Part C — Comparison (comparison.py)

執行 Part C：
python comparison.py
將自動產生以下四個檔案：

1️⃣ results/tfidf_similarity_matrix.png

📌 傳統方法 TF-IDF 的 Cosine Similarity 視覺化矩陣。

你可在報告中比較：

哪些句子因為「字面重複」而相似度較高

與 AI 方法之結果是否一致

2️⃣ results/classification_results.csv

📌 傳統方法 vs AI 的分類結果比較。

包含欄位：

ID	Text	Rule Sentiment	AI Sentiment	Rule Topic	AI Topic	AI Confidence

助教可明確看到兩種方法的差異。

3️⃣ results/summarization_comparison.txt

📌 傳統摘要 vs AI 摘要的直接比較。

內容包含：
=== 原始文章 ===
（完整文章）

=== 傳統方法摘要 ===
（取字頻的摘要）

=== AI 摘要 ===
（GPT-3.5-turbo 產生摘要）

4️⃣ results/performance_metrics.json

📌 效能與速度比較（秒數）

範例：
{
  "similarity_traditional": {
    "time_sec": 0.0021,
    "avg_similarity": 0.1938
  },
  "similarity_ai": {
    "time_sec": 1.842,
    "avg_similarity": 0.692
  }
}

結語

本作業完整比較：

傳統 NLP 方法（可控、快速）

AI 生成式方法（語意理解強）

並能透過 results/ 輸出的四項檔案清楚展示兩者差異。
