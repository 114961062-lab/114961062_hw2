"""
comparison.py
Part C：Traditional (Part A) vs Modern AI (Part B) 比較 + 額外加分項目

輸出檔案（基本要求）：
1. results/tfidf_similarity_matrix.png
2. results/classification_results.csv
3. results/summarization_comparison.txt
4. results/performance_metrics.json

額外加分輸出：
5. results/wordcloud_simple.png                # 詞頻視覺化（柱狀圖）
6. results/word2vec_similarity.txt            # Word2Vec 相似度比較
7. results/large_text_performance.txt         # 大量文本效能測試
8. results/ai_keyword_expansion.txt           # AI 產生行政法關鍵詞
"""

import os
import csv
import json
import time
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer

# 可選：Word2Vec，若未安裝 gensim 不會中斷程式
try:
    from gensim.models import Word2Vec
    HAS_GENSIM = True
except ImportError:
    HAS_GENSIM = False

from openai import OpenAI

from traditional_methods import (
    documents,                         # 六則法律句子 (情感/主題測試)
    sklearn_tfidf_similarity,
    RuleBasedSentimentClassifier,
    TopicClassifier,
    StatisticalSummarizer,
    article                            # A-3 測試文章
)

from modern_methods import (
    ai_similarity,
    ai_classify,
    ai_summarize,
)


# ============================================================
# 準備環境
# ============================================================

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

texts = documents  # 使用 Part A 的六則文本


# ============================================================
# 1. TF-IDF 相似度矩陣圖（基本要求）
# ============================================================

def produce_similarity_png():
    _, sim_matrix = sklearn_tfidf_similarity(texts)

    plt.figure(figsize=(6, 5))
    plt.imshow(sim_matrix, interpolation="nearest", cmap="viridis")
    plt.title("TF-IDF Cosine Similarity (Traditional)")
    plt.colorbar()
    plt.xticks(range(len(texts)), range(1, len(texts) + 1))
    plt.yticks(range(len(texts)), range(1, len(texts) + 1))
    plt.tight_layout()

    output_path = os.path.join(RESULT_DIR, "tfidf_similarity_matrix.png")
    plt.savefig(output_path)
    plt.close()

    print(f"[OK] 已產生：{output_path}")


# ============================================================
# 2. 分類比較 CSV（基本要求）
# ============================================================

def produce_classification_csv(api_key=None):
    sentiment_clf = RuleBasedSentimentClassifier()
    topic_clf = TopicClassifier()

    path = os.path.join(RESULT_DIR, "classification_results.csv")

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id",
            "text",
            "rule_sentiment",
            "ai_sentiment",
            "rule_topic",
            "ai_topic",
            "ai_confidence"
        ])

        for i, text in enumerate(texts, start=1):
            # Traditional
            rule_sent = sentiment_clf.classify(text)
            rule_topic = topic_clf.classify(text)

            # AI
            try:
                res = ai_classify(text, api_key=api_key)
                ai_sent = res["sentiment"]
                ai_topic = res["topic"]
                ai_conf = res["confidence"]
            except Exception as e:
                print(f"[警告] AI 分類失敗（第 {i} 筆）：{e}")
                ai_sent, ai_topic, ai_conf = ("錯誤", "錯誤", 0.0)

            writer.writerow([
                i,
                text,
                rule_sent,
                ai_sent,
                "|".join(rule_topic),
                ai_topic,
                ai_conf
            ])

    print(f"[OK] 已產生：{path}")


# ============================================================
# 3. 摘要比較 TXT（基本要求）
# ============================================================

def produce_summarization_txt(api_key=None):
    summarizer = StatisticalSummarizer()
    path = os.path.join(RESULT_DIR, "summarization_comparison.txt")

    with open(path, "w", encoding="utf-8") as f:
        f.write("=== 原始文章 ===\n")
        f.write(article.strip() + "\n\n")

        # Traditional summary
        trad_summary = summarizer.summarize(article, ratio=0.3)
        f.write("=== 傳統方法摘要 ===\n")
        f.write(trad_summary + "\n\n")

        # AI summary
        try:
            ai_summary = ai_summarize(article, api_key=api_key, max_words=80)
        except Exception as e:
            ai_summary = f"(AI 摘要失敗：{e})"

        f.write("=== AI 摘要（GPT） ===\n")
        f.write(ai_summary + "\n")

    print(f"[OK] 已產生：{path}")


# ============================================================
# 4. 效能比較 JSON（基本要求）
# ============================================================

def produce_performance_json(api_key=None):
    metrics = {}

    # ---- 傳統 TF-IDF ----
    start = time.perf_counter()
    _, sim = sklearn_tfidf_similarity(texts)
    end = time.perf_counter()
    metrics["similarity_traditional"] = {
        "time_sec": end - start,
        "avg_similarity": float(sim.mean())
    }

    # ---- AI 相似度 ----
    try:
        base = texts[0]
        scores = []
        start = time.perf_counter()
        for t in texts[1:]:
            scores.append(ai_similarity(base, t, api_key=api_key))
        end = time.perf_counter()

        avg_sim = sum(scores) / len(scores) if scores else 0.0
        metrics["similarity_ai"] = {
            "time_sec": end - start,
            "avg_similarity": avg_sim
        }
    except Exception as e:
        metrics["similarity_ai"] = {
            "time_sec": None,
            "avg_similarity": None,
            "error": str(e)
        }

    # ---- 傳統分類 ----
    s_clf = RuleBasedSentimentClassifier()
    t_clf = TopicClassifier()
    start = time.perf_counter()
    for t in texts:
        s_clf.classify(t)
        t_clf.classify(t)
    end = time.perf_counter()
    metrics["classification_traditional"] = {"time_sec": end - start}

    # ---- 傳統摘要 ----
    summarizer = StatisticalSummarizer()
    start = time.perf_counter()
    summarizer.summarize(article)
    end = time.perf_counter()
    metrics["summarization_traditional"] = {"time_sec": end - start}

    # ---- AI 摘要 ----
    try:
        start = time.perf_counter()
        ai_summarize(article, api_key=api_key, max_words=80)
        end = time.perf_counter()
        metrics["summarization_ai"] = {"time_sec": end - start}
    except Exception as e:
        metrics["summarization_ai"] = {"time_sec": None, "error": str(e)}

    path = os.path.join(RESULT_DIR, "performance_metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"[OK] 已產生：{path}")


# ============================================================
# ★ 加分 1：詞頻視覺化（替代詞雲）
# ============================================================

def produce_simple_wordcloud():
    """
    使用 jieba 分詞 + matplotlib 畫出前 30 高頻詞的水平長條圖，
    作為「詞雲視覺化」的簡化替代方案。
    """
    all_words = []
    for text in documents:
        all_words.extend(list(jieba.cut(text)))

    counter = Counter(all_words)
    common = counter.most_common(30)

    labels = [w for w, _ in common]
    values = [v for _, v in common]

    plt.figure(figsize=(10, 6))
    plt.barh(labels, values)
    plt.gca().invert_yaxis()
    plt.title("詞頻視覺化（Top 30）")
    plt.xlabel("Frequency")
    plt.tight_layout()

    path = os.path.join(RESULT_DIR, "wordcloud_simple.png")
    plt.savefig(path)
    plt.close()

    print(f"[OK] 已產生：{path}")


# ============================================================
# ★ 加分 2：Word2Vec 相似度比較
# ============================================================

def produce_word2vec_similarity():
    """
    利用 gensim Word2Vec 在小語料上訓練詞向量，
    示範與 TF-IDF 不同的向量表示方式。
    若未安裝 gensim，則產出說明文字。
    """
    path = os.path.join(RESULT_DIR, "word2vec_similarity.txt")

    if not HAS_GENSIM:
        with open(path, "w", encoding="utf-8") as f:
            f.write("未安裝 gensim，無法執行 Word2Vec 測試。\n")
        print(f"[警告] 未安裝 gensim，已在 {path} 寫入說明。")
        return

    # 建立簡單語料
    corpus = [list(jieba.cut(t)) for t in documents]
    model = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=1, workers=4)

    # 取第一句與第二句的平均向量，計算 Cosine 相似度
    def avg_vec(tokens):
        vecs = [model.wv[w] for w in tokens if w in model.wv]
        if not vecs:
            return np.zeros(model.vector_size)
        return np.mean(vecs, axis=0)

    v1 = avg_vec(corpus[0])
    v2 = avg_vec(corpus[1])

    # cosine similarity
    sim = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))

    with open(path, "w", encoding="utf-8") as f:
        f.write("Word2Vec 平均向量相似度 (句1 vs 句2)：\n")
        f.write(f"{sim:.4f}\n")

    print(f"[OK] 已產生：{path}")


# ============================================================
# ★ 加分 3：大量文本效能測試
# ============================================================

def performance_large_text_test():
    """
    模擬大量法律句子（例如 20,000 句）進行 TF-IDF 向量化，
    測量耗時，作為效能優化示範。
    """
    fake_docs = ["本院認為行政處分違法，應予撤銷。" for _ in range(20000)]

    start = time.perf_counter()
    vectorizer = TfidfVectorizer(min_df=2)  # 使用 min_df 作為簡單降維優化
    vectorizer.fit_transform(fake_docs)
    end = time.perf_counter()

    cost = end - start
    path = os.path.join(RESULT_DIR, "large_text_performance.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("處理 20,000 句文本之 TF-IDF 向量化效能測試：\n")
        f.write(f"耗時：約 {cost:.4f} 秒\n")

    print(f"[OK] 已產生：{path}")


# ============================================================
# ★ 加分 4：AI 產生法律關鍵詞（創新應用）
# ============================================================

def produce_ai_keyword_expansion(api_key=None, topic="行政法"):
    """
    呼叫 GPT-3.5，請模型產生指定領域的 20 個法律關鍵詞，
    作為「關鍵詞擴展」示範，對應 LegalTech 實務。
    """
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        path = os.path.join(RESULT_DIR, "ai_keyword_expansion.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("未設定 OPENAI_API_KEY，無法產生 AI 關鍵詞。\n")
        print(f"[警告] 未設定 OPENAI_API_KEY，已在 {path} 寫入說明。")
        return

    client = OpenAI(api_key=key)

    prompt = (
        f"請列出 20 個屬於「{topic}」領域的重要法律關鍵詞，"
        "使用條列式列出，每行一個詞。"
    )

    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=400,
    )

    content = resp.choices[0].message.content

    path = os.path.join(RESULT_DIR, "ai_keyword_expansion.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"[OK] 已產生：{path}")


# ============================================================
# 主程式入口
# ============================================================

if __name__ == "__main__":
    print("\n=== Part C：Comparisons + Bonus ===\n")

    # 若已設定環境變數 OPENAI_API_KEY，這裡保持 None 即可
    API_KEY = None

    # ---- 作業基本要求 ----
    produce_similarity_png()
    produce_classification_csv(api_key=API_KEY)
    produce_summarization_txt(api_key=API_KEY)
    produce_performance_json(api_key=API_KEY)

    # ---- 額外加分項目 ----
    produce_simple_wordcloud()
    produce_word2vec_similarity()
    performance_large_text_test()
    produce_ai_keyword_expansion(api_key=API_KEY, topic="行政法")

    print("\n全部結果已產生在 results/ 資料夾。\n")
