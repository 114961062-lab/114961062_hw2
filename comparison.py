"""
comparison.py
Part C：Traditional (Part A) vs Modern AI (Part B) 比較
會輸出四個檔案到 results/：
1. tfidf_similarity_matrix.png
2. classification_results.csv
3. summarization_comparison.txt
4. performance_metrics.json
"""

import os
import csv
import json
import time
import matplotlib.pyplot as plt
import numpy as np

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
# 1. 相似度矩陣圖 (TF-IDF)
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
# 2. 分類比較 CSV：Traditional vs AI
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
# 3. 摘要比較 TXT
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
            ai_summary = ai_summarize(article, max_words=80)
        except Exception as e:
            ai_summary = f"(AI 摘要失敗：{e})"

        f.write("=== AI 摘要（GPT） ===\n")
        f.write(ai_summary + "\n")

    print(f"[OK] 已產生：{path}")


# ============================================================
# 4. 效能比較 JSON
# ============================================================

def produce_performance_json(api_key=None):
    metrics = {}

    # ---- TF-IDF 傳統方法 ----
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
# 主程式入口
# ============================================================

if __name__ == "__main__":
    print("\n=== Part C：Comparisons ===\n")

    API_KEY = None   # 若你已設定環境變數 OPENAI_API_KEY 就保持 None

    produce_similarity_png()
    produce_classification_csv(api_key=API_KEY)
    produce_summarization_txt(api_key=API_KEY)
    produce_performance_json(api_key=API_KEY)

    print("\n全部結果已產生在 results/ 資料夾。\n")
