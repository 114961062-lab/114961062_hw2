import math
import jieba
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def tokenize(text):
    return [w for w in jieba.lcut(text) if w.strip()]


def calculate_tf(word_dict, total_words):
    return {w: count / total_words for w, count in word_dict.items()}


def calculate_idf(documents, word):
    N = len(documents)
    df = sum(1 for doc in documents if word in doc)
    return math.log(N / (df if df else 1))


def manual_tfidf_vectors(texts):
    docs = [tokenize(t) for t in texts]
    vocab = sorted(list(set([w for doc in docs for w in doc])))

    tf_list = []
    for doc in docs:
        word_counts = {}
        for w in doc:
            word_counts[w] = word_counts.get(w, 0) + 1
        tf_list.append(calculate_tf(word_counts, len(doc)))

    idf_dict = {w: calculate_idf(docs, w) for w in vocab}

    tfidf = np.zeros((len(texts), len(vocab)))
    for i, tf in enumerate(tf_list):
        for j, w in enumerate(vocab):
            tfidf[i][j] = tf.get(w, 0) * idf_dict[w]
    return tfidf


def similarity_manual(texts):
    vectors = manual_tfidf_vectors(texts)
    norm = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    v = vectors / norm
    sim = v @ v.T
    return sim


def similarity_sklearn(texts):
    vectorizer = TfidfVectorizer(tokenizer=lambda x: tokenize(x), lowercase=False)
    X = vectorizer.fit_transform(texts)
    sim = cosine_similarity(X, X)
    return sim

def load_texts_from_csv(path, text_column="text"):
    """
    從 CSV 檔讀取文字欄位
    1. 如果有名為 text_column 的欄位，就用它
    2. 如果沒有，就自動改用「第一個欄位」
    """
    df = pd.read_csv(path)

    # 把整欄都是空的先丟掉，避免亂入
    df = df.dropna(how="all", axis=1)

    # 如果指定的欄位不存在，就用第一個欄位當文字欄
    if text_column not in df.columns:
        first_col = df.columns[0]
        print(f"找不到欄位 '{text_column}'，改用第一個欄位 '{first_col}'。")
        text_column = first_col

    # 丟掉這個欄位中為 NaN 的列
    df = df.dropna(subset=[text_column])

    texts = df[text_column].astype(str).tolist()
    return texts

def run_tfidf_task():
    csv_path = "data/texts.csv"
    text_column = "text"

    texts = load_texts_from_csv(csv_path, text_column)
    print(f"載入 {len(texts)} 筆文本資料")

    sim_manual = similarity_manual(texts)
    pd.DataFrame(sim_manual).to_csv("results/tfidf_similarity_manual.csv", index=False)

    sim_sklearn = similarity_sklearn(texts)
    pd.DataFrame(sim_sklearn).to_csv("results/tfidf_similarity_sklearn.csv", index=False)

    print("A-1 完成，已輸出到 results/")

def rule_based_label(text):
    """
    非機器學習的規則式分類器：
    依關鍵字判斷此句屬於哪一類
    你可以依作業要求調整類別與關鍵字
    """
    t = text.lower()

    # 類別 1：人工智慧 / 機器學習相關
    ai_keywords = ["人工智慧", "機器學習", "深度學習", "ai"]
    if any(k in t for k in ai_keywords):
        return "AI/ML"

    # 類別 2：運動 / 健康
    sport_keywords = ["運動", "健康", "習慣"]
    if any(k in t for k in sport_keywords):
        return "Sport/Health"

    # 其他
    return "Other"

def run_rule_based_classification():
    """
    A-2：規則式分類
    讀同一份 CSV，對每一筆文本產生類別標籤，輸出到 results/rule_based_labels.csv
    """
    csv_path = "data/texts.csv"
    text_column = "text"   # 或 "text"，視你實際欄位而定

    texts = load_texts_from_csv(csv_path, text_column)
    labels = [rule_based_label(t) for t in texts]

    df_out = pd.DataFrame({
        "text": texts,
        "label": labels
    })
    df_out.to_csv("results/rule_based_labels.csv", index=False, encoding="utf-8-sig")
    print("A-2 完成，已輸出 results/rule_based_labels.csv")



if __name__ == "__main__":
    run_tfidf_task()
    run_rule_based_classification()  # A-2 規則式分類