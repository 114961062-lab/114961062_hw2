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


def run_tfidf_task():
    texts = [
        "人工智慧正在改變世界，理解學習是基本的技術。",
        "如果要掌握好人工智慧的發展，特別是在機器學習領域。",
        "今天天氣很好，適合出去運動。",
        "機器學習和深度學習是人工智慧的重要分支。",
        "運動有益健康，每天都應保持適當運動習慣。"
    ]

    sim_manual = similarity_manual(texts)
    pd.DataFrame(sim_manual).to_csv("results/tfidf_similarity_manual.csv", index=False)

    sim_sklearn = similarity_sklearn(texts)
    pd.DataFrame(sim_sklearn).to_csv("results/tfidf_similarity_sklearn.csv", index=False)

    print("A-1 完成，已輸出到 results/")


if __name__ == "__main__":
    run_tfidf_task()
