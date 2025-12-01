import math
import jieba
import numpy as np
import pandas as pd
import re 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def tokenize(text):
    return [w for w in jieba.lcut(text) if w.strip()]

def split_sentences(text: str):
    """
    將一段中文文字依「。！？!?」切成句子
    回傳句子 list，已去掉空白
    """
    if not isinstance(text, str):
        text = str(text)
    # 用中文與英文的句號、問號、驚嘆號斷句
    parts = re.split(r'(?<=[。！？!?])', text)
    sentences = [p.strip() for p in parts if p.strip()]
    return sentences

def summarize_text(text: str, max_sentences: int = 2) -> str:
    """
    對單一篇文章做抽取式摘要：
    - 將文章切成句子
    - 以 TF-IDF 計算每句分數
    - 取分數最高的 max_sentences 句，依原順序組合
    """
    sentences = split_sentences(text)

    # 如果句子數很少，就直接回傳原文
    if len(sentences) <= max_sentences:
        return text.strip()

    # 使用前面已經 import 的 TfidfVectorizer 與 tokenize
    vectorizer = TfidfVectorizer(
        tokenizer=lambda x: tokenize(x),
        lowercase=False
    )
    X = vectorizer.fit_transform(sentences)  # shape: (num_sentences, vocab_size)

    # 每句分數 = 該行 TF-IDF 權重總和
    scores = X.sum(axis=1).A1  # 轉成 1D numpy array

    # 取分數最高的 max_sentences 句
    top_idx = np.argsort(-scores)[:max_sentences]
    # 為了摘要閱讀順序自然，依原來句子順序排序
    top_idx_sorted = sorted(top_idx)

    summary_sentences = [sentences[i] for i in top_idx_sorted]
    summary = "".join(summary_sentences)
    return summary

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

def rule_based_label(text: str) -> str:
    """
    A-2 規則式分類器
    這裡先用範例類別，你要依照作業說明，把類別名稱與關鍵字改掉
    """
    t = text.lower()

    # ======= 類別 1：AI / 機器學習相關 =======
    ai_keywords = ["人工智慧", "機器學習", "深度學習", "ai"]
    if any(k in t for k in ai_keywords):
        return "AI/ML"

    # ======= 類別 2：運動 / 健康 =======
    sport_keywords = ["運動", "健康", "跑步", "健身"]
    if any(k in t for k in sport_keywords):
        return "Sport/Health"

    # ======= 其他：無法歸類到上述 =======
    return "Other"

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

def run_rule_based_A2():
    """
    A-2：規則式分類
    讀取 data/texts.csv，對每筆文本產生一個類別標籤，
    並輸出到 results/rule_based_A2.csv
    """
    csv_path = "data/texts.csv"
    # 你現在 CSV 第一欄叫 'A-1'，如果之後改成 'text' 要記得一起改
    text_column = "text"

    texts = load_texts_from_csv(csv_path, text_column)
    print(f"A-2 載入 {len(texts)} 筆文本資料")

    labels = [rule_based_label(t) for t in texts]

    df_out = pd.DataFrame({
        "text": texts,
        "label": labels,
    })

    output_path = "results/rule_based_A2.csv"
    df_out.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"A-2 完成，已輸出到 {output_path}")

def run_summary_A3():
    """
    A-3：傳統方法的抽取式摘要
    讀取 data/texts.csv，對每筆文本產生摘要，
    輸出到 results/summary_A3.csv
    """
    csv_path = "data/texts.csv"
    # 目前你的 CSV 第一欄欄位名稱是 'A-1'，若之後改成 'text' 要同步修改
    text_column = "A-1"

    texts = load_texts_from_csv(csv_path, text_column)
    print(f"A-3 載入 {len(texts)} 筆文本資料")

    # 每篇文章取 2 句作摘要；若作業有指定句數，請改 max_sentences
    summaries = [summarize_text(t, max_sentences=2) for t in texts]

    df_out = pd.DataFrame({
        "text": texts,
        "summary": summaries,
    })
    output_path = "results/summary_A3.csv"
    df_out.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"A-3 完成，已輸出到 {output_path}")

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
    # A-1：TF-IDF 相似度
    run_tfidf_task()

    # A-2：規則式分類
    run_rule_based_A2()

    # A-3：抽取式摘要
    run_summary_A3()