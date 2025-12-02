# ============================================================
# traditional_methods.py
# Part A：傳統方法 — TF-IDF、規則式分類、統計式摘要
# ============================================================

import math
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# A-1：TF 手動計算
# ============================================================

def calculate_tf(word_dict, total_words):
    """
    手動計算 TF
    Args:
        word_dict: 詞頻字典，例如 {"契": 2, "約": 3}
        total_words: 文本總詞數
    Returns:
        tf_dict: {word: tf}
    """
    tf_dict = {}
    for word, count in word_dict.items():
        tf_dict[word] = count / total_words if total_words > 0 else 0.0
    return tf_dict


def calculate_idf(documents, word):
    """
    手動計算 IDF
    Args:
        documents: 文本列表，每一個元素是一段文字
        word: 目標字詞（此範例以單一字元為 token）
    Returns:
        idf_value: IDF 數值
    """
    N = len(documents)
    containing = sum(1 for doc in documents if word in doc)
    # (N + 1) / (containing + 1) 避免除以 0
    return math.log((N + 1) / (containing + 1)) + 1


# ============================================================
# 測試資料：六則具有不同情緒傾向的法律文本
# （前五則如前，外加一則刻意設計為中性）
# ============================================================

documents = [
    # 正面
    "若買賣契約經雙方善意履行並遵守誠信義務，通常能促成交易順利完成，帶來正面法律效果。",
    # 負面
    "行政處分若嚴重違反比例原則，不僅屬違法，更可能造成對人民權利的重大侵害，後果相當負面。",
    # 負面
    "行為人因重大過失造成他人損害，依法仍須承擔刑事責任，此情形通常被視為極不當的法律行為。",
    # 正面
    "憲法保障人民言論自由，有助於促進民主社會之發展，雖仍需配合法律之必要限制，但整體效果多屬正面。",
    # 正面
    "人民在行政訴訟中得依法聲請停止執行，以確保自身權益能受到妥善保護，此制度具積極的保障作用。",
    # 中性（沒有刻意放入正負面情緒字）
    "本案中，法院僅就事實與法律適用進行審理，最後作成本於既有判例與學說之判決。"
]


# ============================================================
# A-1：手動計算 TF-IDF 矩陣與相似度
# ============================================================

def manual_tfidf(documents):
    """
    使用 calculate_tf 與 calculate_idf 手動計算 TF-IDF 矩陣
    此處為示範，採「字元」為 token（中文未做斷詞）
    Returns:
        tfidf_matrix: shape = (num_docs, vocab_size) 的 2D list
        vocab: 字彙表（list）
    """
    # 建立字彙表（簡化版：所有出現過的字元）
    vocab = sorted(set(ch for doc in documents for ch in doc if ch.strip()))
    # 每篇文件的詞頻
    doc_word_counts = []
    for doc in documents:
        counts = {}
        for ch in doc:
            if not ch.strip():
                continue
            counts[ch] = counts.get(ch, 0) + 1
        doc_word_counts.append(counts)

    # 計算每篇的 TF
    tf_list = []
    for counts in doc_word_counts:
        total = sum(counts.values())
        tf_list.append(calculate_tf(counts, total))

    # 計算每個詞的 IDF
    idf_dict = {}
    for word in vocab:
        idf_dict[word] = calculate_idf(documents, word)

    # 組成 TF-IDF 矩陣
    tfidf_matrix = []
    for tf in tf_list:
        row = []
        for word in vocab:
            row.append(tf.get(word, 0.0) * idf_dict[word])
        tfidf_matrix.append(row)

    return tfidf_matrix, vocab


# ============================================================
# A-1：使用 scikit-learn 計算 TF-IDF 與相似度
# ============================================================

def sklearn_tfidf_similarity(texts):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    return tfidf_matrix, similarity_matrix


# ============================================================
# A-2：規則式文本分類
# ============================================================

class RuleBasedSentimentClassifier:
    """
    簡易情感分類：
    - 看到正面詞就加分
    - 看到負面詞就扣分
    - 不再做「否定詞反轉」處理，避免中文語境誤判（例如「不僅」）
    """

    def __init__(self):
        # 關鍵詞設計，盡量對應上面 documents 中的用詞
        self.positive_words = [
            "善意", "順利", "正面", "有助於", "保障", "妥善", "積極", "發展", "圓滿"
        ]
        self.negative_words = [
            "違法", "侵害", "負面", "嚴重", "重大過失", "損害", "不當", "違反"
        ]

    def classify(self, text: str) -> str:
        score = 0

        # 以「關鍵詞是否出現在句中」為準
        for w in self.positive_words:
            if w in text:
                score += 1

        for w in self.negative_words:
            if w in text:
                score -= 1

        if score > 0:
            return "正面"
        elif score < 0:
            return "負面"
        else:
            return "中性"


class TopicClassifier:
    """
    主題分類：
    依據關鍵詞判斷主要法律領域
    """

    def __init__(self):
        self.topic_keywords = {
            "民法": ["買賣", "契約", "債務", "誠信義務"],
            "刑法": ["刑事責任", "重大過失", "犯罪", "不當的法律行為"],
            "行政法": ["行政處分", "比例原則", "行政機關"],
            "憲法": ["憲法", "言論自由", "基本權", "民主社會"],
            "訴訟法": ["行政訴訟", "停止執行", "訴訟", "程序", "法院", "判決"],
        }

    def classify(self, text: str):
        result = []
        for topic, keywords in self.topic_keywords.items():
            if any(k in text for k in keywords):
                result.append(topic)

        return result if result else ["未分類"]


# 測試資料：同上六則文本
test_texts = documents.copy()


# ============================================================
# A-3：統計式自動摘要
# ============================================================

class StatisticalSummarizer:
    def __init__(self):
        # 簡易停用字，可依需要擴充
        self.stop_words = set(["的", "了", "在", "是", "並", "其", "於", "可"])

    def sentence_score(self, sentence, word_freq):
        """
        計算單一句子的分數：將句中每字的詞頻加總
        """
        score = 0
        for word in sentence:
            if word in word_freq:
                score += word_freq[word]
        return score

    def summarize(self, text, ratio=0.3):
        """
        簡易統計式摘要：
        1. 以「，」切分句子（示範用，可視需要改成更嚴謹切分）
        2. 建立字元詞頻
        3. 對每句計算分數
        4. 取前 ratio 比例的高分句子組成摘要
        """
        sentences = [s for s in text.split("，") if s.strip()]
        words = [ch for ch in text if ch.strip()]

        # 計算詞頻
        word_freq = {}
        for w in words:
            if w not in self.stop_words:
                word_freq[w] = word_freq.get(w, 0) + 1

        # 計算每句分數
        scores = []
        for s in sentences:
            score = self.sentence_score(s, word_freq)
            scores.append((s, score))

        # 由高到低排序
        scores.sort(key=lambda x: x[1], reverse=True)

        # 依比例選句數
        k = max(1, int(len(sentences) * ratio))
        top_sentences = [s for s, _ in scores[:k]]

        summary = "，".join(top_sentences)
        return summary


# 測試文章（法律領域）
article = """
行政處分若違反比例原則，即可能構成違法，人民得依行政訴訟法提起撤銷訴訟。
此外，人民亦可聲請停止執行，以避免處分造成難以回復之損害。
法院審查行政裁量時，將檢驗裁量基礎、目的與適當性，以確保行政行為合乎法治國原則。
憲法保障人民基本權利，行政機關行使權限不得侵害此等權利。
"""


# ============================================================
# 主程式：示範所有功能，方便截圖
# ============================================================

if __name__ == "__main__":
    # ---- A-1 手動 TF-IDF ----
    print("=== A-1：手動 TF-IDF Cosine Similarity ===")
    manual_matrix, vocab = manual_tfidf(documents)
    manual_matrix_np = np.array(manual_matrix)
    manual_sim = cosine_similarity(manual_matrix_np)
    print("字彙表 vocab：", "".join(vocab))
    print("手動 TF-IDF 相似度矩陣：")
    print(manual_sim)

    # ---- A-1 scikit-learn TF-IDF ----
    print("\n=== A-1：sklearn TF-IDF Cosine Similarity ===")
    tfidf_matrix, sklearn_sim = sklearn_tfidf_similarity(documents)
    print("sklearn 相似度矩陣：")
    print(sklearn_sim)

    # ---- A-2 規則式情感分類 ----
    print("\n=== A-2：Rule-based Sentiment ===")
    s = RuleBasedSentimentClassifier()
    for t in test_texts:
        print(f"{t} => {s.classify(t)}")

    # ---- A-2 主題分類 ----
    print("\n=== A-2：Topic Classification ===")
    tc = TopicClassifier()
    for t in test_texts:
        print(f"{t} => {tc.classify(t)}")

    # ---- A-3 統計式摘要 ----
    print("\n=== A-3：Statistical Summary ===")
    summ = StatisticalSummarizer()
    print("原始文章：")
    print(article.strip())
    print("\n摘要結果：")
    print(summ.summarize(article, ratio=0.3))
