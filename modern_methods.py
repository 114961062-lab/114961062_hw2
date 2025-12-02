"""
modern_methods.py (OpenAI Python SDK v1.x)
Part B：AI 方法（使用 GPT-3.5-turbo）
"""

import os
import json
from typing import Dict, Any
from openai import OpenAI


# ============================================================
# 初始化 Client
# ============================================================

def init_client(api_key: str = None) -> OpenAI:
    """
    建立新版 OpenAI Client。
    如果未傳 api_key，會從環境變數 OPENAI_API_KEY 讀取。
    """
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("請先設定 OPENAI_API_KEY 或傳入 api_key。")
    return OpenAI(api_key=key)


# ============================================================
# B-1：AI 相似度
# ============================================================

def ai_similarity(text1: str, text2: str, api_key: str = None) -> float:
    client = init_client(api_key)

    system_msg = (
        "你是一位法律文本分析助理。請給出兩段文字之間的語義相似度，"
        "以 0 到 1 表示，請只輸出數字，不要解釋。"
    )

    user_msg = f"文字 A：{text1}\n文字 B：{text2}\n請輸出相似度："

    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
        max_tokens=10,
    )

    content = resp.choices[0].message.content.strip()

    try:
        score = float(content)
    except:
        score = 0.0

    return max(0.0, min(1.0, score))


# ============================================================
# B-2：AI 文本分類
# ============================================================

def ai_classify(text: str, api_key: str = None) -> Dict[str, Any]:
    client = init_client(api_key)

    system_msg = (
        "你是一位法律研究助理，請將輸入文字分類：\n"
        "1. sentiment：正面、負面、中性\n"
        "2. topic：民法、刑法、行政法、憲法、訴訟法、其他\n"
        "3. confidence：0~1\n"
        "務必輸出合法 JSON，不要加入多餘文字。"
    )

    user_msg = f"請分類以下文本：{text}"

    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.1,
        max_tokens=200,
    )

    raw = resp.choices[0].message.content

    # JSON 解析安全處理
    try:
        obj = json.loads(raw)
    except:
        obj = {"sentiment": "中性", "topic": "其他", "confidence": 0.0}

    sentiment = obj.get("sentiment", "中性")
    topic = obj.get("topic", "其他")

    try:
        confidence = float(obj.get("confidence", 0.0))
    except:
        confidence = 0.0

    return {
        "sentiment": sentiment,
        "topic": topic,
        "confidence": confidence,
    }


# ============================================================
# B-3：AI 自動摘要
# ============================================================

def ai_summarize(text: str, api_key: str = None, max_words: int = 80) -> str:
    client = init_client(api_key)

    system_msg = (
        "你是一位法律摘要撰寫助理，請以正式、條理清晰的中文撰寫摘要，"
        "保留法律爭點與結論。"
    )

    user_msg = f"請將以下文本濃縮為約 {max_words} 字的摘要：\n{text}"

    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.3,
        max_tokens=300,
    )

    return resp.choices[0].message.content.strip()


# ============================================================
# 主程式（示範）
# ============================================================

if __name__ == "__main__":
    print("=== Part B：AI 方法示範 ===\n")

    text1 = "行政處分若違反比例原則，不僅屬違法，更可能造成重大侵害。"
    text2 = "若行政行為違背比例原則，法院通常認定為違法處分。"

    article = """
行政處分若違反比例原則，即可能構成違法，人民得依行政訴訟法提起撤銷訴訟。
人民亦可聲請停止執行，以避免處分造成難以回復之損害。
法院審查行政裁量時，將檢驗裁量基礎與目的，以確保行政行為合乎法治國原則。
"""

    # B-1
    try:
        print(">>> B-1 相似度：")
        print(ai_similarity(text1, text2))
    except Exception as e:
        print("相似度錯誤：", e)

    print("\n-------------------------\n")

    # B-2
    try:
        print(">>> B-2 文本分類：")
        print(json.dumps(ai_classify(text1), ensure_ascii=False, indent=2))
    except Exception as e:
        print("分類錯誤：", e)

    print("\n-------------------------\n")

    # B-3
    try:
        print(">>> B-3 AI 摘要：")
        print(ai_summarize(article, max_words=60))
    except Exception as e:
        print("摘要錯誤：", e)
