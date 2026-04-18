from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import torch
import streamlit as st

# キャッシュを使用してモデルのロードを一度だけにする
@st.cache_resource
def load_finbert():
    try:
        model_name = "ProsusAI/finbert"
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(model_name)
        nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
        return nlp
    except Exception as e:
        print(f"FinBERTのロードに失敗しました: {e}")
        return None

def analyze_sentiment(text_list, use_finbert=True):
    """
    ニュースのリストを受け取り、感情分析スコアを返す
    FinBERT (高性能) または VADER (軽量) を使用
    """
    if not text_list:
        return 0.0, "中立"

    if use_finbert:
        nlp = load_finbert()
        if nlp:
            try:
                # OOMやトークン超過エラーを防ぐため、truncationとmax_lengthを指定
                results = nlp(text_list, truncation=True, max_length=512)
                # FinBERTの結果は [{'label': 'positive', 'score': 0.9}, ...]
                # スコアを数値化 (-1.0 to 1.0)
                scores = []
                for res in results:
                    s = res['score']
                    if res['label'] == 'positive':
                        scores.append(s)
                    elif res['label'] == 'negative':
                        scores.append(-s)
                    else:
                        scores.append(0.0)
                
                avg_score = sum(scores) / len(scores)
                mood = "ポジティブ" if avg_score >= 0.1 else ("ネガティブ" if avg_score <= -0.1 else "中立")
                return avg_score, mood
            except Exception as e:
                print(f"FinBERTの推論中にエラーが発生しました: {e}")
                # 失敗した場合はVADERにフォールバック
                pass

    # VADER (フォールバックまたは軽量モード)
    analyzer = SentimentIntensityAnalyzer()
    total_score = 0
    for text in text_list:
        v_res = analyzer.polarity_scores(text)
        total_score += v_res['compound']
    
    avg_score = total_score / len(text_list)
    mood = "ポジティブ" if avg_score >= 0.05 else ("ネガティブ" if avg_score <= -0.05 else "中立")
    return avg_score, mood

def summarize_news(text_list):
    """
    ニュースのリストを簡潔にまとめる
    (本来はLLMを使用するが、ここでは簡易的な重要文抽出を行うか、将来の拡張用)
    """
    if not text_list:
        return "ニュースはありません。"
    
    # 将来的にGoogle Gemini APIやOpenAI APIを使って要約する場合のプレースホルダー
    # 現状はリストを箇条書きにするだけ
    summary = "\n".join([f"- {t[:100]}..." for t in text_list[:3]])
    return summary
