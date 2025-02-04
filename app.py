import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import openai
from azure.core.credentials import AzureKeyCredential
import azure.cognitiveservices.speech as speechsdk
from azure.ai.textanalytics import TextAnalyticsClient
from azure.cognitiveservices.speech import SpeechConfig, SpeechRecognizer

# Azure Text Analytics の設定

# Azure Speech SDK の設定

# --- Azure OpenAI の設定（openaiライブラリを使用） ---

# 埋め込みに利用する deployment 名（例："text-embedding-ada-002"）
deployment_id = "text-embedding-ada-002"

print("回答を音声で入力してください...")

result = speech_recognizer.recognize_once_async().get()

if result.reason == speechsdk.ResultReason.RecognizedSpeech:
    full_text = result.text
    print("認識されたテキスト: {}".format(full_text))
else:
    print("音声を認識できませんでした。もう一度試してください。")
    exit()



def get_embedding(text):
    """Azure OpenAI の埋め込み機能を使ってテキストのベクトルを取得する"""
    try:
        response = openai.Embedding.create(
            engine=deployment_id,
            input=text
        )
        return np.array(response["data"][0]["embedding"])
    except Exception as e:
        print(f"get_embedding 関数でエラーが発生しました: {e}")
        return None

def find_best_match_azure(segment, questions, threshold=0.8):
    """
    Azure OpenAI の埋め込みを利用して、セグメントと各質問候補の
    コサイン類似度を計算し、最も類似度の高い質問を返す。
    閾値未満の場合は None を返す。
    """
    if not segment.strip():
        return None

    try:
        # セグメントと質問候補それぞれの埋め込みを取得
        segment_embedding = get_embedding(segment)
        if segment_embedding is None:
            print(f"セグメントの埋め込み取得に失敗しました: {segment}")
            return None

        question_embeddings = []
        for q in questions:
            qe = get_embedding(q)
            if qe is None:
                print(f"質問の埋め込み取得に失敗しました: {q}")
                return None
            question_embeddings.append(qe)

        # 各質問候補とのコサイン類似度を計算
        sims = [cosine_similarity([segment_embedding], [qe])[0][0] for qe in question_embeddings]
        best_idx = int(np.argmax(sims))
        best_sim = sims[best_idx]

        if best_sim >= threshold:
            return questions[best_idx]
        else:
            return None
    except Exception as e:
        print(f"find_best_match_azure 関数でエラーが発生しました: {e}")
        return None

# --- 質問候補の設定 ---
possible_questions = [
    "今日の天気は",
    "好きな色は？"
    "今の気分は？",
]

# --- テキスト分割（セグメントの改善を検討する余地あり） ---
try:
    segments = re.split(r'[、。！？\n]', full_text)
except Exception as e:
    print(f"テキスト分割中にエラーが発生しました: {e}")
    segments = []

# --- 各セグメントに対して最適な質問を推定 ---
SIMILARITY_THRESHOLD = 0.2  # 埋め込みの場合は0.8前後が一般的（要調整）
matched_results = []

for seg in segments:
    seg = seg.strip()
    if not seg:
        continue
    try:
        matched_question = find_best_match_azure(seg, possible_questions, threshold=SIMILARITY_THRESHOLD)
        if matched_question is not None:
            matched_results.append((seg, matched_question))
        else:
            print(f"無関係と思われる文章（無視）: {seg}")
    except Exception as e:
        print(f"セグメント '{seg}' の処理中にエラーが発生しました: {e}")

# --- 結果の表示 ---
if matched_results:
    for answer_segment, question in matched_results:
        print(f"【回答の一部】{answer_segment}")
        print(f"【推定された質問】{question}")
        print("-" * 40)
else:
    print("有効な回答が検出されませんでした。")
