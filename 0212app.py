import os
import re
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import azure.cognitiveservices.speech as speechsdk
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from azure.cognitiveservices.speech import SpeechConfig, SpeechRecognizer
import json
import time
import threading
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics

# Azure Speech SDK の設定
speech_key = "key"
service_region = "japaneast"
speech_config = SpeechConfig(subscription=speech_key, region=service_region)
speech_recognizer = SpeechRecognizer(speech_config=speech_config, language="ja-JP")

# Azure Text Analytics の設定
credential = AzureKeyCredential("key")
endpoint = "key"
text_analytics_client = TextAnalyticsClient(endpoint=endpoint, credential=credential)

# --- Azure OpenAI の設定（openaiライブラリを使用） ---
openai.api_type = "azure"
openai.api_base = "key/"
openai.api_version = "2023-05-15"  # 利用しているAPIバージョンに合わせる
openai.api_key = "key"

# 埋め込みに利用する deployment 名（例："text-embedding-ada-002"）
deployment_id = "text-embedding-ada-002"
LLM_DEPLOYMENT_NAME = "gpt-4"

# NotoSansJPフォントを登録（同一フォルダ内のファイルを指定）
font_path = os.path.join(os.path.dirname(__file__), "NotoSansJP[wght].ttf")
pdfmetrics.registerFont(TTFont("NotoSansJP", font_path))

#質問のインポート
with open("questions.json", "r", encoding="utf-8") as f:
    questions = json.load(f)

# 音声で入力（"s" が入力されるまで繰り返す）
def speech_to_text_loop():
    recognized_texts = []  # 認識したテキストを保存するリスト
    stop_flag = threading.Event()

    # キーボード入力を監視するスレッドの定義
    def wait_for_stop():
        while not stop_flag.is_set():
            user_input = input("終了するには 's' を入力してください: ").strip().lower()
            if user_input == "s":
                stop_flag.set()

    # キーボード監視スレッドを起動
    stop_thread = threading.Thread(target=wait_for_stop, daemon=True)
    stop_thread.start()

    # 音声認識の開始

    print("音声入力を開始します。話してください。")
    while not stop_flag.is_set():
        # 各認識呼び出しごとに新しい SpeechRecognizer インスタンスを生成する
        temp_recognizer = SpeechRecognizer(speech_config=speech_config, language="ja-JP")
        result = temp_recognizer.recognize_once_async().get()  # 音声認識を実行
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            recognized_texts.append(result.text)
            print(f"認識結果: {result.text}")
        else:
            print("音声を認識できませんでした。もう一度試してください。")
        
        time.sleep(1)  # ループの処理を少し待機することで負荷を軽減
    
    return " ".join(recognized_texts)  # 認識したテキストを連結して返す


# Azure Text Analyticsでテキスト解析（例：キーフレーズ抽出）
def extract_key_phrases(text: str, endpoint: str, key: str):
        text_analytics_client = TextAnalyticsClient(endpoint=endpoint, credential=credential)

        response = text_analytics_client.extract_key_phrases(documents=[text])[0]
        if not response.is_error:
            return response.key_phrases
        else:
            print("キーフレーズ抽出に失敗しました。", response.error)
            return []

    

# --- Embedding取得 ---
def get_embedding(text):
    try:
        response = openai.Embedding.create(
            engine="text-embedding-ada-002",
            input=text
        )
        return np.array(response["data"][0]["embedding"])
    except Exception as e:
        print(f"Embeddingエラー: {e}")
        return None

# --- 質問リストは冒頭で"questions"として定義 ---

# 埋め込みの保存用リスト
question_embeddings = []

for q in questions:
    # 質問 + 選択肢 の埋め込み
    full_text = q["question"] + " " + ", ".join(q["options"])
    full_embedding = get_embedding(full_text)

    # 選択肢のみの埋め込み
    options_text = ", ".join(q["options"])
    options_embedding = get_embedding(options_text)

    # IDと埋め込みを保存
    question_embeddings.append({
        "id": q["id"],
        "question": q["question"],
        # 埋め込みが None でない場合はリストに変換する
        "question_embedding": full_embedding.tolist() if full_embedding is not None else None,
        "options_embedding": options_embedding.tolist() if options_embedding is not None else None
    })

# 埋め込みをファイルに保存 (例: JSON)
with open("question_embeddings.json", "w", encoding="utf-8") as f:
    json.dump(question_embeddings, f, ensure_ascii=False, indent=2)

# 5. Embeddingにより上位候補の質問を抽出
def find_top_candidates(segment):
    """Embeddingでsegmentに類似する候補を抽出し、
    ・「質問＋選択肢」の埋め込みによる上位2件と、
    ・「選択肢のみ」の埋め込みによる上位4件
    をマージした候補リスト（(id, 質問文, score)のタプル）を返す"""
    seg_emb = get_embedding(segment)
    if seg_emb is None:
        return []
    
    # candidatesリストの初期化を追加
    candidates = []

    sims = []
    for q in question_embeddings:
        # 「質問＋選択肢」の埋め込みとの類似度を計算
        score_full = cosine_similarity([seg_emb], [q["question_embedding"]])[0][0]
        # 「選択肢のみ」の埋め込みとの類似度を計算
        score_options = cosine_similarity([seg_emb], [q["options_embedding"]])[0][0]

        candidates.append({
            "id": q["id"],
            "question": q["question"],
            "score_full": score_full,
            "score_options": score_options
        })
    
    # 「質問＋選択肢」埋め込みによる上位2件
    top_full = sorted(candidates, key=lambda x: x["score_full"], reverse=True)[:2]
    # 「選択肢のみ」埋め込みによる上位4件
    top_options = sorted(candidates, key=lambda x: x["score_options"], reverse=True)[:4]
    
    # 両方の候補リストをマージ（IDが重複している場合は最高スコアを採用）
    merged = {}
    for cand in top_full:
        merged[cand["id"]] = cand  # 既にscore_fullが入っているので、scoreは後で設定
    for cand in top_options:
        if cand["id"] in merged:
            # 既存候補と比べ、より高いスコア（score_options側）を採用
            merged[cand["id"]]["score"] = max(merged[cand["id"]].get("score", merged[cand["id"]]["score_full"]), cand["score_options"])
        else:
            merged[cand["id"]] = cand
            merged[cand["id"]]["score"] = cand["score_options"]
    
    # 各候補について、scoreが設定されていなければ max(score_full, score_options) を設定
    final_candidates = []
    for cand in merged.values():
        if "score" not in cand:
            cand["score"] = max(cand["score_full"], cand["score_options"])
        final_candidates.append((cand["id"], cand["question"], cand["score"]))
    
    # スコアの高い順にソートして返す
    final_candidates = sorted(final_candidates, key=lambda x: x[2], reverse=True)
    return final_candidates

# 6. LLMにより、該当の質問と質問部分を除去した回答文を判定
def llm_disambiguation(segment, candidates):
    """
    LLMに「このテキストは候補のどの質問に対する回答か？」を尋ねる。
    candidates は (id, 質問文, score) のタプルのリストを前提とする。
    """
    system_message = {
        "role": "system",
        "content": (
            "以下の短いテキストが、どの質問に対する回答かを推定してください。"
            "候補の質問リストを渡すので、その中で最も適切なものを1つ選び、JSONで出力してください。"
            "また、テキストから質問文を除去した回答文も出力してください。"
            "該当が無い場合は 'none' と出力してください。"
        )
    }


    # 返り値が (id, score) の場合の整形コード
    candidate_str = "\n".join([f"{c[0]} :{c[1]}(embedding_score={c[(2)]:.2f})" for c in candidates])

    user_message = {
        "role": "user",
        "content": f"""
【テキスト】:
{segment}

【候補の質問リスト】:
{candidate_str}

出力形式例:
{{
  "selected_question_id": 1,
  "reason": "理由や推定根拠"
  "answer": "質問文を除去した回答文"
}}
もし該当が無ければ
{{
  "selected_question_id": "none",
  "reason": "回答なし"
}}
"""
    }

    response = openai.ChatCompletion.create(
        engine=LLM_DEPLOYMENT_NAME,
        messages=[system_message, user_message],
        temperature=0.0
    )

    resp_text = response.choices[0].message.content
    try:
        data = json.loads(resp_text)
        return data
    except Exception as e:
        return {
            "selected_question_id": "none",
            "reason": f"JSON parse failed: {resp_text}",
            "answer": ""
        }

def new_func():
    return 2

def create_pdf_with_answers(questions, user_answers, output_filename="output.pdf"):
    """
    A4 サイズの PDF に問題文と選択肢を横並びに印刷し、音声入力の回答に〇をつける
    :param questions: 質問リスト（JSON から読み込んだもの）
    :param user_answers: 音声入力で認識された回答のリスト
    :param output_filename: 出力する PDF ファイルの名前
    """
    # A4 サイズのキャンバス作成
    pdf = canvas.Canvas(output_filename, pagesize=A4)
    width, height = A4  # A4のサイズ（595 × 842）

    # 日本語フォント設定
    pdf.setFont("NotoSansJP", 12)

    y_position = height - 50  # テキストの開始位置
    option_spacing = 100  # 選択肢の横間隔

    for q in questions:
        # 質問を印刷
        pdf.drawString(50, y_position, f"Q: {q['question']}")
        y_position -= 30  # 次の行に移動

        x_position = 70  # 選択肢の開始位置
        for option in q["options"]:
            # 〇 を選択肢の上に描画（回答がある場合）
            if option in user_answers:
                pdf.circle(x_position + 10, y_position + 15, 7)  # (X座標, Y座標, 半径)

            # 選択肢を横並びに配置
            pdf.drawString(x_position, y_position, option)
            x_position += option_spacing  # 横方向に間隔を空ける

        y_position -= 30  # 問題ごとにスペースを開ける

        # 次のページが必要なら改ページ
        if y_position < 50:
            pdf.showPage()
            pdf.setFont("NotoSansJP", 12)
            y_position = height - 50

    pdf.save()
    print(f"PDF を {output_filename} に保存しました。")


# --- 実行フロー ---
recognized_text = speech_to_text_loop()
if not recognized_text:
    print("音声認識失敗。終了します。")
    exit()

# テキストを適当に分割（単純な例）
segments = re.split(r'[。、？！\n]', recognized_text)
segments = [seg.strip() for seg in segments if seg.strip()]

mapped_results = []

for seg in segments:
    # Embeddingで候補を絞り込み
    candidates = find_top_candidates(seg)

    # 候補が空の場合、全ての質問を候補としてLLMへ依頼
    if not candidates:
        candidates = [(q["id"], q["question"], 0.0) for q in questions]

    # LLMに最終判断させる
    llm_result = llm_disambiguation(seg, candidates)
    selected_id = llm_result["selected_question_id"]
    reason = llm_result.get("reason", "N/A")
    answer_text = llm_result.get("answer", "")

    if selected_id == "none":
        print(f"【無関係 or 不明】'{seg}' -> none (理由: {reason})")
    else:
        mapped_results.append((seg, selected_id, reason, answer_text))
        answered_options = [answer_text for seg, selected_id, reason, answer_text in mapped_results]
        
print("\n--- 最終マッピング結果 ---")
for seg, qid, reason, answer_text in mapped_results:
    # 元の質問テキストを検索
    qmatch = next((q for q in questions if q["id"] == qid), None)
    if qmatch:
        if not answer_text:
            answer_text = seg.replace(qmatch["question"], "").strip()
        print(f"質問: {qmatch['question']}")
        print(f"回答: {answer_text}")
        print(f"判定理由: {reason}")
        print("-" * 40)

# --- PDF生成 ---
create_pdf_with_answers(questions, answered_options, "回答付き問題.pdf")
