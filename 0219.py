import os
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
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
import csv
import fitz
import win32com.client

# Azure Speech SDK の設定
speech_key = ""
service_region = "japaneast"
speech_config = SpeechConfig(subscription=speech_key, region=service_region)
speech_recognizer = SpeechRecognizer(speech_config=speech_config, language="ja-JP")

# Azure Text Analytics の設定
credential = AzureKeyCredential("")
endpoint = "https://ai2nd20tamaki-language.cognitiveservices.azure.com/"
text_analytics_client = TextAnalyticsClient(endpoint=endpoint, credential=credential)

# Azure OpenAI の設定
openai.api_type = "azure"
openai.api_base = ""
openai.api_version = "2023-05-15"
openai.api_key = ""

# デプロイメント設定
deployment_id = "text-embedding-ada-002"
LLM_DEPLOYMENT_NAME = "gpt-4"

# フレーズリストの作成
phrase_list_grammar = speechsdk.PhraseListGrammar.from_recognizer(speech_recognizer)

# CSVファイルからフレーズを読み込む
csv_file = "03neurological_exam_terms.csv"
base_boost = 100   # すべてのフレーズの強調回数

loaded_phrases = []  # 読み込んだフレーズを格納
with open(csv_file, encoding="utf-8") as file:
    reader = csv.reader(file)
    for phrase in reader:
        if phrase:  # 空行をスキップ
            japanese = phrase[0].strip()
            # すべてのフレーズを強調
            for _ in range(base_boost):
                phrase_list_grammar.addPhrase(japanese)
            
            loaded_phrases.append(japanese)

def speech_to_text_loop():
    recognized_texts = []
    print("音声入力を開始します。認識を終了するには 'q' を入力してください...")
    
    while True:
        temp_recognizer = SpeechRecognizer(speech_config=speech_config, language="ja-JP")
        result = temp_recognizer.recognize_once()
        
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            recognized_texts.append(result.text)
            print(f"認識結果: {result.text}")
        elif result.reason == speechsdk.ResultReason.NoMatch:
            print("音声が検出されませんでした。")
        
        try:
            user_input = input("続けますか？ (続ける: Enter / 終了: q): ")
            if user_input.lower() == 'q':
                break
        except KeyboardInterrupt:
            print("\n認識を終了します。")
            break
    
    return " ".join(recognized_texts)

def clean_recognized_text(text):
    system_message = {
        "role": "system",
        "content": (
            "あなたは神経学的診察の専門アシスタントです。"
            "以下の音声認識テキストは徒手筋力検査（MMT）の結果を口述したものです。"
            "\n"
            "【入力テキストの特性】\n"
            "・音声認識のため誤変換（例：「4」→「死」）、略語（「上腕二頭筋」→「に筋」）が含まれる\n"
            "・くり返しや省略も含まれる\n"
            "・検査結果は「部位」「左右」「0-5の数値」の組み合わせで表現される\n"
            "・筋肉は'頸部伸展','頸部屈曲','三角筋','上腕二頭筋','上腕三頭筋','手関節背屈','手関節掌屈','母指対立筋',"
            "'腸腰筋','大腿四頭筋','大腿屈筋群','前脛骨筋','下腿三頭筋'のいずれかです\n"
            "\n"
            "【出力要件】\n"
            "■ JSON形式で整形\n"
            "■ 左右別数値（該当なしはnull）\n"
            "■ 数値範囲外/不整合時は論理推定（例：「6」→5、「なし」→0）\n"
            "■ 解釈不能な要素は除外\n"
        )
    }
    user_message = {
        "role": "user",
        "content": text
    }
    response = openai.ChatCompletion.create(
        engine=LLM_DEPLOYMENT_NAME,
        messages=[system_message, user_message],
        temperature=0.0
    )
    cleaned_text = response.choices[0].message.content
    return cleaned_text

def process_excel_results(formatted_json):
    excel = None
    workbook = None
    try:
        print("Excel自動操作開始：Excelファイルをオープンします")
        excel = win32com.client.Dispatch("Excel.Application")
        excel.Visible = False
        excel.DisplayAlerts = False

        excel_file_path = os.path.join(os.path.dirname(__file__), "19.xls")
        if not os.path.exists(excel_file_path):
            raise FileNotFoundError(f"Excelファイルが見つかりません: {excel_file_path}")

        workbook = excel.Workbooks.Open(excel_file_path)
        sheet = workbook.Sheets(1)

        # マッピングファイルの読み込み
        with open("mapping_excel.json", encoding="utf-8") as f:
            mapping = json.load(f)

        # 数値を丸付き数字に変換する辞書
        circled_numbers = {
            "0": "０",
            "1": "①",
            "2": "②",
            "3": "③",
            "4": "④",
            "5": "⑤"
        }

        # JSONの各項目に対して処理
        for muscle, values in formatted_json.items():
            for side in ["右", "左"]:
                if side in values and values[side] is not None:
                    value = str(values[side])
                    if muscle in mapping:
                        cell_address = mapping[muscle][side]
                        try:
                            # もとのセルの内容を取得
                            original_text = "５　４　３　２　１　０"
                            # 該当する数字を丸付き数字に置換
                            if value in circled_numbers:
                                # 数字の位置に応じて置換
                                positions = {"5": 0, "4": 2, "3": 4, "2": 6, "1": 8, "0": 10}
                                if value in positions:
                                    pos = positions[value]
                                    new_text = (original_text[:pos] + 
                                              circled_numbers[value] + 
                                              original_text[pos+1:])
                                    sheet.Range(cell_address).Value = new_text
                        except Exception as e:
                            print(f"セル {cell_address} への書き込みに失敗しました: {str(e)}")
                    else:
                        print(f"DEBUG: マッピング {muscle} が見つかりませんでした。")

        # 保存して閉じる
        workbook.Save()
        print("Excel処理が完了しました")

    except Exception as e:
        print(f"Excelの処理中にエラーが発生しました: {str(e)}")
        
    finally:
        # 必ずExcelを終了する処理
        try:
            if workbook is not None:
                workbook.Close(SaveChanges=True)
            if excel is not None:
                excel.Quit()
                del excel
        except Exception as e:
            print(f"Excelの終了処理中にエラーが発生しました: {str(e)}")

def main():
    # 音声認識の実行
    recognized_text = speech_to_text_loop()
    if not recognized_text:
        print("音声認識失敗。終了します。")
        return

    # ChatGPT APIで認識テキストを整形してJSONを取得
    cleaned_json = clean_recognized_text(recognized_text)
    
    # 結果の表示
    print("\n=== 神経学的診察結果 ===")
    print("【音声認識テキスト】")
    print(recognized_text)
    print("\n【整形されたMMT結果】")
    try:
        # JSON文字列をPythonオブジェクトに変換して整形して表示
        formatted_json = json.loads(cleaned_json)
        print(json.dumps(formatted_json, ensure_ascii=False, indent=2))
        
        # 結果をJSONファイルとして保存
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_filename = f"mmt_result_{timestamp}.json"
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(formatted_json, f, ensure_ascii=False, indent=2)
        print(f"\n結果を {output_filename} に保存しました。")

        # Excel処理の実行
        process_excel_results(formatted_json)
        
    except json.JSONDecodeError:
        print("JSONの解析に失敗しました。生のテキストを表示します：")
        print(cleaned_json)

if __name__ == "__main__":
    main()

