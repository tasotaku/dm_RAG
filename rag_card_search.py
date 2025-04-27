import pandas as pd
import numpy as np
from datetime import datetime
import os
import matplotlib.pyplot as plt
import matplotlib
import faiss
import pickle
from openai import OpenAI

from filter import filter_index_and_texts

# === OpenAIクライアント初期化 ===
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
embedding_model = "text-embedding-3-large"

# === 埋め込み取得関数 ===
def get_embedding(text, model=embedding_model):
    response = client.embeddings.create(
        input=[text],
        model=model
    )
    return response.data[0].embedding

# === データ前処理（CSV読み込み → テキスト化） ===
def load_card_texts_from_csv(csv_path):
    df = pd.read_csv(csv_path)

    def row_to_text(row):
        ability_parts = []
        other_parts = []
        name_parts = []

        for k, v in row.items():
            if pd.isna(v) or not v:
                continue

            if k.startswith("特殊能力"):
                ability_parts.append(f"{k}: {v}")
            elif k.startswith("レアリティ") or k.startswith("イラストレーター") or k.startswith("フレーバー") or k.startswith("マナ"):
                continue  # スキップ
            elif k.startswith("カード名"):
                name_parts.append(f"{k}: {v}")
            else:
                other_parts.append(f"{k}: {v}")

        return " | ".join(ability_parts + other_parts + name_parts)


    texts = df.apply(row_to_text, axis=1).tolist()
    return texts

# === ベクトル化とFAISSインデックス作成 ===
def create_faiss_index(texts):
    embeddings = [get_embedding(text) for text in texts]
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    return index, texts, embeddings

def save_index_and_texts(index, texts, index_path="card_data/duelmasters_cards.index", texts_path="card_data/duelmasters_cards.texts"):
    faiss.write_index(index, index_path)
    with open(texts_path, "wb") as f:
        pickle.dump(texts, f)
        
def load_index_and_texts(index_path="card_data/duelmasters_cards.index", texts_path="card_data/duelmasters_cards.texts"):
    index = faiss.read_index(index_path)
    with open(texts_path, "rb") as f:
        texts = pickle.load(f)
    return index, texts

def append_cards_to_index(new_csv_path, index_path="card_data/duelmasters_cards.index", texts_path="card_data/duelmasters_cards.texts"):
    # 既存読み込み
    index, texts = load_index_and_texts(index_path, texts_path)

    # 新データ読み込み・埋め込み
    new_texts = load_card_texts_from_csv(new_csv_path)
    new_embeddings = [get_embedding(text) for text in new_texts]

    # 追加
    index.add(np.array(new_embeddings).astype("float32"))
    texts += new_texts

    # 保存
    save_index_and_texts(index, texts, index_path, texts_path)

    print(f"{len(new_texts)} 件の新カードを追加しました。")

# === 類似カードの検索 ===
def plot_distances(distances, max_distance_threshold):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.family'] = 'Meiryo'

    # 閾値以下の距離のみ抽出
    filtered_distances = [(i+1, d) for i, d in enumerate(distances) if d <= max_distance_threshold]

    if not filtered_distances:
        print("閾値以下のデータがありません。")
        return

    x_vals, y_vals = zip(*filtered_distances)

    plt.figure(figsize=(10, 5))
    plt.plot(x_vals, y_vals, marker='o')
    plt.title(f"フィルタ済み類似カード（距離 ≦ {max_distance_threshold}）")
    plt.xlabel("順位")
    plt.ylabel("L2距離（distance）")
    plt.grid(True)

    # 最大距離のアノテーション
    max_dist = max(y_vals)
    max_idx = x_vals[y_vals.index(max_dist)]
    plt.annotate(f"最大: {max_dist:.4f}",
                 xy=(max_idx, max_dist),
                 xytext=(max_idx, max_dist + 0.05),
                 arrowprops=dict(arrowstyle="->", color='red'),
                 fontsize=10, color='red')

    plt.tight_layout()
    plt.show()

    print(f"最大距離は {max_dist:.4f}（{max_idx} 番目）")
    

def search_similar_cards(
    query,
    index,
    texts,
    top_k=20,
    max_distance_threshold=0.5,
    plot=True,
    save_path="card_data/retrieved_cards.txt",
    filter_func=None  # フィルター関数（DataFrame -> DataFrame）
):
    # 先にインデックス・テキストをフィルターで絞る
    if filter_func is not None:
        index, texts = filter_index_and_texts(index, texts, filter_func)

    query_vec = get_embedding(query)
    distances, indices = index.search(np.array([query_vec]).astype("float32"), top_k)

    distances = distances[0].tolist()
    indices = indices[0].tolist()

    if plot:
        plot_distances(distances, max_distance_threshold)

    results = []
    saved_lines = []

    for dist, idx in zip(distances, indices):
        if dist <= max_distance_threshold:
            results.append(texts[idx])
            saved_lines.append(f"[{idx}] (距離: {dist:.4f})\n{texts[idx]}\n\n---\n")

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            f.writelines(saved_lines)
        print(f"類似カード {len(results)} 件を保存しました: {save_path}")

    return results

# === 回答生成（LLMに渡す） ===
def generate_search_query_with_llm(question: str) -> str:
    system = "あなたはデュエル・マスターズのカードに詳しい検索エンジンの設計者です。"
    user = f"""以下はユーザーの質問です。
質問: 「{question}」

この質問から、デュエル・マスターズのカードを検索するためのキーワード（検索クエリ）を数語で抜き出してください。
返答はキーワードのみを半角スペース区切りで出力してください。
"""

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
    )
    return response.choices[0].message.content.strip()

def summarize_prompt_for_filename(prompt: str) -> str:
    """質問文を簡潔なファイル名用タイトルに要約する（最大20文字）"""
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "以下の質問をファイル名に使えるように10〜20文字で簡潔に表現してください。句読点やスペースはできるだけ使わず、わかりやすく要約してください。"},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

def save_answer_text(prompt, answer, save_dir="answers"):
    os.makedirs(save_dir, exist_ok=True)

    # タイトル要約 & ファイル名用に安全化
    title = summarize_prompt_for_filename(prompt)
    safe_title = "".join(c for c in title if c.isalnum() or c in ('ー', '・', '_')).rstrip()

    # タイムスタンプ付きファイル名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{safe_title}.txt"
    file_path = os.path.join(save_dir, filename)

    # 保存
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"質問: {prompt}\n\n回答:\n{answer}")

    print(f"回答を保存しました: {file_path}")
    
def answer_question_with_rag(
    user_question,
    index,
    texts,
    save_dir="answers",
    search_query=None,
    top_k=500,
    max_distance_threshold=5,
    plot=True,
    filter_func=None  # フィルター関数を追加
):
    # 検索クエリを決定（手動 or 自動）
    actual_query = search_query or generate_search_query_with_llm(user_question)

    # フィルター処理（必要であれば）
    if filter_func:
        index, texts = filter_index_and_texts(index, texts, filter_func)
        print(f"フィルタ後のカード数: {len(texts)}")

    # 類似カード検索
    retrieved_contexts = search_similar_cards(
        actual_query,
        index,
        texts,
        top_k=top_k,
        max_distance_threshold=max_distance_threshold,
        plot=plot
    )
    context = "\n".join(retrieved_contexts)

    # LLM用プロンプト作成
    prompt = f"""以下はカード情報の抜粋です。\n{context}\n\n質問: {user_question}\n答え:"""
    with open("prompt.txt", "w", encoding="utf-8") as f:
        f.write(prompt)

    # 回答生成
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "system",
                "content": ("あなたはデュエル・マスターズのカードに詳しいアシスタントです。ユーザーは条件に合うカードを探しています。"
                            "提示されたカードデータの中から、ユーザーが探している条件に少しでも合うカードを全て見つけてください。"
                            "そしてそのカードのデータを全て出力してください。"
                )
            },
            {"role": "user", "content": prompt}
        ]
    )
    answer = response.choices[0].message.content

    # 保存
    save_answer_text(user_question, answer, save_dir)
    return answer

def generate_answer_from_context(
    user_question,
    context,
    save_dir="answers",
    save_prompt_path="prompt.txt"
):
    # プロンプト作成
    prompt = f"""以下はカード情報の抜粋です。\n{context}\n\n質問: {user_question}\n答え:"""

    # プロンプトを保存
    if save_prompt_path:
        with open(save_prompt_path, "w", encoding="utf-8") as f:
            f.write(prompt)

    # 回答生成
    system_prompt = """
    あなたはデュエル・マスターズのカードに詳しいアシスタントです。ユーザーの質問に、提示されたカードデータの中から答えてください。
    デュエマのルール
    ターン初めにカードを1枚引く。手札からマナを1枚チャージ。コスト数だけマナをタップしてカードを使用。基本、マナをアンタップできるのはターン開始時のみ。
    つまり、何かマナを増やすカードとか軽減するカードを使わなければ、ターン数=マナの数=支払うことができる合計コスト。
    """
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {"role": "user", "content": prompt}
        ]
    )
    answer = response.choices[0].message.content

    # 回答を保存
    save_answer_text(user_question, answer, save_dir)

    return answer