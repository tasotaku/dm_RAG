import numpy as np
from datetime import datetime
import os
from openai import OpenAI
import matplotlib.pyplot as plt
import matplotlib
from filter import filter_index_with_dataframe
from make_card_index import get_embedding

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === 類似カードの検索 ===
def plot_distances(distances, max_distance_threshold):
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
    

import numpy as np

def search_similar_cards(
    query,
    index,
    df,
    top_k=20,
    max_distance_threshold=0.5,
    plot=True,
    save_path="card_data/retrieved_cards.txt",
    filter_func=None  # フィルター関数（DataFrame -> DataFrame）
):
    """
    検索用ベクトルは df[text_column] から作成されている前提。
    ヒット結果として df の該当行を返す。
    """

    # フィルターがある場合、indexとdfの対応を保ったまま絞り込む
    if filter_func is not None:
        index, df = filter_index_with_dataframe(index, df, filter_func)

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
            row = df.iloc[idx]
            results.append(row)
            saved_lines.append(f"[{idx}] (距離: {dist:.4f})\n{row.to_string()}\n\n---\n")

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