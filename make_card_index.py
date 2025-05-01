import pandas as pd
import numpy as np
import os
import faiss
from datetime import datetime
from openai import OpenAI

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
        ability_parts = [
            f"{k}: {v}"
            for k, v in row.items()
            if k.startswith("特殊能力") and pd.notna(v) and str(v).strip()
        ]

        if not ability_parts:
            return "特殊能力: なし"

        return " | ".join(ability_parts)

    texts = df.apply(row_to_text, axis=1).tolist()
    return texts

# === ベクトル化とFAISSインデックス作成 ===
def create_faiss_index(texts, index_path="card_data/duelmasters_cards.index"):
    embeddings = [get_embedding(text) for text in tqdm(texts, desc="埋め込み処理中")]
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    faiss.write_index(index, index_path)

def merge_faiss_indexes(
    new_index_path,
    existing_index_path,
    output_index_path=None
):
    new_index = faiss.read_index(new_index_path)
    existing_index = faiss.read_index(existing_index_path)

    assert new_index.d == existing_index.d, "インデックスの次元数が一致しません"

    # それぞれのベクトルを取得
    new_vectors = new_index.reconstruct_n(0, new_index.ntotal)
    existing_vectors = existing_index.reconstruct_n(0, existing_index.ntotal)

    # 新 → 旧 の順で連結
    merged_vectors = np.vstack([new_vectors, existing_vectors]).astype("float32")
    merged_index = faiss.IndexFlatL2(merged_vectors.shape[1])
    merged_index.add(merged_vectors)

    # 保存先が指定されていなければ日付付きファイル名にする
    if output_index_path is None:
        today = datetime.today().strftime("%Y%m%d")
        output_index_path = f"card_data/duelmasters_cards_{today}.index"

    # 保存
    faiss.write_index(merged_index, output_index_path)
    print(f"結合されたインデックスを保存しました: {output_index_path}")