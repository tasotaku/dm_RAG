import pandas as pd
import numpy as np
import faiss

def filter_by_special_ability(
    df,
    include_keywords_and=None,
    include_keywords_or=None,
    exclude_keywords_and=None,
    exclude_keywords_or=None
):
    include_keywords_and = include_keywords_and or []
    include_keywords_or = include_keywords_or or []
    exclude_keywords_and = exclude_keywords_and or []
    exclude_keywords_or = exclude_keywords_or or []

    mask = pd.Series([True] * len(df), index=df.index)

    # AND含む: 全てのキーワードが含まれる
    if include_keywords_and:
        mask &= df.apply(
            lambda row: all(
                any(kw in str(row.get(col, "")) for col in df.columns if "特殊能力" in col)
                for kw in include_keywords_and
            ),
            axis=1
        )

    # OR含む: いずれかのキーワードを含む
    if include_keywords_or:
        mask &= df.apply(
            lambda row: any(
                kw in str(row.get(col, ""))
                for col in df.columns if "特殊能力" in col
                for kw in include_keywords_or
            ),
            axis=1
        )

    # AND除外: 全てのキーワードが含まれていない
    if exclude_keywords_and:
        mask &= df.apply(
            lambda row: all(
                not any(kw in str(row.get(col, "")) for col in df.columns if "特殊能力" in col)
                for kw in exclude_keywords_and
            ),
            axis=1
        )

    # OR除外: どれか1つでも含まれていたら除外
    if exclude_keywords_or:
        mask &= df.apply(
            lambda row: all(
                kw not in str(row.get(col, ""))
                for col in df.columns if "特殊能力" in col
                for kw in exclude_keywords_or
            ),
            axis=1
        )

    return df[mask]


def filter_by_civilization(df, include_civs=None, exclude_civs=None):
    include_civs = include_civs or []
    exclude_civs = exclude_civs or []

    mask = pd.Series([True] * len(df), index=df.index)

    if include_civs:
        mask &= df.apply(
            lambda row: any(
                civ in str(row.get(col, ""))
                for col in df.columns if "文明" in col
                for civ in include_civs
            ),
            axis=1
        )

    if exclude_civs:
        mask &= df.apply(
            lambda row: all(
                civ not in str(row.get(col, ""))
                for col in df.columns if "文明" in col
                for civ in exclude_civs
            ),
            axis=1
        )

    return df[mask]

def filter_by_cost(df, min_cost=None, max_cost=None):
    cost_mask = pd.Series([False] * len(df), index=df.index)

    for col in df.columns:
        if "コスト" in col:
            try:
                col_values = pd.to_numeric(df[col], errors="coerce")
                this_mask = pd.Series([True] * len(df), index=df.index)

                if min_cost is not None:
                    this_mask &= (col_values >= min_cost)
                if max_cost is not None:
                    this_mask &= (col_values <= max_cost)

                cost_mask |= this_mask.fillna(False)
            except Exception:
                continue

    return df[cost_mask]

def filter_by_race_keyword(df, race_keywords=None):
    race_keywords = race_keywords or []
    mask = pd.Series([True] * len(df), index=df.index)

    for keyword in race_keywords:
        mask |= df.apply(
            lambda row: any(
                keyword.lower() in str(row.get(col, "")).lower()
                for col in df.columns if "種族" in col
            ),
            axis=1
        )

    return df[mask]

def apply_all_filters(
    df,
    include_ability_and=None,
    include_ability_or=None,
    exclude_ability_and=None,
    exclude_ability_or=None,
    civ_include=None,
    civ_exclude=None,
    min_cost=None,
    max_cost=None,
    race_keywords=None
):
    df = filter_by_special_ability(df, include_ability_and, include_ability_or, exclude_ability_and, exclude_ability_or)
    df = filter_by_civilization(df, civ_include, civ_exclude)
    df = filter_by_cost(df, min_cost, max_cost)
    df = filter_by_race_keyword(df, race_keywords)
    return df.reset_index(drop=False)

def parse_texts_to_dataframe(texts):
    # textsの各要素は "列名: 値 | 列名: 値 | ..." なので dict に変換
    rows = []
    for text in texts:
        row = {}
        for part in text.split(" | "):
            if ": " in part:
                key, val = part.split(": ", 1)
                row[key.strip()] = val.strip()
        rows.append(row)
    return pd.DataFrame(rows)

def filter_index_with_dataframe(index, df, filter_func):
    df_filtered = filter_func(df)
    filtered_indices = df_filtered.index.tolist()

    all_vectors = index.reconstruct_n(0, index.ntotal)
    filtered_vectors = np.array([all_vectors[i] for i in filtered_indices]).astype("float32")

    new_index = faiss.IndexFlatL2(index.d)
    new_index.add(filtered_vectors)

    return new_index, df_filtered