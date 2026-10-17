import json
import urllib.parse
from pathlib import Path
from datetime import datetime
from collections.abc import Sequence

import pandas as pd

from card_import import HtmlClient, list_page, parse_card, read_rows, row_name, save_rows

def add_card_to_csv(card_url: str, csv_path: str | Path = "card_data/duelmasters_cards.csv", *,
                    client: HtmlClient | None = None) -> None:
    # AI_NOTE: 旧ノートブックの呼び出しもHTML保存と重複防止を共通で使う。
    client = client or HtmlClient()
    card = parse_card(client.html(card_url))
    path = Path(csv_path)
    rows = read_rows(path) if path.exists() else []
    if row_name(card) in {row_name(row) for row in rows}:
        print(f"登録済み: {card['カード名_1']}")
        return
    save_rows(path, rows + [card])
    print(f"追加完了: {card['カード名_1']} → {csv_path}")


def add_card_list_to_csv(card_list: Sequence[str], csv_path: str | Path = "card_data/duelmasters_cards.csv",
                         delay: float = 2, max_retries: int = 3) -> None:
    # AI_NOTE: 成功済みHTMLを再利用し、失敗時は止めて同じ呼び出しから再開できる。
    client = HtmlClient(delay=delay)
    for i, card_url in enumerate(dict.fromkeys(card_list), 1):
        print(f"{i}/{len(card_list)}: {card_url}")
        add_card_to_csv(card_url, csv_path, client=client)
    print(f"完了: 通信 {client.requests}件、保存済みHTML利用 {client.cache_hits}件")


def make_card_list_url(page: int, show_same_name: bool = False, products: str | None = None) -> str:
    query = {
        "suggest": "on",
        "keyword_type": ["card_name", "card_ruby", "card_text"],
        "culture_cond": ["単色", "多色"],
        "pagenum": str(page),
        "sort": "release_new",
    }

    # 同名カードを表示するかどうか
    if show_same_name:
        query["samename"] = "show"
        
    if products:
        query["products"] = products

    json_str = json.dumps(query, ensure_ascii=False, separators=(",", ":"))
    encoded = urllib.parse.quote(json_str, safe=":,[]\"")
    return f"https://dm.takaratomy.co.jp/card/?v={encoded}"

def get_card_urls_from_pages(start_page: int = 1, end_page: int = 300,
                             products: str | None = None) -> list[str]:
    # AI_NOTE: Chromeを起動せず、ページ切替も公式フォームと同じPOSTで取得する。
    client = HtmlClient()
    product = str(products).lower().removeprefix("dm").replace("-", "") if products else None
    urls = []
    for page in range(start_page, end_page + 1):
        found, has_next = list_page(client, page, product)
        if set(found).issubset(urls):
            raise ValueError("同じカード一覧が繰り返されています")
        urls.extend(url for url in found if url not in urls)
        if not has_next:
            break
    return urls


def make_new_cards_only(
    new_csv_path,
    existing_csv_path,
    output_new_only_csv_path="card_data/new_cards_only.csv"
):
    df_new = pd.read_csv(new_csv_path)
    df_existing = pd.read_csv(existing_csv_path)

    # 既存のカード名に含まれていない行だけ抽出（＝新規カード）
    df_new_only = df_new[~df_new["カード名_1"].isin(df_existing["カード名_1"])]

    df_new_only.to_csv(output_new_only_csv_path, index=False, encoding="utf-8-sig")
    
def concat_new_and_existing_cards(
    new_csv_path,
    existing_csv_path,
    output_csv_path=None
):
    df_existing = pd.read_csv(existing_csv_path)
    df_new = pd.read_csv(new_csv_path)

    # 列順を既存の順に合わせる
    df_new = df_new.reindex(columns=df_existing.columns)

    # 新カードを先頭に連結
    df_combined = pd.concat([df_new, df_existing], ignore_index=True)

    # 保存先のパスが空なら現在日付で自動命名
    if not output_csv_path:
        today = datetime.today().strftime("%Y%m%d")
        output_csv_path = f"card_data/duelmasters_cards_{today}.csv"

    df_combined.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
