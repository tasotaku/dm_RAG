from bs4 import BeautifulSoup
import requests
import csv
from collections import defaultdict
import os
from selenium import webdriver
import time
import chromedriver_autoinstaller
import json
import urllib.parse
import pandas as pd
from datetime import datetime

def add_card_to_csv(card_url, csv_path="card_data/duelmasters_cards.csv"):
    res = requests.get(card_url)
    soup = BeautifulSoup(res.text, 'html.parser')

    # カード名ごとに card-itself をまとめる辞書
    grouped_cards = defaultdict(list)
    card_sections = soup.select("div.card-itself")

    if not card_sections:
        print("[警告] <div class='card-itself'> が見つかりません")
        return

    for section in card_sections:
        title_elem = section.select_one("h3.card-name")
        base_title = title_elem.find(text=True).strip() if title_elem else "（カード名不明）"
        grouped_cards[base_title].append(section)

    card_data_list = []

    for card_name, sections in grouped_cards.items():
        card_info = {}
        for i, section in enumerate(sections):
            suffix = f"_{i+1}"
            
            # カード名にも suffix をつけて追加
            card_info[f"カード名{suffix}"] = card_name

            text_block = section.select_one("div.grid.small-12.medium-8.column")
            if not text_block:
                print(f"[警告] 情報ブロックが見つかりません：{card_name} 側面{i+1}")
                continue

            tables = text_block.find_all("table")
            if not tables:
                print(f"[警告] tableが見つかりません：{card_name} 側面{i+1}")
                continue

            for table in tables:
                ths = table.find_all("th")
                tds = table.find_all("td")
                for th, td in zip(ths, tds):
                    key = th.get_text(strip=True) + suffix
                    val = td.get_text(separator=" ", strip=True)
                    card_info[key] = val

        card_data_list.append(card_info)

    print("card_data_list", card_data_list)
    # ===== 既存CSVの読み込み（あれば）=====
    existing_data = []
    existing_fieldnames = []
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            existing_data = list(reader)
            if reader.fieldnames:
                existing_fieldnames = reader.fieldnames

    # フィールド順の決定：「カード名」→既存順→新規順（読み込み順）
    seen = set()
    fieldnames = []

    # カード名を最初に
    for d in card_data_list:
        for key in d.keys():
            if key.startswith("カード名") and key not in seen:
                fieldnames.append(key)
                seen.add(key)

    # 既存フィールド（カード名以外）を順番通り追加
    for key in existing_fieldnames:
        if key != "カード名" and key not in seen:
            fieldnames.append(key)
            seen.add(key)

    # 新しいデータから新しいフィールドを追加（読み込み順）
    for d in card_data_list:
        for key in d.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    # データ結合して保存（既存＋新カード）
    all_data = existing_data + card_data_list

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_data)

    print(f"追加完了: {card_data_list[0].get('カード名')} → {csv_path}")


def add_card_list_to_csv(card_list, csv_path="card_data/duelmasters_cards.csv", delay=2, max_retries=3):
    for i, card_url in enumerate(card_list):
        print(f"{i+1}/{len(card_list)}: {card_url}")
        for attempt in range(1, max_retries + 1):
            try:
                add_card_to_csv(card_url, csv_path)
                break  # 成功したらループ抜ける
            except Exception as e:
                print(f"エラー: {e}（{attempt}/{max_retries} 回目）")
                time.sleep(3)
        time.sleep(delay)

    print(f"追加完了: {len(card_list)}件 → {csv_path}")
    

def make_card_list_url(page: int, show_same_name=False, products: str | None = None) -> str:
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

def get_card_urls_from_pages(start_page=1, end_page=300, products=None) -> list[str]:
    """
    指定された範囲のカード一覧ページからカード詳細URLをすべて取得する関数。
    """
    chromedriver_autoinstaller.install()
    driver = webdriver.Chrome()

    all_card_urls = []

    for page in range(start_page, end_page + 1):
        print(f"ページ {page} を取得中...")
        url = make_card_list_url(page, products=products)
        print(f"URL: {url}")
        driver.get(url)
        time.sleep(2)

        soup = BeautifulSoup(driver.page_source, "html.parser")
        card_elements = soup.find_all(class_="cardImage")

        if not card_elements:
            print(f"[警告] カードが見つかりません（ページ {page}）")
            break

        for card in card_elements:
            card_url = "https://dm.takaratomy.co.jp" + card["data-href"]
            all_card_urls.append(card_url)

    driver.quit()
    return all_card_urls

def merge_and_replace_cards(new_csv, existing_csv, output_csv=None, new_only_csv="card_data/new_cards_only.csv"):
    # デフォルトファイル名を生成（当日日付付き）
    if output_csv is None:
        today = datetime.today().strftime("%Y%m%d")
        output_csv = f"duelmasters_cards_{today}.csv"

    df_existing = pd.read_csv(existing_csv)
    df_new = pd.read_csv(new_csv)

    def extract_main_name(row):
        for col in row.index:
            if col.startswith("カード名") and pd.notna(row[col]):
                return row[col]
        return None

    new_cards_dict = {}
    for _, row in df_new.iterrows():
        name = extract_main_name(row)
        if name:
            new_cards_dict[name] = row

    updated_rows = []
    existing_names = set()

    for _, row in df_existing.iterrows():
        name = extract_main_name(row)
        if name in new_cards_dict:
            updated_rows.append(new_cards_dict[name])  # 上書き
            existing_names.add(name)
            del new_cards_dict[name]
        else:
            updated_rows.append(row)

    updated_rows.extend(new_cards_dict.values())

    df_merged = pd.DataFrame(updated_rows)
    df_merged = df_merged.reindex(columns=df_existing.columns)

    df_merged.to_csv(output_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(new_cards_dict.values()).to_csv(new_only_csv, index=False, encoding="utf-8-sig")

    print(f"マージ完了（上書き含む）→ {output_csv}")
    print(f"上書き件数: {len(existing_names)}")
    print(f"新規追加件数: {len(new_cards_dict)} → {new_only_csv}")