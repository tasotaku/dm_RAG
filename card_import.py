"""パック単位の差分取得。ブラウザー・画像取得・埋め込みAPIは使用しない。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlencode, urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Tag

OFFICIAL = "https://dm.takaratomy.co.jp"
LIST_URL = f"{OFFICIAL}/card/"


class HtmlClient:
    """HTMLのみ取得し、応答を保存する。間隔はキャッシュに当たらない通信間に置く。"""

    def __init__(self, cache: Path = Path(".card_cache"), delay: float = 2) -> None:
        # AI_NOTE: 同じクライアントを一覧と詳細で共有し、通信間隔をまとめて管理する。
        if delay < 2:
            raise ValueError("公式サイトへの通信間隔は2秒以上を指定してください")
        self.cache = cache
        self.delay = delay
        self.session = requests.Session()
        self.last_request = 0.0
        self.requests = 0
        self.cache_hits = 0

    def html(self, url: str, data: dict[str, str] | None = None, *, refresh: bool = False) -> str:
        # AI_NOTE: requestsはHTML内の画像・script・CSSを自動取得しない。
        request_key = url + "\n" + urlencode(sorted((data or {}).items()))
        path = self.cache / (hashlib.sha256(request_key.encode()).hexdigest() + ".html")
        if path.exists() and not refresh:
            self.cache_hits += 1
            return path.read_text(encoding="utf-8")
        time.sleep(max(0, self.delay - (time.monotonic() - self.last_request)))
        try:
            self.requests += 1
            response = self.session.request(
                "POST" if data is not None else "GET", url, data=data,
                timeout=(10, 60), allow_redirects=False,
            )
            response.raise_for_status()
            if response.is_redirect or "html" not in response.headers.get("Content-Type", "").lower():
                raise ValueError(f"HTML以外の応答または転送を検出しました: {url}")
            html = response.content.decode("utf-8")
        finally:
            self.last_request = time.monotonic()
        self.cache.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(html, encoding="utf-8")
        temporary.replace(path)
        return html


def name_key(name: str) -> str:
    # AI_NOTE: Wikiと公式の引用符・全半角・空白差を吸収し、ツインパクトは両面名で比較する。
    name = unicodedata.normalize("NFKC", name).translate(str.maketrans("’‘“”", "''\"\""))
    return re.sub(r"\s+", "", name).strip("《》")


def number_key(number: str) -> str:
    # AI_NOTE: 収録番号の種類を残してゼロ埋めだけを揃える。不明な記号を推測で削らない。
    number = number.replace("㊙", "SEC")
    number = unicodedata.normalize("NFKC", number).upper().strip()
    number = re.sub(r"SEC(\d+)超", r"CHOSEC\1", number)
    number = re.sub(r"DM(\d+)SEC", r"SECDM\1", number)
    return re.sub(r"\d+", lambda m: str(int(m.group())), number)


def wiki_sections(html: str) -> list[Tag]:
    # AI_NOTE: 収録セット内の新規・再録の小見出しを許し、紹介文や参考資料を含めない。
    soup = BeautifulSoup(html, "html.parser")
    heading = next((tag for tag in soup.find_all(["h2", "h3", "h4"])
                    if tag.get_text(strip=True).startswith(("収録カード", "収録セット"))), None)
    if heading is None:
        raise ValueError("Wikiの収録カード欄が見つかりません。取得を中止します")
    sections = []
    level = int(heading.name[1])
    for sibling in heading.next_siblings:
        if not isinstance(sibling, Tag):
            continue
        if re.fullmatch(r"h[1-6]", sibling.name):
            text = sibling.get_text(strip=True)
            if text.startswith(("参考", "関連", "その他")) or int(sibling.name[1]) < level:
                break
            if int(sibling.name[1]) == level and not text.startswith(("新規カード", "再録カード")):
                break
        sections.append(sibling)
    return sections


def wiki_link_name(link: Tag) -> str:
    # AI_NOTE: 表示が片面の省略名でも、リンク先の正式な両面名を優先する。
    title = str(link.get("title", ""))
    return (title if title.startswith("《") else link.get_text(strip=True)).strip("《》")


def wiki_cards(html: str) -> dict[str, str]:
    # AI_NOTE: 同じ行の複数番号と複数面のカード名を保持する。
    cards: dict[str, str] = {}
    ambiguous: set[str] = set()
    for sibling in wiki_sections(html):
        for item in sibling.find_all("li"):
            links = [a for a in item.find_all("a") if a.find_parent("li") is item
                     and a.get_text(strip=True).startswith("《")]
            if not links:
                continue
            name = "/".join(dict.fromkeys(wiki_link_name(link) for link in links))
            tail = " ".join(str(text) for text in item.find_all(string=True)
                            if text.find_parent("li") is item and text.find_parent("a") not in links)
            for number in re.findall(r"[（(]([^()（）\s/]+)\s*/[^()（）]+[）)]", tail):
                key = number_key(number)
                if key in ambiguous:
                    continue
                if key in cards and name_key(cards[key]) != name_key(name):
                    # 元パックの型番を維持した再録など、番号だけでは区別できない行は除外判定に使わない。
                    del cards[key]
                    ambiguous.add(key)
                    continue
                cards[key] = name
    if not cards and not wiki_deck_cards(html):
        raise ValueError("Wikiの収録番号を解釈できません。取得を中止します")
    return cards


def wiki_deck_cards(html: str) -> dict[str, str]:
    # AI_NOTE: 解説中の表ではなく、収録カード・枚数という見出しを持つ表だけを読む。
    cards = {}
    for section in wiki_sections(html):
        tables = ([section] if section.name == "table" else []) + list(section.find_all("table"))
        for table in tables:
            first = table.find("tr")
            if table.find("table") or first is None:
                continue
            if not all(word in first.get_text() for word in ("収録カード", "枚数")):
                continue
            for row in table.find_all("tr"):
                links = [a for a in row.find_all("a") if a.get_text(strip=True).startswith("《")]
                if links:
                    name = "/".join(dict.fromkeys(wiki_link_name(link) for link in links))
                    cards[name] = urljoin("https://dmwiki.net", str(links[0].get("href", "")))
    return cards


def wiki_source_numbers(html: str, wiki_page: str) -> list[str]:
    # AI_NOTE: デッキで番号が省略されている場合は、未登録カードのWikiだけで番号を補う。
    numbers: list[str] = []
    for section in wiki_sections(html):
        for link in section.find_all("a"):
            if unquote(urlparse(str(link.get("href", ""))).path).lstrip("/") != wiki_page:
                continue
            item = link.find_parent("li")
            if item is None:
                continue
            local_text = " ".join(str(text) for text in item.find_all(string=True)
                                  if text.find_parent("li") is item)
            numbers.extend(number_key(n) for n in re.findall(r"[（(]([^()（）\s/]+)\s*/[^()（）]+[）)]", local_text))
    return list(dict.fromkeys(numbers))


def list_page(client: HtmlClient, page: int, product: str | None, *, refresh: bool = False) -> tuple[list[str], bool]:
    # AI_NOTE: ?v=の画面用URLではなく、公式の検索フォームと同じPOSTでHTMLだけを取得する。
    data = {"pagenum": str(page), "sort": "release_new"}
    if product:
        data["products"] = product
    soup = BeautifulSoup(client.html(LIST_URL, data, refresh=refresh), "html.parser")
    urls = list(dict.fromkeys(urljoin(OFFICIAL, str(img["data-href"]))
                             for img in soup.select(".cardImage[data-href]")))
    if not urls:
        raise ValueError(f"カード一覧が空です（ページ{page}）。条件・サイト構造を確認してください")
    for url in urls:
        parsed = urlparse(url)
        if parsed.netloc != "dm.takaratomy.co.jp" or parsed.path != "/card/detail/":
            raise ValueError(f"想定外のカードURLです: {url}")
        card_id = parse_qs(parsed.query).get("id", [""])[0]
        if product and not card_id.lower().startswith("dm" + product.lower() + "-"):
            raise ValueError(f"パック条件とカードURLが一致しません: {url}")
    has_next = any(str(a.get("data-page", "")).isdigit() and int(str(a["data-page"])) > page
                   for a in soup.select(".wp-pagenavi a[data-page]"))
    return urls, has_next


def product_urls(client: HtmlClient, product: str, *, refresh: bool = False) -> list[str]:
    # AI_NOTE: 全パックの探索や既存カードが現れた時点での打ち切りは行わない。
    urls: list[str] = []
    for page in range(1, 1001):
        found, has_next = list_page(client, page, product, refresh=refresh)
        if set(found).issubset(urls):
            raise ValueError("同じ一覧ページが繰り返されています。取得を中止します")
        urls.extend(url for url in found if url not in urls)
        if not has_next:
            return urls
    raise ValueError("一覧のページ数が上限を超えました")


def read_rows(path: Path) -> list[dict[str, str]]:
    # AI_NOTE: 既存データを読み取り、書き込み先とは分離する。
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or "カード名_1" not in reader.fieldnames:
            raise ValueError(f"カード名_1列がありません: {path}")
        return list(reader)


def row_name(row: dict[str, str]) -> str:
    # AI_NOTE: 旧CSVの両面同名形式と、両面が別の名前になっている形式の両方を読む。
    names = list(dict.fromkeys(row[k] for k in sorted(row) if k.startswith("カード名_") and row[k]))
    return name_key("/".join(names))


@dataclass
class Candidate:
    url: str
    name: str | None


def plan_cards(urls: list[str], wiki: dict[str, str], known: set[str]) -> tuple[list[Candidate], int]:
    # AI_NOTE: 番号の対応が不明なものは除外せず、詳細取得候補として報告する。
    candidates = []
    skipped = 0
    for url in dict.fromkeys(urls):
        card_id = parse_qs(urlparse(url).query)["id"][0]
        name = wiki.get(number_key(card_id.split("-", 1)[1]))
        if name is not None and name_key(name) in known:
            skipped += 1
            continue
        candidates.append(Candidate(url, name))
    return candidates, skipped


def parse_card(html: str) -> dict[str, str]:
    # AI_NOTE: 同じ詳細ページの全側面を1行に保持し、能力欄などの欠損は成功扱いにしない。
    soup = BeautifulSoup(html, "html.parser")
    sections = soup.select("div.card-itself")
    if not sections:
        raise ValueError("公式詳細のカード情報が見つかりません")
    card: dict[str, str] = {}
    for i, section in enumerate(sections, 1):
        title = section.select_one("h3.card-name")
        if title is None:
            raise ValueError("公式詳細のカード名がありません")
        for ruby in title.select("rt, rp"):
            ruby.decompose()
        for pack in title.select(".packname"):
            pack.decompose()
        title_text = title.get_text(" ", strip=True)
        if not title_text:
            raise ValueError("公式詳細のカード名が空です")
        card[f"カード名_{i}"] = title_text
        block = section.select_one("div.grid.small-12.medium-8.column")
        if block is None or not block.select("table"):
            raise ValueError(f"公式詳細の情報テーブルがありません: {title_text}")
        for table in block.select("table"):
            for th, td in zip(table.find_all("th"), table.find_all("td")):
                card[f"{th.get_text(strip=True)}_{i}"] = td.get_text(" ", strip=True)
        required = ("カードの種類", "文明", "コスト", "特殊能力")
        missing = [label for label in required if f"{label}_{i}" not in card]
        if missing:
            raise ValueError(f"公式詳細の必須欄がありません: {title_text}: {', '.join(missing)}")
    return card


def save_rows(path: Path, rows: list[dict[str, str]]) -> None:
    # AI_NOTE: 取得途中でもカードごとに保存し、書き込み中断時に完成済みCSVを壊さない。
    fields = list(dict.fromkeys(k for row in rows for k in row)) or ["カード名_1"]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def import_product(product: str, existing: Path, output: Path, client: HtmlClient, *,
                   fetch: bool = False, refresh_lists: bool = False, limit: int | None = None,
                   wiki_page: str | None = None) -> dict[str, Any]:
    # AI_NOTE: CSVだけの更新で検索データとの対応を崩さないよう、差分を別ファイルに出す。
    if existing.resolve() == output.resolve():
        raise ValueError("既存CSVと出力先は別のファイルにしてください")
    if not re.fullmatch(r"[a-z0-9]+", product):
        raise ValueError("パックコードは25rp1などの英数字で指定してください")
    rows = read_rows(existing)
    saved = read_rows(output) if output.exists() else []
    known = {row_name(row) for row in rows + saved}
    if wiki_page is None:
        modern = re.fullmatch(r"(\d{2})([a-z]+\d+[a-z]?)", product)
        older = re.fullmatch(r"([a-z]*)(\d+[a-z]?)", product)
        if modern:
            wiki_page = f"DM{modern[1]}-{modern[2].upper()}"
        elif older:
            wiki_page = f"DM{older[1].upper()}-{older[2].upper()}"
        else:
            raise ValueError("Wikiページ名を--wiki-pageで指定してください")
    if not re.fullmatch(r"[A-Za-z0-9-]+", wiki_page):
        raise ValueError("Wikiページ名はDM25-RP1などのパックコードを指定してください")
    wiki_html = client.html(f"https://dmwiki.net/{wiki_page}", refresh=refresh_lists)
    wiki = wiki_cards(wiki_html)
    deck = wiki_deck_cards(wiki_html)
    unresolved_names = []
    for name, url in deck.items():
        if name_key(name) in known or any(name_key(value) == name_key(name) for value in wiki.values()):
            continue
        numbers = wiki_source_numbers(client.html(url), wiki_page)
        if not numbers:
            unresolved_names.append(name)
        for key in numbers:
            if key in wiki and name_key(wiki[key]) != name_key(name):
                raise ValueError(f"Wikiのカード別収録番号が競合しています: {name}")
            wiki[key] = name
    urls = product_urls(client, product, refresh=refresh_lists)
    candidates, skipped = plan_cards(urls, wiki, known)
    roster_names = {name_key(name) for name in list(deck) + list(wiki.values())}
    mapped_names = {name_key(card.name) for card in candidates if card.name is not None}
    missing_roster_names = roster_names - known
    roster_complete = bool(deck) and len(roster_names) == len(urls) and missing_roster_names <= mapped_names
    if roster_complete:
        # AI_NOTE: Wikiの名簿と公式の種類数が一致し、未登録名をすべてURL対応できた時だけ既存の番号なし行を除外する。
        candidates = [card for card in candidates if card.name is not None]
        skipped = len(urls) - len(candidates)
    report: dict[str, Any] = {
        "product": product, "mode": "fetch" if fetch else "plan", "official_list_cards": len(urls),
        "wiki_numbers": len(wiki), "skipped_registered": skipped,
        "wiki_roster_complete": roster_complete, "unresolved_wiki_names": unresolved_names,
        "candidates": [asdict(card) for card in candidates],
        "candidate_details_max": len(candidates),
        "unmatched_numbers": sum(card.name is None for card in candidates),
        "added": 0, "skipped_after_detail": 0, "skipped_same_name": 0,
    }
    if fetch:
        detail_count = 0
        for candidate in candidates:
            if candidate.name is not None and name_key(candidate.name) in known:
                report["skipped_same_name"] += 1
                continue
            if limit is not None and detail_count >= limit:
                break
            card = parse_card(client.html(candidate.url))
            detail_count += 1
            key = row_name(card)
            if candidate.name is not None and key != name_key(candidate.name):
                raise ValueError(f"Wikiと公式のカード名が一致しません: {candidate.url}")
            if key in known:
                report["skipped_after_detail"] += 1
                continue
            saved.append(card)
            save_rows(output, saved)
            known.add(key)
            report["added"] += 1
        if not output.exists():
            save_rows(output, saved)
    report.update(http_requests=client.requests, cache_hits=client.cache_hits, output=str(output))
    return report


def main() -> None:
    # AI_NOTE: 既定では件数確認だけを行い、--fetch指定時だけカード詳細を取得する。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("product", help="例: 25rp1 または DM25-RP1")
    parser.add_argument("--existing", type=Path, required=True, help="比較元CSV（変更しません）")
    parser.add_argument("--output", type=Path, help="新規カードだけのCSV")
    parser.add_argument("--fetch", action="store_true", help="詳細を取得して差分CSVを保存")
    parser.add_argument("--refresh-lists", action="store_true", help="公式一覧とWikiを再取得（発売前の一覧更新時など）")
    parser.add_argument("--delay", type=float, default=2, help="通信間隔、秒（最低2秒）")
    parser.add_argument("--cache", type=Path, default=Path(".card_cache"))
    parser.add_argument("--wiki-page", help="Wiki側のページ名が通常形式と異なる場合に指定")
    parser.add_argument("--limit", type=int, help="今回取得する詳細の上限（試験取得用）")
    args = parser.parse_args()
    product = args.product.lower().removeprefix("dm").replace("-", "")
    output = args.output or Path(f"card_data/new_cards_{product}.csv")
    if args.limit is not None and args.limit < 1:
        parser.error("--limitは1以上を指定してください")
    try:
        report = import_product(product, args.existing, output, HtmlClient(args.cache, args.delay),
                                fetch=args.fetch, refresh_lists=args.refresh_lists, limit=args.limit,
                                wiki_page=args.wiki_page)
    except (requests.RequestException, ValueError, OSError) as error:
        parser.exit(1, f"取得を中止しました: {error}\n保存済みHTML・差分CSVから再開できます。\n")
    report_path = output.with_suffix(".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "candidates"}, ensure_ascii=False, indent=2))
    print(f"候補URL・カード名の一覧: {report_path}")


if __name__ == "__main__":
    main()
