"""通信せずに、誤除外・再取得・途中失敗を確認する。"""

import csv
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests

from card_import import (
    HtmlClient, import_product, list_page, name_key, number_key,
    parse_card, plan_cards, product_urls, row_name, save_rows, wiki_cards,
    wiki_deck_cards, wiki_source_numbers,
)


@pytest.mark.parametrize(("wiki", "official"), [
    ("1", "001"), ("S1", "S01"), ("TR3", "TR03"), ("TD1", "TD01"),
    ("PR1", "PR001"), ("SP1", "SP1"), ("㊙1", "Sec01"),
    ("㊙1超", "ChoSec001"), ("DM1㊙", "SecDM1"),
])
def test_special_numbers(wiki, official):
    assert number_key(wiki) == number_key(official)


def test_names_and_twinpact():
    assert name_key("Ａ ／ Ｂ") == name_key("A / B")
    assert name_key("Drache der’Zen") == name_key("Drache der'Zen")
    assert row_name({"カード名_1": "A/B", "カード名_2": "A/B"}) == name_key("A/B")
    assert row_name({"カード名_1": "A", "カード名_2": "B"}) == name_key("A/B")
    assert name_key("A/B") != name_key("A")


WIKI = """<h3>紹介</h3><ul><li><a>《紹介だけ》</a>（3/3）</li></ul>
<h3>収録カード</h3><h4>通常</h4><ul>
<li><a>《登録済み》</a>（1/2）（PR1/PR2）</li>
<li><a>《新カード》</a>（2/2）</li>
<li><a>《新カード》</a>（㊙1/㊙1）（㊙1超/㊙1）</li></ul>
<h3>参考</h3><ul><li><a>《参考だけ》</a>（4/4）</li></ul>"""


def listing(ids, next_page=None):
    cards = "".join(f'<img class="cardImage" data-href="/card/detail/?id=dm25rp1-{i}">' for i in ids)
    nav = f'<div class="wp-pagenavi"><a data-page="{next_page}"></a></div>' if next_page else ""
    return cards + nav


def detail(name="新カード"):
    return f'''<div class="card-itself"><h3 class="card-name">{name}<span class="packname">(PACK)</span></h3>
    <div class="grid small-12 medium-8 column"><table>
    <tr><th>カードの種類</th><td>クリーチャー</td></tr>
    <tr><th>文明</th><td>水</td></tr><tr><th>コスト</th><td>3</td></tr>
    <tr><th>特殊能力</th><td>能力１<br>能力２</td></tr></table></div></div>'''


def test_wiki_multiple_numbers_and_scope():
    cards = wiki_cards(WIKI)
    assert cards == {"1": "登録済み", "PR1": "登録済み", "2": "新カード", "SEC1": "新カード", "CHOSEC1": "新カード"}


def test_ambiguous_wiki_numbers_are_not_used_for_exclusion():
    cards = wiki_cards(WIKI.replace("（2/2）", "（1/2）"))
    assert '1' not in cards
    assert cards['PR1'] == '登録済み'


def test_missing_wiki_stops():
    with pytest.raises(ValueError):
        wiki_cards("<h3>別のページ</h3>")


def test_plan_keeps_unknown_ids_and_deduplicates_urls():
    urls = [f"https://dm.takaratomy.co.jp/card/detail/?id=dm25rp1-{i}" for i in ["001", "Sec01", "unknown", "unknown"]]
    plan, skipped = plan_cards(urls, wiki_cards(WIKI), {name_key("登録済み")})
    assert skipped == 1
    assert [c.name for c in plan] == ["新カード", None]


def test_pagination_posts_and_checks_product():
    client = Mock()
    client.html.side_effect = [listing(["001"], 2), listing(["002"])]
    assert len(product_urls(client, "25rp1")) == 2
    assert client.html.call_args_list[0].args[1]["products"] == "25rp1"
    assert client.html.call_args_list[1].args[1]["pagenum"] == "2"
    client.html.return_value = listing(["001"]).replace("dm25rp1", "dm26ex4")
    client.html.side_effect = None
    with pytest.raises(ValueError, match="パック条件"):
        list_page(client, 1, "25rp1")


def test_repeated_page_stops():
    client = Mock()
    client.html.return_value = listing(["001"], 3)
    with pytest.raises(ValueError, match="繰り返"):
        product_urls(client, "25rp1")


def test_detail_keeps_both_sides_and_excludes_pack_label():
    card = parse_card(detail("A / B") + detail("A / B"))
    assert card["カード名_1"] == card["カード名_2"] == "A / B"
    assert card["特殊能力_2"] == "能力１ 能力２"


def test_incomplete_detail_stops():
    with pytest.raises(ValueError):
        parse_card('<div class="card-itself"><h3 class="card-name">名前だけ</h3></div>')


def test_missing_ability_field_stops_but_empty_ability_is_valid():
    with pytest.raises(ValueError, match="特殊能力"):
        parse_card(detail().replace('<tr><th>特殊能力</th><td>能力１<br>能力２</td></tr>', ''))
    assert parse_card(detail().replace('能力１<br>能力２', ''))['特殊能力_1'] == ''


def response(html):
    r = requests.Response()
    r.status_code = 200
    r._content = html.encode()
    r.headers["Content-Type"] = "text/html; charset=UTF-8"
    return r


def test_cache_requests_only_html_and_preserves_interval(tmp_path):
    client = HtmlClient(tmp_path)
    client.session = Mock()
    client.session.request.return_value = response('<img src="image.jpg"><script src="script.js"></script>')
    with patch("card_import.time.sleep") as sleep, patch("card_import.time.monotonic", return_value=100):
        first = client.html("https://example.com/page")
        assert client.html("https://example.com/page") == first
        client.html("https://example.com/other")
    assert client.session.request.call_count == 2
    assert sleep.call_args_list[-1].args[0] == 2
    assert client.cache_hits == 1
    assert all(call.args[1].endswith(("/page", "/other")) for call in client.session.request.call_args_list)


def test_http_failure_is_not_cached_or_retried(tmp_path):
    client = HtmlClient(tmp_path)
    client.session = Mock()
    r = response("error")
    r.status_code = 429
    client.session.request.return_value = r
    with pytest.raises(requests.HTTPError):
        client.html("https://example.com/page")
    assert client.session.request.call_count == 1
    assert not list(tmp_path.glob("*.html"))


def fake_client():
    client = Mock()
    client.requests = 0
    client.cache_hits = 0
    def html(url, data=None, **kwargs):
        if "dmwiki" in url:
            return WIKI
        if data is not None:
            return listing(["001", "002", "Sec01"])
        return detail()
    client.html.side_effect = html
    return client


def test_fetch_filters_before_detail_and_resume_does_not_fetch_again(tmp_path):
    existing, output = tmp_path / "existing.csv", tmp_path / "new.csv"
    save_rows(existing, [{"カード名_1": "登録済み"}])
    original = existing.read_bytes()
    client = fake_client()
    report = import_product("25rp1", existing, output, client, fetch=True)
    assert report["added"] == 1
    assert report["skipped_registered"] == 1
    assert report["skipped_same_name"] == 1
    assert len([call for call in client.html.call_args_list if "/detail/" in call.args[0]]) == 1
    assert existing.read_bytes() == original
    client.html.reset_mock()
    report = import_product("25rp1", existing, output, client, fetch=True)
    assert report["added"] == 0
    assert not any("/detail/" in call.args[0] for call in client.html.call_args_list)
    with output.open(encoding="utf-8-sig", newline="") as f:
        assert len(list(csv.DictReader(f))) == 1


def test_plan_does_not_fetch_details_or_write_csv(tmp_path):
    existing, output = tmp_path / "existing.csv", tmp_path / "new.csv"
    save_rows(existing, [{"カード名_1": "登録済み"}])
    client = fake_client()
    import_product("25rp1", existing, output, client)
    assert not output.exists()
    assert not any("/detail/" in call.args[0] for call in client.html.call_args_list)


def test_wrong_wiki_name_stops_without_writing(tmp_path):
    existing, output = tmp_path / "existing.csv", tmp_path / "new.csv"
    save_rows(existing, [{"カード名_1": "登録済み"}])
    client = fake_client()
    original = client.html.side_effect
    client.html.side_effect = lambda url, data=None, **kw: detail("別カード") if "/detail/" in url else original(url, data, **kw)
    with pytest.raises(ValueError, match="一致しません"):
        import_product("25rp1", existing, output, client, fetch=True)
    assert not output.exists()


def test_existing_csv_cannot_be_overwritten(tmp_path):
    path = tmp_path / "master.csv"
    with pytest.raises(ValueError, match="別のファイル"):
        import_product("25rp1", path, path, fake_client(), fetch=True)


def test_minimum_delay():
    with pytest.raises(ValueError):
        HtmlClient(delay=0)


def test_partial_failure_keeps_completed_rows(tmp_path):
    existing, output = tmp_path / "existing.csv", tmp_path / "new.csv"
    save_rows(existing, [])
    client = fake_client()
    original = client.html.side_effect
    def html(url, data=None, **kwargs):
        if "/detail/" in url:
            if "001" in url:
                return detail("登録済み")
            raise requests.Timeout("interrupted")
        return original(url, data, **kwargs)
    client.html.side_effect = html
    with pytest.raises(requests.Timeout):
        import_product("25rp1", existing, output, client, fetch=True)
    assert output.exists()
    with output.open(encoding="utf-8-sig", newline="") as f:
        assert next(csv.DictReader(f))["カード名_1"] == "登録済み"


def test_new_csv_columns_are_preserved(tmp_path):
    output = tmp_path / "new.csv"
    save_rows(output, [{"カード名_1": "A"}, {"カード名_1": "B", "特殊能力_3": "三面目"}])
    with output.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[1]["特殊能力_3"] == "三面目"


def test_set_heading_keeps_same_level_new_and_reprint_subsections():
    html = '<h3>収録セット</h3><h3>新規カード 全1種</h3><ul><li><a>《A》</a>（N1/N1）</li></ul><h3>再録カード 全1種</h3><ul><li><a>《B》</a>/<a>《C》</a>（1/1）</li></ul><h3>参考</h3><ul><li><a>《D》</a>（2/2）</li></ul>'
    assert wiki_cards(html) == {'N1': 'A', '1': 'B/C'}


def test_deck_roster_ignores_commentary_tables_and_uses_canonical_name():
    html = '<h3>収録カード</h3><table><tr><td>■収録カード</td><td>枚数</td></tr><tr><td><a href="/A" title="《A／B》">《A》</a></td><td>4</td></tr></table><table><tr><td>解説</td></tr><tr><td><a>《紹介だけ》</a></td></tr></table><h3>参考</h3>'
    assert wiki_cards(html) == {}
    assert wiki_deck_cards(html) == {'A／B': 'https://dmwiki.net/A'}


def test_source_numbers_only_use_exact_pack_inside_source_section():
    html = '<h3>収録セット</h3><ul><li><a href="/DM25-BD3">DM25-BD3</a>（12/20）（㊙1/㊙8）</li><li><a href="/DM25-BD2">DM25-BD2</a>（3/20）</li></ul><h3>参考</h3>'
    assert wiki_source_numbers(html, 'DM25-BD3') == ['12', 'SEC1']


def test_source_numbers_do_not_read_ancestor_illustrator_sibling_packs():
    html = '<h3>収録セット</h3><ul><li>illus.Artist<ul><li><a href="/DM25-BD3">target</a>（12/20）</li><li><a href="/DM24-RP1">other</a>（1/77）</li></ul></li></ul><h3>参考</h3>'
    assert wiki_source_numbers(html, 'DM25-BD3') == ['12']


def test_numbered_roster_ignores_ancestor_li_and_keeps_interleaved_face_numbers():
    html = '<h3>収録カード</h3><ul><li>元パックと同じ<ul><li><a>《A》</a>（1/2）</li><li><a>《B》</a>（2/2）</li></ul></li><li><a>《C》</a>（3a/3）/<a>《D》</a>（3b/3）</li></ul><h3>参考</h3>'
    assert wiki_cards(html) == {'1': 'A', '2': 'B', '3A': 'C/D', '3B': 'C/D'}


def test_deck_prefilter_requires_roster_count_and_every_missing_name(tmp_path):
    existing, output = tmp_path / 'existing.csv', tmp_path / 'new.csv'
    save_rows(existing, [{'カード名_1': '登録済み'}])
    roster = '<h3>収録カード</h3><table><tr><td>収録カード</td><td>枚数</td></tr><tr><td><a href="/known">《登録済み》</a></td><td>4</td></tr><tr><td><a href="/new">《新カード》</a></td><td>4</td></tr></table><h3>参考</h3>'
    client = fake_client()
    def html(url, data=None, **kw):
        if url.endswith('/DM25-RP1'):
            return roster
        if url.endswith('/new'):
            return '<h3>収録セット</h3><ul><li><a href="/DM25-RP1">pack</a>（2/2）</li></ul><h3>参考</h3>'
        if data:
            return listing(['001', '002'])
        raise AssertionError('plan must not fetch details or registered Wiki card')
    client.html.side_effect = html
    report = import_product('25rp1', existing, output, client)
    assert report['wiki_roster_complete']
    assert report['candidate_details_max'] == 1
    assert report['skipped_registered'] == 1
