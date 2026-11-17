"""Veri guncelleme betiklerinin ag gerektirmeyen kisimlari."""

import json
import os

import pandas as pd
import pytest

from scripts import update_data
from scripts.add_league import read_fbref_table

from scripts.update_leagues import UpdateError as LeagueUpdateError, count_rows, table_html_to_csv

FBREF_HTML = """
<html><body>
<div id="all_stats_standard"><!--
<table id="stats_standard">
<thead>
<tr class="over_header"><th colspan="4"></th><th colspan="2" class="over_header">Playing Time</th></tr>
<tr><th data-stat="ranker">Rk</th><th data-stat="player">Player</th><th data-stat="team">Squad</th>
<th data-stat="position">Pos</th><th data-stat="minutes">Min</th><th data-stat="goals">Gls</th>
<th data-stat="matches">Matches</th></tr>
</thead>
<tbody>
<tr><th data-stat="ranker">1</th><td data-stat="player" data-append-csv="abc123"><a>Kerem Aktürkoğlu</a></td>
<td data-stat="team">Fenerbahçe</td><td data-stat="position">FW,MF</td><td data-stat="minutes">1,234</td>
<td data-stat="goals">5</td><td data-stat="matches">Matches</td></tr>
<tr class="thead"><th>Rk</th><th>Player</th><th>Squad</th><th>Pos</th><th>Min</th><th>Gls</th><th>Matches</th></tr>
<tr><th data-stat="ranker">2</th><td data-stat="player" data-append-csv="def456">Mauro Icardi</td>
<td data-stat="team">Galatasaray</td><td data-stat="position">FW</td><td data-stat="minutes">450</td>
<td data-stat="goals">3</td><td data-stat="matches">Matches</td></tr>
</tbody></table>
--></div></body></html>
"""


def test_html_table_inside_comment_converts_to_fbref_csv(tmp_path):
    pytest.importorskip("bs4")
    text = table_html_to_csv(FBREF_HTML, "stats_standard")
    assert count_rows(text) == 2
    path = tmp_path / "stats.csv"
    path.write_text(text, encoding="utf-8")
    df = read_fbref_table(str(path))
    assert list(df["Player"]) == ["Kerem Aktürkoğlu", "Mauro Icardi"]
    assert list(df["FBref_ID"]) == ["abc123", "def456"]
    # binlik ayirici temizlenir
    assert df.loc[0, "Min"] in (1234, "1234")


def test_missing_table_raises():
    pytest.importorskip("bs4")
    with pytest.raises(LeagueUpdateError):
        table_html_to_csv("<html></html>", "stats_keeper")


def test_build_light_keeps_existing_columns_in_order(tmp_path):
    light = pd.DataFrame({"Player": ["A", "B"], "Squad": ["X", "Y"], "Comp": ["eng Premier League"] * 2,
                          "Min": [900, 450], "Min%": [50.0, 25.0]})
    light_path = tmp_path / "light.csv"
    light.to_csv(light_path, index=False)
    full = light.assign(Min=[990, 540], Extra_stats_misc=[1, 2])[["Extra_stats_misc", "Min%", "Min", "Comp",
                                                                   "Squad", "Player"]]
    full_path = tmp_path / "full.csv"
    full.to_csv(full_path, index=False)

    new = update_data.build_light(str(full_path), str(light_path))
    assert list(new.columns) == list(light.columns)
    assert list(new["Min"]) == [990, 540]


def test_build_light_rejects_missing_columns_and_shrunken_files(tmp_path):
    light = pd.DataFrame({"Player": list("ABCDE"), "Squad": "X", "Comp": "c", "Min": 100, "onG": 1})
    light_path = tmp_path / "light.csv"
    light.to_csv(light_path, index=False)

    no_col = tmp_path / "no_col.csv"
    light.drop(columns="onG").to_csv(no_col, index=False)
    with pytest.raises(update_data.UpdateError, match="eksik"):
        update_data.build_light(str(no_col), str(light_path))

    small = tmp_path / "small.csv"
    light.head(2).to_csv(small, index=False)
    with pytest.raises(update_data.UpdateError, match="supheli"):
        update_data.build_light(str(small), str(light_path))


def test_committed_light_file_matches_versions_file():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(base, "data", "data_versions.json"), encoding="utf-8") as fh:
        versions = json.load(fh)
    for key in ("big5_updated", "transfermarkt_snapshot", "leagues_updated"):
        pd.Timestamp(versions[key])  # gecerli tarih


def test_saved_page_detection():
    pytest.importorskip("bs4")
    from scripts.update_leagues import detect_saved_page
    html = ('<html><head><link rel="canonical" href="https://fbref.com/en/comps/26/stats/Super-Lig-Stats">'
            '</head><body>' + FBREF_HTML + '</body></html>')
    assert detect_saved_page(html) == ("super_lig", "stats")
    assert detect_saved_page(FBREF_HTML) is None  # canonical yoksa taninmaz
