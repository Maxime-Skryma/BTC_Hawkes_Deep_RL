from cjr.fees import fee_label, fee_table


def test_fee_display_converts_rates_without_exposing_internal_names():
    fees = {'vip6_bnb': {'maker': .00015, 'taker': .0002175}}
    table = fee_table(fees)
    assert table.iloc[0]['Maker (bps)'] == 1.5
    assert table.iloc[0]['Taker (bps)'] == 2.175
    assert 'vip' not in table.to_string()
    assert fee_label(fees['vip6_bnb']) == 'Maker 1.5 bps · taker 2.175 bps'


def test_high_fees_are_identified():
    assert 'frais élevés' in fee_label({'maker': .00075, 'taker': .00075})
