"""Présentation des scénarios de frais, sans modifier leurs identifiants sauvegardés."""
import pandas as pd


def fee_label(fees):
    maker = float(fees['maker']) * 1e4
    taker = float(fees['taker']) * 1e4
    label = f'Maker {maker:g} bps · taker {taker:g} bps'
    if maker >= 7.5:
        label += ' — frais élevés'
    return label


def fee_table(fees):
    rows = [{'Scénario': f'Maker {float(v["maker"])*1e4:g} bps',
             'Maker (bps)': float(v['maker'])*1e4,
             'Taker (bps)': float(v['taker'])*1e4,
             'Usage': 'Test de frais élevés' if float(v['maker'])*1e4 >= 7.5 else 'Comparaison'}
            for v in fees.values()]
    return pd.DataFrame(rows).set_index('Scénario').round(6)
