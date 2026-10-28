"""Données Tardis (trades + book_ticker) par jour, horloge de réception, cache parquet.

book   : t, bid, ask (t en secondes depuis minuit UTC).
orders : proxy de market orders = trades consécutifs de même horodatage d’échange,
         même sens et même horodatage de réception. Aucun regroupement de fills reçus plus tard.
         t = réception du lot observable (même réception pour tous ses fills), side +1 achat / -1 vente, price_ext = prix extrême atteint,
         mid = mid reçu strictement avant, depth = side × (price_ext − mid) / mid en bps (la « portée » D).
"""
from pathlib import Path
import os

import numpy as np
import pandas as pd

# Cache régénérable hors du volume de travail (plein) ; surchargeable par CJR_CACHE.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE = os.environ.get("CJR_CACHE", str(PROJECT_ROOT / "data/cache"))


def raw_file(cfg, kind, day):
    for d in cfg["data"]["raw"]:
        directory = Path(d).expanduser()
        if not directory.is_absolute():directory = PROJECT_ROOT / directory
        p = directory / kind / f"{cfg['data']['symbol']}_{day:%Y-%m-%d}.csv.gz"
        if p.exists():
            return p
    raise FileNotFoundError(f"{kind} {cfg['data']['symbol']} {day:%Y-%m-%d} introuvable dans {cfg['data']['raw']}")


def load_day(cfg, day, cache_root=CACHE):
    day = pd.Timestamp(day)
    cache = Path(cache_root) / "causal_v4" / cfg["name"] / f"{day:%Y-%m-%d}"
    if not (cache / "orders.parquet").exists():
        t0 = day.value // 1000  # µs
        b = pd.read_csv(raw_file(cfg, "book_ticker", day), engine="pyarrow",
                        usecols=["local_timestamp", "bid_price", "ask_price"])
        b = b[(b.ask_price > b.bid_price) & (b.local_timestamp >= t0) &
              (b.local_timestamp < t0 + 86400_000000)].sort_values("local_timestamp", kind="stable")
        book = pd.DataFrame({"t": (b.local_timestamp.to_numpy() - t0) / 1e6,
                             "bid": b.bid_price.to_numpy(), "ask": b.ask_price.to_numpy()})
        tr = pd.read_csv(raw_file(cfg, "trades", day), engine="pyarrow",
                         usecols=["timestamp", "local_timestamp", "side", "price"])
        tr = tr[tr.side.isin(["buy", "sell"]) & (tr.local_timestamp >= t0) &
                (tr.local_timestamp < t0 + 86400_000000)].sort_values(["local_timestamp", "timestamp"], kind="stable")
        orders = aggregate_orders(tr, t0)
        i = np.searchsorted(book.t.to_numpy(), orders.t.to_numpy(), side="left") - 1
        mid = ((book.bid + book.ask) / 2).to_numpy()
        orders["mid"] = np.where(i >= 0, mid[np.clip(i, 0, None)], np.nan)
        orders["depth"] = orders.side * (orders.price_ext - orders.mid) / orders.mid * 1e4
        orders = orders.sort_values("t", kind="stable").reset_index(drop=True)
        cache.mkdir(parents=True, exist_ok=True)
        book.to_parquet(cache / "book.parquet")
        orders.to_parquet(cache / "orders.parquet")
    return pd.read_parquet(cache / "book.parquet"), pd.read_parquet(cache / "orders.parquet")


def aggregate_orders(tr, t0):
    """Lot observable : même timestamp d'échange, même sens ET même réception.

    Les fills reçus plus tard restent des événements distincts : une extension du flux
    ne peut modifier la portée attribuée à un événement déjà observé. Sans identifiant
    de parent, ce proxy ne reconstitue pas les véritables market orders.
    """
    tr = tr.sort_values(["local_timestamp", "timestamp"], kind="stable")
    new = ((tr.timestamp != tr.timestamp.shift()) | (tr.side != tr.side.shift()) |
           (tr.local_timestamp != tr.local_timestamp.shift())).to_numpy()
    gid = np.cumsum(new) - 1
    grouped = tr.groupby(gid, sort=False)
    first = tr[new]
    side = np.where(first.side == "buy", 1, -1)
    ext = grouped.price.agg(["max", "min"])
    return pd.DataFrame({"t": (grouped.local_timestamp.max().to_numpy() - t0) / 1e6,
                         "side": side,
                         "price_ext": np.where(side == 1, ext["max"].to_numpy(), ext["min"].to_numpy())})
