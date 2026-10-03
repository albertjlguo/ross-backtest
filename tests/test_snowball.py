import numpy as np
import pandas as pd

from snowball.fundamentals import annual_table, cagr, dcf_value, derive, implied_growth


def _fact(tag_vals, duration=True):
    items = []
    for y, v in tag_vals.items():
        it = {"end": f"{y}-12-31", "val": v, "form": "10-K", "filed": f"{y + 1}-02-15", "fy": y, "fp": "FY"}
        if duration:
            it["start"] = f"{y}-01-01"
        items.append(it)
    return {"units": {"USD": items}}


def _js():
    yrs = range(2012, 2025)
    rev = {y: 1000 * 1.1 ** (y - 2012) for y in yrs}
    g = {"Revenues": _fact(rev),
         "OperatingIncomeLoss": _fact({y: v * 0.3 for y, v in rev.items()}),
         "NetIncomeLoss": _fact({y: v * 0.22 for y, v in rev.items()}),
         "DepreciationDepletionAndAmortization": _fact({y: v * 0.05 for y, v in rev.items()}),
         "PaymentsToAcquirePropertyPlantAndEquipment": _fact({y: v * 0.08 for y, v in rev.items()}),
         "NetCashProvidedByUsedInOperatingActivities": _fact({y: v * 0.3 for y, v in rev.items()}),
         "ShareBasedCompensation": _fact({y: v * 0.02 for y, v in rev.items()}),
         "WeightedAverageNumberOfDilutedSharesOutstanding": {"units": {"shares": _fact({y: 100 for y in yrs})["units"]["USD"]}},
         "Assets": _fact({y: v * 1.2 for y, v in rev.items()}, False),
         "LiabilitiesCurrent": _fact({y: v * 0.2 for y, v in rev.items()}, False),
         "CashAndCashEquivalentsAtCarryingValue": _fact({y: v * 0.1 for y, v in rev.items()}, False),
         "PropertyPlantAndEquipmentNet": _fact({y: v * 0.5 for y, v in rev.items()}, False),
         "PropertyPlantAndEquipmentGross": _fact({y: v * 0.8 for y, v in rev.items()}, False),
         "StockholdersEquity": _fact({y: v * 0.7 for y, v in rev.items()}, False)}
    # 混入一条季度数据，应被忽略
    g["Revenues"]["units"]["USD"].append({"start": "2024-01-01", "end": "2024-03-31", "val": 1, "form": "10-Q", "filed": "2024-05-01"})
    return {"facts": {"us-gaap": g}}


def test_annual_and_derive():
    d = derive(annual_table(_js()))
    assert len(d) == 13 and d.index[-1] == pd.Timestamp("2024-12-31")
    L = d.iloc[-1]
    assert 0.2 < L["roic"] < 0.4
    assert L["maint_base"] == L["dep"] and L["maint_stress"] >= L["maint_base"] and L["maint_worst"] >= L["dep"]
    assert L["maint_greenwald"] <= L["capex"]
    assert L["owner_earnings"] == L["net_income"] + L["da"] - L["maint_base"]
    assert abs(L["useful_life"] - 16) < 0.01            # 0.8 / 0.05
    assert abs(cagr(d["revenue"], 10) - 0.1) < 1e-9


def test_dcf_roundtrip():
    v = dcf_value(100, 0.08, 0.07)
    assert abs(implied_growth(v, 100, 0.07) - 0.08) < 1e-4
    assert np.isnan(implied_growth(1000, -5, 0.07))
