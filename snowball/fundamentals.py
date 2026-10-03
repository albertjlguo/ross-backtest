"""
从 SEC companyfacts（XBRL）取年度财报，算"第二道门的财务指纹"和"第四道门的算术"。

口径（与 /snowball skill 一致）：
  所有者收益 = 净利润 + 折旧摊销 − 维持性资本开支        （股权激励已在净利润里作为费用扣过）
  维持性资本开支三种估法：
    dep        固定资产折旧（折旧摊销 − 无形资产摊销；拿不到摊销就用全部折旧摊销）
    greenwald  资本开支 − 过去 5 年"固定资产净值/收入"均值 × 当年收入增量（限制在 0 ~ 资本开支之间）
    full       满载折旧 = 固定资产原值 ÷ 使用年限（使用年限 = 历年"原值/折旧"的中位数）
  基准 = max(dep, greenwald)；压力 = max(基准, full)
  投入资本 = 总资产 − 现金及短期投资 −（流动负债 − 短期有息负债）
  ROIC = 营业利润 ×(1 − 21%) ÷ 期初期末平均投入资本
  增量 ROIC（5 年）= Δ税后营业利润 ÷ Δ投入资本
金融类（银行/保险/放贷）不适用以上口径，只给 ROE 和市净率。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TAX = 0.21
DUR = {  # 期间值（利润表/现金流量表），按顺序取第一个有数据的标签，缺的年份用后面的补
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "RevenuesNetOfInterestExpense"],
    "op_income": ["OperatingIncomeLoss"],
    "pretax": ["IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
               "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "da": ["DepreciationDepletionAndAmortization", "DepreciationAmortizationAndAccretionNet",
           "DepreciationAndAmortization", "Depreciation"],
    "amort": ["AmortizationOfIntangibleAssets"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities"],
    "sbc": ["ShareBasedCompensation", "AllocatedShareBasedCompensationExpense"],
    "interest": ["InterestExpense", "InterestExpenseDebt", "InterestExpenseNonoperating"],
    "buyback": ["PaymentsForRepurchaseOfCommonStock"],
    "dividends": ["PaymentsOfDividends", "PaymentsOfDividendsCommonStock"],
    "shares_diluted": ["WeightedAverageNumberOfDilutedSharesOutstanding"],
    "shares_basic": ["WeightedAverageNumberOfSharesOutstandingBasic"],
}
INST = {  # 时点值（资产负债表）
    "assets": ["Assets"],
    "liab_current": ["LiabilitiesCurrent"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue",
             "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "st_inv": ["ShortTermInvestments", "MarketableSecuritiesCurrent", "AvailableForSaleSecuritiesDebtSecuritiesCurrent"],
    "ppe_net": ["PropertyPlantAndEquipmentNet",
                "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAfterAccumulatedDepreciationAndAmortization"],
    "ppe_gross": ["PropertyPlantAndEquipmentGross",
                  "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetBeforeAccumulatedDepreciationAndAmortization"],
    "acc_dep": ["AccumulatedDepreciationDepletionAndAmortizationPropertyPlantAndEquipment",
                "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAccumulatedDepreciationAndAmortization"],
    "debt_lt": ["LongTermDebtNoncurrent", "LongTermDebt", "LongTermDebtAndCapitalLeaseObligations"],
    "debt_st": ["LongTermDebtCurrent", "DebtCurrent", "ShortTermBorrowings"],
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
}
ANNUAL_FORMS = {"10-K", "10-K/A", "20-F", "40-F", "10-KT"}


def _series(facts: dict, tag: str, duration: bool) -> pd.Series:
    """某个标签的年度序列：index = 财年截止日，值 = 最新披露的数。"""
    node = facts.get("us-gaap", {}).get(tag) or facts.get("dei", {}).get(tag)
    if not node:
        return pd.Series(dtype=float)
    rows = []
    for unit, items in node.get("units", {}).items():
        if unit not in ("USD", "shares"):
            continue
        for it in items:
            if it.get("form") not in ANNUAL_FORMS or "end" not in it:
                continue
            if duration:
                if "start" not in it:
                    continue
                days = (pd.Timestamp(it["end"]) - pd.Timestamp(it["start"])).days
                if not 340 <= days <= 380:
                    continue
            elif "start" in it:
                continue
            rows.append((it["end"], it.get("filed", ""), float(it["val"])))
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows, columns=["end", "filed", "val"]).sort_values(["end", "filed"])
    s = df.groupby("end")["val"].last()
    s.index = pd.to_datetime(s.index)
    return s


def _pick(facts: dict, tags: list[str], duration: bool) -> pd.Series:
    out = pd.Series(dtype=float)
    for t in tags:
        s = _series(facts, t, duration)
        out = s if out.empty else out.combine_first(s)
    return out


def annual_table(js: dict) -> pd.DataFrame:
    """companyfacts JSON → 每个财年一行的原始科目表（单位：美元 / 股）。"""
    facts = (js or {}).get("facts", {})
    cols = {k: _pick(facts, v, True) for k, v in DUR.items()}
    cols.update({k: _pick(facts, v, False) for k, v in INST.items()})
    df = pd.DataFrame(cols).sort_index()
    if df.empty or "revenue" not in df:
        return df
    # 只保留财年截止日（以净利润或总资产有数的日期为准），时点值里混入的季度末去掉
    fy = df.index[df["net_income"].notna() | df["revenue"].notna()]
    df = df.loc[fy]
    df.index.name = "fy_end"
    return df


def derive(df: pd.DataFrame, financial: bool = False) -> pd.DataFrame:
    d = df.copy()
    for c in list(DUR) + list(INST):
        if c not in d:
            d[c] = np.nan
    d["shares"] = d["shares_diluted"].fillna(d["shares_basic"])
    d["roe"] = d["net_income"] / d["equity"].rolling(2, min_periods=1).mean()
    if financial:
        return d
    cash = d["cash"].fillna(0) + d["st_inv"].fillna(0)
    d["debt"] = d["debt_lt"].fillna(0) + d["debt_st"].fillna(0)
    d["net_debt"] = d["debt"] - cash
    d["invested_capital"] = d["assets"] - cash - (d["liab_current"] - d["debt_st"].fillna(0))
    d["nopat"] = d["op_income"] * (1 - TAX)
    d["roic"] = d["nopat"] / d["invested_capital"].rolling(2, min_periods=1).mean().where(lambda x: x > 0)
    d["roic_incr_5y"] = (d["nopat"] - d["nopat"].shift(5)) / (d["invested_capital"] - d["invested_capital"].shift(5)).where(lambda x: x > 0)
    d["op_margin"] = d["op_income"] / d["revenue"]
    # ---- 维持性资本开支 ----
    d["dep"] = (d["da"] - d["amort"].fillna(0)).clip(lower=0)
    ratio = (d["ppe_net"] / d["revenue"]).rolling(5, min_periods=3).mean().shift(1)
    growth_capex = (ratio * d["revenue"].diff()).clip(lower=0)
    d["maint_greenwald"] = (d["capex"] - growth_capex).clip(lower=0).where(ratio.notna())
    d["maint_greenwald"] = np.minimum(d["maint_greenwald"], d["capex"])
    gross = d["ppe_gross"].fillna(d["ppe_net"] + d["acc_dep"])
    life = (gross / d["dep"].where(d["dep"] > 0)).expanding(min_periods=3).median()
    d["useful_life"] = life
    d["maint_full"] = gross / life
    d["maint_base"] = d[["dep", "maint_greenwald"]].max(axis=1)
    d["maint_stress"] = d[["maint_base", "maint_full"]].max(axis=1)
    d["owner_earnings"] = d["net_income"] + d["da"] - d["maint_base"]
    d["owner_earnings_stress"] = d["net_income"] + d["da"] - d["maint_stress"]
    d["fcf"] = d["cfo"] - d["capex"]
    d["fcf_ex_sbc"] = d["fcf"] - d["sbc"].fillna(0)
    d["oe_ps"] = d["owner_earnings"] / d["shares"]
    d["capex_to_dep"] = d["capex"] / d["dep"].where(d["dep"] > 0)
    d["interest_cover"] = d["op_income"] / d["interest"].where(d["interest"] > 0)
    return d


def cagr(s: pd.Series, years: int) -> float:
    s = s.dropna()
    if len(s) <= years or s.iloc[-1 - years] <= 0 or s.iloc[-1] <= 0:
        return np.nan
    return float((s.iloc[-1] / s.iloc[-1 - years]) ** (1 / years) - 1)


def dcf_value(oe: float, g: float, r: float, years: int = 10, g_term: float = 0.025) -> float:
    """前 years 年按 g 增长，之后按 g_term 永续；折现率 r。"""
    if oe <= 0 or r <= g_term:
        return np.nan
    pv, cf = 0.0, oe
    for t in range(1, years + 1):
        cf *= 1 + g
        pv += cf / (1 + r) ** t
    return pv + cf * (1 + g_term) / (r - g_term) / (1 + r) ** years


def implied_growth(mcap: float, oe: float, r: float, years: int = 10, g_term: float = 0.025) -> float:
    """现价隐含的未来 10 年所有者收益年增速。"""
    if not (oe > 0 and mcap > 0):
        return np.nan
    lo, hi = -0.5, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if dcf_value(oe, mid, r, years, g_term) < mcap:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
