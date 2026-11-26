"""Features from listing descriptions (PropTx PublicRemarks).

Two kinds:

* Updates a homeowner can tell us: roof, windows, furnace, A/C, water heater
  (owned or rented), kitchen, bathrooms, basement, flooring. Each gets a
  "mentioned as new/updated" flag and, when the description gives a year
  ("new roof 2021", "furnace & A/C (2019)"), its age at the sale. These are
  the same questions the public tool can ask, so the model learns what each
  answer is worth.
* A compact summary of the wording (TF-IDF, reduced to TEXT_DIMS numbers),
  for research only: a homeowner has no listing description.

Prices and other numbers are stripped first so a description can't leak the
list price ("priced to sell at $649,900").
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

TEXT_DIMS = 24
NEW = r"(?:new|newer|newly|updated|upgraded|replaced|renovated|remodel+ed|redone|installed|brand new)"

# name -> regex for the item (matched within a short window of a NEW word or a year)
ITEMS = {
    "roof": r"roof|shingles",
    "windows": r"windows?",
    "furnace": r"furnace|heat pump|hvac",
    "ac": r"a/?c\b|air condition\w*|central air",
    "hwt": r"hwt|hot water (?:tank|heater)|water heater|on[- ]demand",
    "kitchen": r"kitchen",
    "bath": r"bath(?:room)?s?|ensuite",
    "basement": r"basement",
    "flooring": r"floor(?:ing|s)|hardwood|laminate|lvp",
}
CONDITION = {
    "tlc": r"\btlc\b|handyman|needs? (?:some )?(?:work|updating|love)|fixer|as[- ]is|investor|contractor",
    "turnkey": r"turn[- ]?key|move[- ]in ready|meticulous|pristine|immaculate|shows? (?:like new|beautifully)",
    "hwt_rental": r"(?:hwt|hot water (?:tank|heater)|water heater)[^.]{0,30}\b(?:rent(?:al|ed)?|lease[d]?)\b"
                  r"|\b(?:rent(?:al|ed)?)\b[^.]{0,15}(?:hwt|hot water (?:tank|heater)|water heater)",
    "hwt_owned": r"(?:hwt|hot water (?:tank|heater)|water heater)[^.]{0,30}\bowned\b"
                 r"|\bowned\b[^.]{0,15}(?:hwt|hot water (?:tank|heater)|water heater)",
}
YEAR = r"(?:19[89]\d|20[0-4]\d)"
GAP = r"[^\w.!?;]+"  # between words, never across a sentence


def _strip_prices(s: str) -> str:
    s = re.sub(r"\$\s?[\d,.]+\s?[km]?", " ", s, flags=re.I)
    return re.sub(r"\b\d[\d,.]*\s?(?:k|m|million|thousand)\b", " ", s, flags=re.I)


def update_features(remarks: pd.Series, sale_year: pd.Series) -> pd.DataFrame:
    """Flags and ages for updates mentioned in the description."""
    text = remarks.fillna("").str.lower().map(_strip_prices)
    out = pd.DataFrame(index=remarks.index)
    for name, item in ITEMS.items():
        near_new = rf"(?:{NEW}{GAP}(?:\w+{GAP}){{0,3}}(?:{item}))|(?:(?:{item}){GAP}(?:\w+{GAP}){{0,3}}{NEW})"
        out[f"upd_{name}"] = text.str.contains(near_new, regex=True).astype(float)
        # "roof (2019)", "roof 2019", "roof replaced in 2019"; then "2019 roof"
        after = text.str.extract(rf"(?:{item}){GAP}(?:\w+{GAP}){{0,3}}?({YEAR})", expand=False)
        before = text.str.extract(rf"({YEAR}){GAP}(?:\w+{GAP}){{0,2}}?(?:{item})", expand=False)
        year = pd.to_numeric(after.fillna(before), errors="coerce")
        age = sale_year - year
        out[f"age_{name}"] = age.where((age >= 0) & (age <= 40))
        out.loc[out[f"age_{name}"].notna() & (out[f"age_{name}"] <= 10), f"upd_{name}"] = 1.0
    for name, rx in CONDITION.items():
        out[f"txt_{name}"] = text.str.contains(rx, regex=True).astype(float)
    out.loc[remarks.isna(), :] = np.nan
    return out


def summary_features(remarks: pd.Series, dims: int = TEXT_DIMS) -> pd.DataFrame:
    """TF-IDF of the wording, reduced with SVD. Unsupervised: never sees prices."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    text = remarks.fillna("").str.lower().map(_strip_prices).str.replace(r"\d+", " ", regex=True)
    has = text.str.strip() != ""
    out = pd.DataFrame(np.nan, index=remarks.index, columns=[f"txt_svd_{i}" for i in range(dims)])
    if has.sum() < dims * 5:
        return out
    tf = TfidfVectorizer(ngram_range=(1, 2), min_df=20, max_df=0.5, sublinear_tf=True, stop_words="english")
    X = tf.fit_transform(text[has])
    k = min(dims, X.shape[1] - 1)
    out.loc[has, out.columns[:k]] = TruncatedSVD(k, random_state=42).fit_transform(X)
    return out


UPDATE_COLS = [f"upd_{k}" for k in ITEMS] + [f"age_{k}" for k in ITEMS] + [f"txt_{k}" for k in CONDITION]
SUMMARY_COLS = [f"txt_svd_{i}" for i in range(TEXT_DIMS)]
