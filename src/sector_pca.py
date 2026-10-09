"""PCA on the Indeed sector panel: isolate the shock common to every sector.

The panel is built as sectors (rows) x months (columns), each sector standardised over
time. PCA is fit on its transpose (months as observations), so the only centring is
within each sector's own series, which standardisation has already done. Fitting on the
sectors x months orientation instead would subtract the cross-sector mean at every date,
which is exactly the common movement PC1 is meant to capture.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent.parent
SECTOR_CSV = ROOT / "data" / "processed" / "indeed_sector.csv"


def sector_matrix(df: pd.DataFrame, variable: str = "total postings"
                  ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Monthly-mean index per sector, and the same matrix standardised per sector.

    Returns (raw sectors x months, standardised sectors x months, sector_group by sector).
    """
    d = df[df["variable"] == variable].copy()
    d["month"] = pd.to_datetime(d["date"]).dt.to_period("M").dt.to_timestamp()
    raw = d.pivot_table(index="display_name", columns="month",
                        values="indeed_job_postings_index", aggfunc="mean")
    z = raw.sub(raw.mean(axis=1), axis=0).div(raw.std(axis=1), axis=0)
    groups = d.groupby("display_name")["sector_group"].first().reindex(raw.index)
    return raw, z, groups


def fit_sector_pca(z: pd.DataFrame) -> tuple[PCA, pd.DataFrame, pd.DataFrame]:
    """Fit PCA with months as observations.

    Returns the model, the component time profiles (months x PCs) and each sector's
    loading (sectors x PCs, the correlation between the sector series and the PC).
    PC signs are oriented so the average sector loading is positive.
    """
    pca = PCA().fit(z.T)
    scores = pca.transform(z.T)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    sign = np.where(loadings.mean(axis=0) < 0, -1, 1)
    pca.components_ *= sign[:, None]
    cols = [f"PC{i + 1}" for i in range(pca.n_components_)]
    profile = pd.DataFrame(scores * sign, index=z.columns, columns=cols)
    loadings = pd.DataFrame(loadings * sign, index=z.index, columns=cols)
    return pca, profile, loadings


def remove_pc1(z: pd.DataFrame, pca: PCA, profile: pd.DataFrame) -> pd.DataFrame:
    """Each standardised sector series minus its PC1 reconstruction (sectors x months)."""
    pc1_part = np.outer(profile["PC1"], pca.components_[0]) + pca.mean_
    return z - pd.DataFrame(pc1_part, index=z.columns, columns=z.index).T


def break_table(resid: pd.DataFrame, groups: pd.Series,
                at: str = "2022-12-01") -> pd.DataFrame:
    """Mean residual before and after a date, the shift in pre-period SDs, and its rank."""
    pre, post = resid.loc[:, resid.columns < at], resid.loc[:, resid.columns >= at]
    t = pd.DataFrame({"sector_group": groups,
                      "mean_pre": pre.mean(axis=1), "mean_post": post.mean(axis=1)})
    t["shift"] = t["mean_post"] - t["mean_pre"]
    t["shift_in_pre_sd"] = t["shift"] / pre.std(axis=1)
    t["rank_abs_shift"] = t["shift_in_pre_sd"].abs().rank(ascending=False).astype(int)
    return t.sort_values("rank_abs_shift")
