"""Readers for the later held-out years (protocols/LATER_YEARS_v001.md): Kelmarsh 2024, Penmanshiel 2022 (Senvion).
The newer SCADA exports can repeat a timestamp many times with mostly empty rows; per timestamp and channel the first
non-empty value is kept and the frame is re-indexed to the full 10-min grid of the year. Nothing else is changed.
read(ds, t, cols) returns (df, month) with the same column names and row layout as the earlier readers."""
from latetruth.config import DATA_DIR, RAW_DIR as OUT_DIR, REPO_DIR, PAPER_DIR
from latetruth.data import find_archive
import sys
import io, csv, zipfile
import numpy as np, pandas as pd
from latetruth.run_local_late_truth_long import ROOT
LATER = {'kelmarsh2024': ('Kelmarsh', 2024), 'penmanshiel2022': ('Penmanshiel', 2022),
         'kelmarsh2020': ('Kelmarsh', 2020), 'penmanshiel2020': ('Penmanshiel', 2020)}   # + CONFIRM_TWO_TIMESCALE_v001


def zip_and_prefix(ds, t):
    farm, year = LATER[ds]; base = DATA_DIR/farm
    if farm == 'Kelmarsh':
        return find_archive(base, f'Kelmarsh_SCADA_{year}_*.zip'), f'Turbine_Data_Kelmarsh_{t}_'
    group = 'WT01-10' if t <= 10 else 'WT11-15'
    return find_archive(base, f'Penmanshiel_SCADA_{year}_{group}_*.zip', recursive=True), f'Turbine_Data_Penmanshiel_{t:02d}_'


def read(ds, t, cols):
    zp, pre = zip_and_prefix(ds, t)
    with zipfile.ZipFile(zp) as z:
        name = next(n for n in z.namelist() if n.startswith(pre))
        with io.TextIOWrapper(z.open(name), encoding='utf-8-sig') as f:
            for line in f:
                if line.startswith('# Date and time'): h = next(csv.reader([line[2:].strip()])); break
            df = pd.read_csv(f, names=h, header=None, usecols=[h[0]] + list(dict.fromkeys(cols)))
    ts = pd.to_datetime(df[h[0]]); df = df.drop(columns=h[0]).groupby(ts.values).first()   # first non-empty value per timestamp
    year = LATER[ds][1]; grid = pd.date_range(f'{year}-01-01 00:00', f'{year}-12-31 23:50', freq='10min')
    df = df.reindex(grid); df.insert(0, h[0], grid)
    return df.reset_index(drop=True), grid.month.to_numpy()
