"""Design study, last step after the revision: L6 = L5 + recency term (Amendment 18, chosen on this year), Kelmarsh 2017, own rows,
same metric as DESIGN_LADDER_kelmarsh2017.json (checked: L5 reproduced). Writes DESIGN_LADDER_L6_kelmarsh2017.json."""
from latetruth.config import DATA_DIR, RAW_DIR as OUT_DIR, REPO_DIR, PAPER_DIR
import sys
import json; import hashlib; import numpy as np; from latetruth import baselines_classic as C
O = C.OUT; L = json.load(open(O/'DESIGN_LADDER_kelmarsh2017.json'))['results']
z = np.load(O/'ours_kelmarsh2017.npz', allow_pickle=False); r = np.load(O/'rec_kelmarsh2017.npz', allow_pickle=False); T = sorted(int(k.split('_')[1]) for k in z.files if k.startswith('Y_'))
Y = {t: z[f'Y_{t}'] for t in T}; OK = {t: z[f'OK_{t}'] for t in T}; m = z['month']
assert abs(C.score(Y, {t: z[f'P_ours_{t}'] for t in T}, OK, m)['all'] - L['L5 = L4 + local late-truth correction']['all']) < 1e-9
out = {'L6 = L5 + recency term (two time scales)': C.score(Y, {t: r[f'P_{t}'] for t in T}, OK, m), 'code_sha256': hashlib.sha256(open(__file__, 'rb').read()).hexdigest()}
(O/'DESIGN_LADDER_L6_kelmarsh2017.json').write_text(json.dumps(out, indent=2)); print(out)
