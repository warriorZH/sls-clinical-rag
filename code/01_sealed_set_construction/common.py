"""Shared helpers: robust CSV I/O for Excel/WPS-edited files and agreement statistics.

Only the Python standard library is used, so the scripts run on any machine
with Python 3.8+.
"""
import csv, io, math, random
from collections import Counter, defaultdict
from pathlib import Path

ENCODINGS = ('utf-8-sig', 'gb18030', 'utf-16')

# The expert_testset_v1 folder (parent of scripts/). Used as the default root so the
# scripts work no matter which directory they are started from.
PKG_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(p, root=PKG_ROOT):
    """Absolute path for p: absolute or existing relative to the current directory as given,
    otherwise relative to the expert_testset_v1 folder."""
    p = Path(p)
    if p.is_absolute() or p.exists():
        return p.resolve()
    return (Path(root) / p).resolve()


def csv_listing(folder):
    """Short description of the CSV files in a folder, for error messages."""
    folder = Path(folder)
    if not folder.is_dir():
        return f'文件夹不存在：{folder}'
    names = sorted(p.name for p in folder.glob('*.csv'))
    return f'文件夹 {folder} 中的 CSV：' + ('、'.join(names) if names else '（没有）')


def read_csv(path):
    """Read a CSV saved by Excel/WPS (UTF-8 with or without BOM, GBK/GB18030 or UTF-16)."""
    raw = Path(path).read_bytes()
    last = None
    for enc in ENCODINGS:
        try:
            text = raw.decode(enc)
        except UnicodeDecodeError as e:
            last = e
            continue
        dialect = csv.excel_tab if text.split('\n', 1)[0].count('\t') > text.split('\n', 1)[0].count(',') else csv.excel
        rows = list(csv.DictReader(io.StringIO(text), dialect=dialect))
        return [{(k or '').strip(): (v or '').strip() for k, v in r.items()} for r in rows]
    raise ValueError(f'无法识别文件编码: {path} ({last})')


def write_csv(path, rows, fieldnames=None):
    """Write UTF-8 with BOM so that Excel/WPS shows Chinese correctly."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)


def find_col(row_or_header, prefix):
    """Find a column by its leading text, so small edits to the header do not break parsing."""
    keys = row_or_header.keys() if isinstance(row_or_header, dict) else row_or_header
    for k in keys:
        if k.replace('【填】', '').startswith(prefix):
            return k
    raise KeyError(prefix)


def norm_int(value, allowed):
    """Parse an annotator cell; returns None for blank/invalid values."""
    v = (value or '').strip().replace('，', ',').upper()
    if v in ('', 'NA', 'N/A', '-'):
        return None
    try:
        x = int(float(v))
    except ValueError:
        return None
    return x if x in allowed else None


# ---------------------------------------------------------------- agreement

def cohen_kappa(a, b, weights=None, categories=None):
    """Cohen's kappa for two raters. weights: None (nominal) or 'linear'."""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return float('nan')
    cats = sorted(set(categories or []) | {x for x, _ in pairs} | {y for _, y in pairs})
    idx = {c: i for i, c in enumerate(cats)}
    k = len(cats)
    if k == 1:
        return 1.0
    n = len(pairs)
    obs = [[0.0] * k for _ in range(k)]
    for x, y in pairs:
        obs[idx[x]][idx[y]] += 1
    ra = [sum(row) / n for row in obs]
    rb = [sum(obs[i][j] for i in range(k)) / n for j in range(k)]
    if weights == 'linear':
        w = [[abs(i - j) / (k - 1) for j in range(k)] for i in range(k)]
    else:
        w = [[0.0 if i == j else 1.0 for j in range(k)] for i in range(k)]
    po = sum(w[i][j] * obs[i][j] / n for i in range(k) for j in range(k))
    pe = sum(w[i][j] * ra[i] * rb[j] for i in range(k) for j in range(k))
    if pe == 0:
        return 1.0 if po == 0 else float('nan')
    return 1 - po / pe


def fleiss_kappa(units):
    """units: list of lists of category labels (same number of raters per unit, no None)."""
    units = [u for u in units if all(x is not None for x in u)]
    if not units:
        return float('nan')
    m = len(units[0])
    units = [u for u in units if len(u) == m]
    n = len(units)
    cats = sorted({x for u in units for x in u})
    if m < 2 or len(cats) < 2:
        return 1.0 if len(cats) == 1 else float('nan')
    p_j = {c: sum(u.count(c) for u in units) / (n * m) for c in cats}
    p_i = [(sum(u.count(c) ** 2 for c in cats) - m) / (m * (m - 1)) for u in units]
    pbar = sum(p_i) / n
    pe = sum(v ** 2 for v in p_j.values())
    return (pbar - pe) / (1 - pe) if pe < 1 else 1.0


def krippendorff_alpha(units, level='nominal'):
    """units: list of lists of values (None = missing); any number of raters.
    level: 'nominal' or 'ordinal' (values must be orderable)."""
    units = [[v for v in u if v is not None] for u in units]
    units = [u for u in units if len(u) >= 2]
    if not units:
        return float('nan')
    values = sorted({v for u in units for v in u})
    if len(values) == 1:
        return 1.0
    o = defaultdict(float)  # coincidence matrix
    for u in units:
        m = len(u)
        cnt = Counter(u)
        for c in cnt:
            for k in cnt:
                pairs = cnt[c] * (cnt[k] - (1 if c == k else 0))
                if pairs:
                    o[(c, k)] += pairs / (m - 1)
    n_c = {c: sum(o[(c, k)] for k in values) for c in values}
    n = sum(n_c.values())

    def delta(c, k):
        if level == 'nominal':
            return 0.0 if c == k else 1.0
        lo, hi = (c, k) if values.index(c) <= values.index(k) else (k, c)
        between = [g for g in values if values.index(lo) <= values.index(g) <= values.index(hi)]
        s = sum(n_c[g] for g in between) - (n_c[lo] + n_c[hi]) / 2
        return s ** 2

    d_o = sum(o[(c, k)] * delta(c, k) for c in values for k in values) / n
    d_e = sum(n_c[c] * n_c[k] * delta(c, k) for c in values for k in values) / (n * (n - 1))
    return 1 - d_o / d_e if d_e > 0 else 1.0


def kappa_from_confusion(conf, cats, weights=None):
    """conf: dict (x, y) -> count; same formula as cohen_kappa but from aggregated counts."""
    k = len(cats)
    n = sum(conf.values())
    if n == 0:
        return float('nan')
    if k == 1:
        return 1.0
    idx = {c: i for i, c in enumerate(cats)}
    ra, rb = [0.0] * k, [0.0] * k
    for (x, y), v in conf.items():
        ra[idx[x]] += v / n
        rb[idx[y]] += v / n
    if weights == 'linear':
        w = lambda i, j: abs(i - j) / (k - 1)
    else:
        w = lambda i, j: 0.0 if i == j else 1.0
    po = sum(w(idx[x], idx[y]) * v / n for (x, y), v in conf.items())
    pe = sum(w(i, j) * ra[i] * rb[j] for i in range(k) for j in range(k))
    if pe == 0:
        return 1.0 if po == 0 else float('nan')
    return 1 - po / pe


def coincidence(units):
    """Krippendorff coincidence contributions of a list of units (lists of values, None = missing)."""
    o = defaultdict(float)
    for u in units:
        u = [v for v in u if v is not None]
        m = len(u)
        if m < 2:
            continue
        cnt = Counter(u)
        for c in cnt:
            for k in cnt:
                pairs = cnt[c] * (cnt[k] - (1 if c == k else 0))
                if pairs:
                    o[(c, k)] += pairs / (m - 1)
    return o


def alpha_from_coincidence(o, level='nominal'):
    values = sorted({c for c, _ in o} | {k for _, k in o})
    if not values:
        return float('nan')
    if len(values) == 1:
        return 1.0
    n_c = {c: sum(o.get((c, k), 0.0) for k in values) for c in values}
    n = sum(n_c.values())
    pos = {v: i for i, v in enumerate(values)}

    def delta(c, k):
        if level == 'nominal':
            return 0.0 if c == k else 1.0
        lo, hi = sorted((pos[c], pos[k]))
        s = sum(n_c[values[g]] for g in range(lo, hi + 1)) - (n_c[values[lo]] + n_c[values[hi]]) / 2
        return s ** 2

    d_o = sum(v * delta(c, k) for (c, k), v in o.items()) / n
    d_e = sum(n_c[c] * n_c[k] * delta(c, k) for c in values for k in values) / (n * (n - 1))
    return 1 - d_o / d_e if d_e > 0 else 1.0


def bootstrap_sum(per_cluster, stat, reps=2000, seed=20261024, alpha=0.05):
    """per_cluster: dict cluster -> aggregated dict (counts); stat: function(summed dict) -> float.
    Whole questions are resampled; their aggregated counts are summed (fast)."""
    keys = list(per_cluster)
    rng = random.Random(seed)
    vals = []
    for _ in range(reps):
        total = defaultdict(float)
        for k in (rng.choice(keys) for _ in keys):
            for cell, v in per_cluster[k].items():
                total[cell] += v
        v = stat(total)
        if not (isinstance(v, float) and math.isnan(v)):
            vals.append(v)
    if not vals:
        return (float('nan'), float('nan'))
    vals.sort()
    return (vals[int(len(vals) * alpha / 2)], vals[min(len(vals) - 1, int(len(vals) * (1 - alpha / 2)))])


def percent_agreement(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    return sum(x == y for x, y in pairs) / len(pairs) if pairs else float('nan')


def cluster_bootstrap(clusters, stat, reps=2000, seed=20261024, alpha=0.05):
    """clusters: dict cluster_id -> list of units; stat: function(list_of_units) -> float.
    Resamples whole questions so that correlated units stay together."""
    keys = list(clusters)
    rng = random.Random(seed)
    vals = []
    for _ in range(reps):
        sample = [u for k in (rng.choice(keys) for _ in keys) for u in clusters[k]]
        v = stat(sample)
        if not (isinstance(v, float) and math.isnan(v)):
            vals.append(v)
    if not vals:
        return (float('nan'), float('nan'))
    vals.sort()
    lo = vals[int(len(vals) * alpha / 2)]
    hi = vals[min(len(vals) - 1, int(len(vals) * (1 - alpha / 2)))]
    return (lo, hi)


def landis_koch(k):
    if k != k:
        return '无法计算'
    if k < 0:
        return '差于随机'
    if k <= 0.20:
        return '轻微'
    if k <= 0.40:
        return '一般'
    if k <= 0.60:
        return '中等'
    if k <= 0.80:
        return '较强'
    return '几乎完全一致'
