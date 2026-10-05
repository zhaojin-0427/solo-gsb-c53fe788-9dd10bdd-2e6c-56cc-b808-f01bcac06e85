"""Chromatographic quantification solver.

Given up to six pure-standard time/intensity templates (with known positive
concentrations), numbered internal-standard retention times in the reference
channel, unnumbered candidate internal-standard times in the sample channel,
and the quantitative sample trace, this module:

1. Enumerates every order-preserving, non-reusing internal-standard anchor
   matching whose consecutive piecewise-linear timing-correction slopes lie
   inside the allowed drift range (missing internal standards / distractor
   peaks are simply left unpaired).
2. For each candidate matching, maps every template onto the sample time grid
   through the shared piecewise-linear time calibration and checks, by linear
   programming, whether non-negative mixture coefficients exist such that
     * every unsaturated sample point is reproduced within ``tolerance``;
     * every saturated sample point reaches ``saturation - tolerance``.
3. Maximises the number of feasible anchor pairs first; over *all* tied
   optimal explanations it then reports certified per-component concentration
   lower/upper bounds (``None`` means +infinity) together with anchor evidence.

All results are exact with respect to the stated constraints: bounds come from
LP extrema and every tied explanation is enumerated.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, diags, vstack


# Tuned for the online target size (12 refs / 24 candidates / 8192 points).
REDUCED_MAX_POINTS = 24          # subsampled points kept on a pruned segment
FULL_CACHE_SIZE = 1024           # LRU capacity for full segment blocks
LP_TIME_LIMIT = 2.0              # seconds per LP call (safety net)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def _finite_array(x, name, min_len=0):
    a = np.asarray(x, dtype=float)
    if a.ndim != 1 or a.size < min_len:
        raise ValueError(f"{name} must be a 1-D array of length >= {min_len}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return a


def validate_payload(payload: dict):
    """Validate the request payload. Raises ValueError on invalid input."""
    standards = payload.get("standards")
    if not isinstance(standards, list) or not (1 <= len(standards) <= 6):
        raise ValueError("standards must be a non-empty list of at most 6")

    temps = []
    concs = []
    for idx, std in enumerate(standards):
        if not isinstance(std, dict):
            raise ValueError(f"standards[{idx}] must be an object")
        t = _finite_array(std.get("times"), f"standards[{idx}].times", 2)
        y = _finite_array(std.get("intensities"),
                          f"standards[{idx}].intensities", 2)
        if t.size != y.size:
            raise ValueError(
                f"standards[{idx}] times/intensities length mismatch")
        if np.any(np.diff(t) <= 0):
            raise ValueError(
                f"standards[{idx}].times must be strictly increasing")
        if np.any(y < 0):
            raise ValueError(f"standards[{idx}].intensities must be >= 0")
        c = std.get("concentration")
        if not isinstance(c, (int, float)) or not math.isfinite(c) or c <= 0:
            raise ValueError(
                f"standards[{idx}].concentration must be a positive finite number")
        temps.append((t, y))
        concs.append(float(c))

    ref = _finite_array(payload.get("reference_times"), "reference_times", 1)
    if np.any(np.diff(ref) <= 0):
        raise ValueError("reference_times must be strictly increasing "
                         "(internal standards are numbered)")

    cand = _finite_array(payload.get("candidate_times"), "candidate_times", 1)
    if np.any(np.diff(cand) < 0):
        raise ValueError("candidate_times must be non-decreasing")

    st = _finite_array(payload.get("sample_times"), "sample_times", 2)
    si = _finite_array(payload.get("sample_intensities"),
                       "sample_intensities", 2)
    if st.size != si.size:
        raise ValueError("sample_times/sample_intensities length mismatch")
    if np.any(np.diff(st) <= 0):
        raise ValueError("sample_times must be strictly increasing")
    if np.any(si < 0):
        raise ValueError("sample_intensities must be >= 0")

    sat = payload.get("saturation")
    tol = payload.get("tolerance")
    if not isinstance(sat, (int, float)) or not math.isfinite(sat) or sat <= 0:
        raise ValueError("saturation must be a positive finite number")
    if not isinstance(tol, (int, float)) or not math.isfinite(tol) or tol <= 0:
        raise ValueError("tolerance must be a positive finite number")

    slope = payload.get("slope_range")
    if (not isinstance(slope, (list, tuple)) or len(slope) != 2 or
            not all(isinstance(v, (int, float)) for v in slope)):
        raise ValueError("slope_range must be [low, high]")
    slo, shi = float(slope[0]), float(slope[1])
    if not (math.isfinite(slo) and math.isfinite(shi)):
        raise ValueError("slope_range must be finite")
    if slo <= 0 or shi < slo:
        raise ValueError("slope_range requires 0 < low <= high")

    return {
        "templates": temps,
        "concentrations": np.array(concs, dtype=float),
        "ref": ref,
        "cand": cand,
        "st": st,
        "si": si,
        "saturation": float(sat),
        "tolerance": float(tol),
        "slope_lo": slo,
        "slope_hi": shi,
    }


# --------------------------------------------------------------------------- #
# LP construction
# --------------------------------------------------------------------------- #
@dataclass
class Block:
    """A subset of residual constraints: A @ alpha <= b (or hard-infeasible)."""
    A: Optional[csr_matrix]
    b: Optional[np.ndarray]
    infeasible: bool = False


def _build_block(idx_lo, idx_hi, beta0, beta1, seg_t, seg_y,
                 templates, sat, tol, point_indices, K):
    """Build constraint rows for sample points ``point_indices`` of a segment.

    phi(t) = beta0 + beta1 * t maps sample time -> reference time.
    Unsaturated point y:   y - P.alpha <= tol  and  P.alpha - y <= tol
    Saturated point y:     P.alpha >= sat - tol
    Rows trivially satisfied for all alpha >= 0 are dropped; a row of the form
    ``0 <= negative`` marks the block as hard infeasible.
    """
    if point_indices.size == 0:
        return Block(csr_matrix((0, K)), np.zeros(0))

    tseg = seg_t[point_indices]
    yseg = seg_y[point_indices]
    phi = beta0 + beta1 * tseg
    n = point_indices.size
    P = np.zeros((n, K), dtype=float)
    for k, (tt, yy) in enumerate(templates):
        m = (phi >= tt[0]) & (phi <= tt[-1])
        if np.any(m):
            P[m, k] = np.interp(phi[m], tt, yy)

    satmask = yseg >= sat
    # Lower residual rows: -P.alpha <= rhs
    lower_keep = satmask | (yseg > tol)
    lower_rhs = np.where(satmask, tol - sat, tol - yseg)
    # Upper residual rows (unsaturated only): P.alpha <= tol + y
    upper_keep = (~satmask)

    rows, cols, vals, rhs = [], [], [], []
    row_id = 0

    def emit(coef, r):
        nonlocal row_id
        nz = np.nonzero(coef)[0]
        if nz.size == 0:
            if r < -1e-12:
                return None  # signal hard infeasibility
        else:
            for c in nz:
                rows.append(row_id)
                cols.append(int(c))
                vals.append(coef[c])
            rhs.append(r)
        row_id += 1
        return True

    for i in range(n):
        if lower_keep[i]:
            if emit(-P[i], lower_rhs[i]) is None:
                return Block(None, None, infeasible=True)
        if upper_keep[i]:
            if emit(P[i], tol + yseg[i]) is None:
                return Block(None, None, infeasible=True)

    if not rhs:
        return Block(csr_matrix((0, K)), np.zeros(0))
    A = csr_matrix(
        (np.asarray(vals, dtype=float),
         (np.asarray(rows), np.asarray(cols))),
        shape=(row_id, K),
    )
    # Drop rows that carry no coefficients (those were trivially feasible).
    keep = A.getnnz(axis=1) > 0
    A = A[np.asarray(keep).ravel()]
    return Block(A, np.asarray(rhs, dtype=float))


def _reduced_indices(idx_lo, idx_hi, seg_y, sat, tol):
    """Representative point subset (a relaxation of the full segment).

    Keeps (a) a uniform stride subsample and (b) every local maximum among
    points whose lower residual can bind (y > tol or saturated), plateau
    edges included.  This is a sound filter against an unexplainable spike:
    a mixture of non-negative templates bounded everywhere except a peak
    must fail at that peak, and a sharp spike is itself a local maximum;
    broad excess is caught by the uniform subsample.
    """
    length = idx_hi - idx_lo
    if length <= REDUCED_MAX_POINTS:
        return np.arange(idx_lo, idx_hi)
    stride = int(math.ceil(length / REDUCED_MAX_POINTS))
    sub = np.arange(idx_lo, idx_hi, stride)
    ys = seg_y[idx_lo:idx_hi]
    binding = (ys >= sat) | (ys > tol)
    extra = [idx_lo + j for j in range(0, length, stride) if binding[j]]
    bind_idx = np.nonzero(binding)[0]
    if bind_idx.size:
        runs = np.split(bind_idx, np.where(np.diff(bind_idx) > 1)[0] + 1)
        for run in runs:
            # apex + both plateau/run edges: catches sharp spikes and
            # plateau-level saturation requirements
            vals = ys[run]
            locs = run[np.flatnonzero(vals == vals.max())]
            extra.append(run[0] + idx_lo)
            extra.append(run[-1] + idx_lo)
            extra.append(locs[0] + idx_lo)
    return np.unique(np.concatenate(
        [sub, np.asarray(sorted(set(extra)), dtype=int)]))


def _assemble(blocks, K):
    """Stack blocks into scaled (A, b); detect hard-infeasible zero rows."""
    mats, rhss = [], []
    for blk in blocks:
        if blk.infeasible:
            return None, None, True
        if blk.A is not None and blk.A.shape[0] > 0:
            mats.append(blk.A)
            rhss.append(blk.b)
    if mats:
        A = vstack(mats, format="csr")
        b = np.concatenate(rhss)
        zero_row = np.asarray(A.getnnz(axis=1) == 0).ravel()
        if np.any(zero_row & (b < -1e-12)):
            return None, None, True
        keep = ~zero_row
        A, b = A[keep], b[keep]
        # Exact row scaling: identical feasible set, well-conditioned rows
        # (gaussian tails otherwise give 1e-300 coefficients next to O(1)).
        # Subnormal/underflowed coefficients are numerically zero; drop them
        # before computing the scale to avoid inf/NaN rows.
        A.data = np.where(np.abs(A.data) < 1e-300, 0.0, A.data)
        A.eliminate_zeros()
        scale = np.abs(A).toarray().max(axis=1)
        pos = scale > 0
        A = A[pos]
        b = b[pos]
        scale = scale[pos]
        A = (diags(1.0 / scale) @ A).tocsr()
        A.eliminate_zeros()
        b = b / scale
    else:
        A = csr_matrix((0, K))
        b = np.zeros(0)
    return A, b, False


try:
    import highspy as _highspy
    _HAVE_HIGHSPY = True
except Exception:  # pragma: no cover - scipy fallback always present
    _highspy = None
    _HAVE_HIGHSPY = False


# Native-HiGHS models cached by the identity tuple of their blocks.  A fresh
# cache is used per quantification run, so block ids never collide across
# requests and the up-to-12 concentration extrema of one explanation reuse a
# single built constraint matrix.
_HIGHS_CACHE: "OrderedDict[tuple, dict]" = OrderedDict()
_HIGHS_CACHE_MAX = 4096

_H_STATUS_OPTIMAL = 7
_H_STATUS_INFEASIBLE = 8
_H_STATUS_UNBOUNDED = 9


def _highs_model(blocks, K, A, b):
    # Keyed by the exact constraint content (not the transient blocks list):
    # interior blocks are shared between prefix and leaf calls, which carry
    # different constraint sets, so object identity alone would alias models.
    key = (A.shape[0], A.nnz,
           hash((A.data.tobytes(), A.indices.tobytes(),
                 A.indptr.tobytes(), b.tobytes())))
    model = _HIGHS_CACHE.get(key)
    if model is not None:
        _HIGHS_CACHE.move_to_end(key)
        return model
    INF = _highspy.kHighsInf
    h = _highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("presolve", "on")
    h.addVars(K, np.zeros(K), np.full(K, INF))
    for r in range(A.shape[0]):
        sl, sr = int(A.indptr[r]), int(A.indptr[r + 1])
        h.addRow(-INF, float(b[r]), sr - sl,
                 A.indices[sl:sr].astype(np.int32),
                 A.data[sl:sr].astype(np.float64))
    model = {"h": h}
    if len(_HIGHS_CACHE) >= _HIGHS_CACHE_MAX:
        _HIGHS_CACHE.popitem(last=False)
    _HIGHS_CACHE[key] = model
    return model


def lp_solve(blocks, K, sign=None, k=None):
    """Solve A.alpha <= b, alpha >= 0.

    sign/k given -> extremise sign*alpha_k (+1 min, -1 max): (status, value).
    sign None    -> feasibility only: (bool, None).
    """
    A, b, bad = _assemble(blocks, K)
    if bad:
        return (False, None) if sign is None else (2, None)

    if _HAVE_HIGHSPY:
        m = _highs_model(blocks, K, A, b)
        h = m["h"]
        idx = np.arange(K, dtype=np.int32)
        if sign is None:
            h.changeColsCost(K, idx, np.zeros(K))
            h.run()
            return h.getModelStatus().value == _H_STATUS_OPTIMAL, None
        cost = np.zeros(K)
        cost[k] = float(sign)
        h.changeColsCost(K, idx, cost)
        h.run()
        status = h.getModelStatus().value
        if status == _H_STATUS_OPTIMAL:
            # cost = sign*x_k (sign=-1 for max, +1 for min); the requested
            # extremal value of x_k is simply its primal column value.
            val = float(h.getSolution().col_value[k])
            return 0, max(0.0, val)
        return (_H_STATUS_UNBOUNDED if status == _H_STATUS_UNBOUNDED else
                (2 if status == _H_STATUS_INFEASIBLE else status)), None

    c = np.zeros(K)
    if sign is not None:
        c[k] = float(sign)
    res = linprog(c, A_ub=A, b_ub=b, bounds=[(0, None)] * K,
                  method="highs",
                  options={"presolve": True, "time_limit": LP_TIME_LIMIT})
    if sign is None:
        return res.status == 0, None
    if res.status == 0:
        return 0, max(0.0, float(sign * res.fun))
    return res.status, None  # 2 infeasible, 3 unbounded, else numerical


# --------------------------------------------------------------------------- #
# Main solver
# --------------------------------------------------------------------------- #
@dataclass
class _Ctx:
    templates: list
    concentrations: np.ndarray
    ref: np.ndarray
    cand: np.ndarray
    st: np.ndarray
    si: np.ndarray
    sat: float
    tol: float
    slo: float
    shi: float
    K: int
    # node arrays
    ni: np.ndarray = field(default=None)   # reference index
    nj: np.ndarray = field(default=None)   # candidate index
    cpos: np.ndarray = field(default=None)  # searchsorted-left in sample grid
    succ: list = field(default=None)
    dp: np.ndarray = field(default=None)
    reduced: dict = field(default_factory=dict)
    reduced_ext: dict = field(default_factory=dict)
    full_cache: OrderedDict = field(default_factory=OrderedDict)
    n_nodes: int = 0


def quantify(payload: dict) -> dict:
    """Top-level entry. Always returns a result dict with ``status``."""
    try:
        v = validate_payload(payload)
    except (ValueError, TypeError, KeyError) as exc:
        return {"status": "invalid_input", "error": str(exc)}

    ctx = _Ctx(
        templates=v["templates"],
        concentrations=v["concentrations"],
        ref=v["ref"], cand=v["cand"], st=v["st"], si=v["si"],
        sat=v["saturation"], tol=v["tolerance"],
        slo=v["slope_lo"], shi=v["slope_hi"], K=len(v["templates"]),
    )
    # Per-request LP model cache: block object ids are only unique within run.
    _HIGHS_CACHE.clear()
    return _solve(ctx)


def _solve(ctx: _Ctx) -> dict:
    ref, cand = ctx.ref, ctx.cand
    st, si = ctx.st, ctx.si

    # --- nodes: every possible reference/candidate pair -------------------
    ni, nj = np.meshgrid(np.arange(ref.size), np.arange(cand.size),
                         indexing="ij")
    ni, nj = ni.ravel(), nj.ravel()
    order = np.lexsort((nj, ni))  # strictly increasing node index by (i, j)
    ni, nj = ni[order], nj[order]
    n = ni.size
    ctx.ni, ctx.nj, ctx.n_nodes = ni, nj, n
    ctx.cpos = np.searchsorted(st, cand[nj], side="left")

    ri, cj = ref[ni], cand[nj]

    # --- DAG transitions: order preserving, non-reusing, slope in range ---
    succ = [[] for _ in range(n)]
    for a in range(n):
        if a == n - 1:
            break
        m = (np.arange(n) > a) & (ni > ni[a]) & (nj > nj[a])
        idx = np.nonzero(m)[0]
        if idx.size:
            dc = cand[nj[idx]] - cj[a]
            dr = ref[ni[idx]] - ri[a]
            # Guard dc == 0 (ties among candidate times): such an edge can
            # never satisfy a finite, positive slope bound.
            s = np.full_like(dc, np.inf, dtype=float)
            nz = dc > 0
            s[nz] = dr[nz] / dc[nz]
            ok = nz & (s >= ctx.slo) & (s <= ctx.shi)
            succ[a] = idx[ok].tolist()
    ctx.succ = succ

    # --- longest chain (purely structural maximum anchor count) ----------
    dp = np.ones(n, dtype=int)
    for a in range(n - 1, -1, -1):
        if succ[a]:
            dp[a] = 1 + dp[succ[a]].max()
    ctx.dp = dp
    pmax = int(dp.max()) if n else 0
    if pmax < 2:
        return {"status": "insufficient_anchors",
                "structural_max_anchors": pmax}

    # --- longest chain feasible under the reduced residual relaxation ----
    result = _enumerate(ctx)
    if result is not None:
        result["status"] = "ok"
        result["structural_max_anchors"] = pmax
        return _finalize(ctx, result)

    return {"status": "infeasible",
            "structural_max_anchors": pmax}


# --------------------------------------------------------------------------- #
# Segment blocks
# --------------------------------------------------------------------------- #
def _segment_beta(ctx, u, v):
    """(beta0, beta1) of the interior segment between nodes u and v."""
    ca, cb = ctx.cand[ctx.nj[u]], ctx.cand[ctx.nj[v]]
    ra, rb = ctx.ref[ctx.ni[u]], ctx.ref[ctx.ni[v]]
    s = (rb - ra) / (cb - ca)
    return ra - s * ca, s


def _full_block(ctx, key, kind, u, v=None):
    """Full constraint block of an interior/exterior segment (LRU cached)."""
    cache_key = ("F", kind, key)
    cached = ctx.full_cache.get(cache_key)
    if cached is not None:
        ctx.full_cache.move_to_end(cache_key)
        return cached

    st, si = ctx.st, ctx.si
    npts = st.size
    if kind == "I":
        # Include the right anchor's grid point in the segment so the sample
        # points coinciding with an anchor time are never mapped through an
        # extrapolated exterior slope (the anchor point itself is mapped
        # identically by both adjacent segments).
        lo, hi = ctx.cpos[u], ctx.cpos[v]
        if hi < npts and st[hi] == ctx.cand[ctx.nj[v]]:
            hi += 1
        b0, b1 = _segment_beta(ctx, u, v)
    elif kind == "L":
        lo, hi = 0, ctx.cpos[u]
        b0, b1 = _segment_beta(ctx, u, v)
    else:  # "R": after last anchor v, slope of segment u->v
        lo, hi = ctx.cpos[v], npts
        b0, b1 = _segment_beta(ctx, u, v)

    blk = _build_block(lo, hi, b0, b1, st, si, ctx.templates,
                       ctx.sat, ctx.tol, np.arange(lo, hi), ctx.K)
    if len(ctx.full_cache) >= FULL_CACHE_SIZE:
        ctx.full_cache.popitem(last=False)
    ctx.full_cache[cache_key] = blk
    return blk


def _reduced_block(ctx, u, v):
    """Relaxed (subsampled) block for an interior transition, cached."""
    key = (u, v)
    blk = ctx.reduced.get(key)
    if blk is not None:
        return blk
    st, si = ctx.st, ctx.si
    lo, hi = ctx.cpos[u], ctx.cpos[v]
    b0, b1 = _segment_beta(ctx, u, v)
    sub = _reduced_indices(lo, hi, si, ctx.sat, ctx.tol)
    blk = _build_block(lo, hi, b0, b1, st, si, ctx.templates,
                       ctx.sat, ctx.tol, sub, ctx.K)
    ctx.reduced[key] = blk
    return blk


def _exterior_reduced_block(ctx, u, v, kind):
    """Reduced relaxation block for the before-first / after-last region.

    kind == "L": samples before anchor ``u`` using the slope of edge u->v.
    kind == "R": samples after  anchor ``v`` using the slope of edge u->v.
    """
    key = ("R", kind, u, v)
    blk = ctx.reduced_ext.get(key)
    if blk is not None:
        return blk
    st, si = ctx.st, ctx.si
    npts = st.size
    b0, b1 = _segment_beta(ctx, u, v)
    if kind == "L":
        lo, hi = 0, ctx.cpos[u]
    else:
        lo, hi = ctx.cpos[v], npts
    sub = _reduced_indices(lo, hi, si, ctx.sat, ctx.tol)
    blk = _build_block(lo, hi, b0, b1, st, si, ctx.templates,
                       ctx.sat, ctx.tol, sub, ctx.K)
    ctx.reduced_ext[key] = blk
    return blk


# --------------------------------------------------------------------------- #
# Enumeration: two-phase longest-feasible-chain search
#
# Phase 1: longest chain whose *reduced* LPs (uniform subsample + every
#          potentially binding point) are feasible.  This is an upper bound
#          on the true optimum and prunes structurally.
# Phase 2: enumerate every chain of that length, requiring reduced prefix
#          feasibility and reduced before/after exterior feasibility, then
#          verify survivors with the full LP and accumulate certified
#          per-component concentration extrema.
# --------------------------------------------------------------------------- #
EXPLANATION_CAP = 256


def _enumerate(ctx: _Ctx):
    K = ctx.K
    dp, succ = ctx.dp, ctx.succ
    n = ctx.n_nodes
    pmax = int(dp.max()) if n else 0

    # Standalone feasibility of each interior edge (cached, lazy).
    edge_state = {}

    def edge_good(u, w):
        cached = edge_state.get((u, w))
        if cached is not None:
            return cached
        blk = _reduced_block(ctx, u, w)
        good = (not blk.infeasible) and lp_solve([blk], K)[0]
        edge_state[(u, w)] = good
        return good

    good_neighbors = {u: [w for w in succ[u]] for u in range(n)}

    dpm = {}

    def dp_good(u):
        z = dpm.get(u)
        if z is not None:
            return z
        best = 0
        for w in good_neighbors[u]:
            if edge_good(u, w):
                cand = 1 + dp_good(w)
                if cand > best:
                    best = cand
        dpm[u] = best
        return best

    # ---- Phase 1: longest chain feasible under the reduced relaxation ----
    def longest_reduced():
        best_len = 0
        # Global structural ceiling for a chain starting at node s is
        # 1 + dp_good(s); the best possible across all starts is pmax.
        def dfs(path, prefix, left_ext):
            nonlocal best_len
            depth = len(path)
            u = path[-1]
            # Subtree via w has at most 1 (w) + dp_good(w) further anchors.
            cands = [w for w in good_neighbors[u]
                     if edge_good(u, w)
                     and depth + 1 + dp_good(w) >= max(best_len, 2)]
            cands.sort(key=dp_good, reverse=True)
            for w in cands:
                blk = _reduced_block(ctx, u, w)
                trial = prefix + [blk]
                if not lp_solve(trial, K)[0]:
                    continue
                # The before-first exterior is extrapolated with the slope
                # of the *first* edge only; build it when that edge (path[0]
                # -> w) is chosen, then reuse the same block deeper down.
                if depth == 1:
                    le = _exterior_reduced_block(ctx, path[0], w, "L")
                    if le.infeasible or not lp_solve(trial + [le], K)[0]:
                        continue
                else:
                    le = left_ext
                    if le is not None and not lp_solve(trial + [le], K)[0]:
                        continue
                if depth + 1 > best_len:
                    best_len = depth + 1
                if best_len < pmax:
                    dfs(path + [w], trial, le)

        for s in range(n):
            if 1 + dp_good(s) < max(best_len, 2):
                continue
            if best_len >= pmax:
                break
            dfs([s], [], None)
        return best_len

    target = longest_reduced()
    if target < 2:
        return None

    # ---- Phase 2: enumerate all reduced-feasible chains of target length -
    lower = np.full(K, np.inf)
    upper = np.full(K, -np.inf)
    upper_inf = np.zeros(K, dtype=bool)
    explanations = []
    pair_union = set()
    pair_inter = None
    truncated = False

    def record(path):
        nonlocal pair_inter, truncated
        blocks = [
            _full_block(ctx, (path[0], path[1]), "L", path[0], path[1])]
        for h in range(len(path) - 1):
            blocks.append(_full_block(ctx, (path[h], path[h + 1]), "I",
                                      path[h], path[h + 1]))
        blocks.append(_full_block(ctx, (path[-2], path[-1]), "R",
                                  path[-2], path[-1]))
        if not lp_solve(blocks, K)[0]:
            return False

        pairs = [(int(ctx.ni[x]), int(ctx.nj[x])) for x in path]
        pset = set(pairs)
        pair_union.update(pset)
        pair_inter = pset if pair_inter is None else (pair_inter & pset)
        if len(explanations) < EXPLANATION_CAP:
            explanations.append(pairs)
        else:
            truncated = True

        for k in range(K):
            if not upper_inf[k]:
                st_max, vmax = lp_solve(blocks, K, sign=-1, k=k)
                if st_max == 3:
                    upper_inf[k] = True
                elif st_max == 0 and vmax > upper[k]:
                    upper[k] = vmax
            st_min, vmin = lp_solve(blocks, K, sign=+1, k=k)
            if st_min == 0 and vmin < lower[k]:
                lower[k] = vmin
        return True

    def dfs2(path, blocks, left_ext):
        depth = len(path)
        u = path[-1]
        need = target - depth
        if need == 0:
            re = _exterior_reduced_block(ctx, path[-2], path[-1], "R")
            if re.infeasible:
                return
            if not lp_solve(blocks + [left_ext, re], K)[0]:
                return
            record(path)
            return
        cands = [w for w in good_neighbors[u]
                 if edge_good(u, w) and dp_good(w) >= need - 1]
        for w in cands:
            blk = _reduced_block(ctx, u, w)
            trial = blocks + [blk]
            if not lp_solve(trial, K)[0]:
                continue
            if depth == 1:
                le = _exterior_reduced_block(ctx, path[0], w, "L")
                if le.infeasible or not lp_solve(trial + [le], K)[0]:
                    continue
                dfs2(path + [w], trial, le)
            else:
                dfs2(path + [w], trial, left_ext)

    for s in range(n):
        if dp_good(s) + 1 >= target:
            dfs2([s], [], None)

    if not explanations:
        # Reduced relaxation was optimistic at that length: the true optimum
        # is shorter. Fall back one length at a time (rare, cheap: reduced
        # feasibility already rules out most chains).
        return _enumerate_shorter(ctx, target - 1,
                                  edge_state, good_neighbors, dpm, dp_good)

    return {
        "lower": lower,
        "upper": upper,
        "upper_inf": upper_inf,
        "explanations": explanations,
        "explanations_truncated": truncated,
        "pair_union": pair_union,
        "pair_inter": pair_inter if pair_inter is not None else set(),
        "max_anchors": target,
    }


def _enumerate_shorter(ctx, target, edge_state, good_neighbors, dpm, dp_good):
    """Fallback: repeat the fixed-length enumeration at shorter lengths."""
    K = ctx.K
    while target >= 2:
        lower = np.full(K, np.inf)
        upper = np.full(K, -np.inf)
        upper_inf = np.zeros(K, dtype=bool)
        explanations = []
        pair_union = set()
        pair_inter = None
        truncated = False

        def record(path):
            nonlocal pair_inter, truncated
            blocks = [
                _full_block(ctx, (path[0], path[1]), "L", path[0], path[1])]
            for h in range(len(path) - 1):
                blocks.append(_full_block(ctx, (path[h], path[h + 1]), "I",
                                          path[h], path[h + 1]))
            blocks.append(_full_block(ctx, (path[-2], path[-1]), "R",
                                      path[-2], path[-1]))
            if not lp_solve(blocks, K)[0]:
                return False
            pairs = [(int(ctx.ni[x]), int(ctx.nj[x])) for x in path]
            pset = set(pairs)
            pair_union.update(pset)
            pair_inter = pset if pair_inter is None else (pair_inter & pset)
            if len(explanations) < EXPLANATION_CAP:
                explanations.append(pairs)
            else:
                truncated = True
            for k in range(K):
                if not upper_inf[k]:
                    st_max, vmax = lp_solve(blocks, K, sign=-1, k=k)
                    if st_max == 3:
                        upper_inf[k] = True
                    elif st_max == 0 and vmax > upper[k]:
                        upper[k] = vmax
                st_min, vmin = lp_solve(blocks, K, sign=+1, k=k)
                if st_min == 0 and vmin < lower[k]:
                    lower[k] = vmin
            return True

        def dfs(path, blocks, left_ext):
            depth = len(path)
            u = path[-1]
            need = target - depth
            if need == 0:
                re = _exterior_reduced_block(ctx, path[-2], path[-1], "R")
                if re.infeasible:
                    return
                if not lp_solve(blocks + [left_ext, re], K)[0]:
                    return
                record(path)
                return
            for w in good_neighbors[u]:
                if not edge_state.get((u, w)) or dp_good(w) < need - 1:
                    continue
                blk = _reduced_block(ctx, u, w)
                trial = blocks + [blk]
                if not lp_solve(trial, K)[0]:
                    continue
                if depth == 1:
                    le = _exterior_reduced_block(ctx, path[0], w, "L")
                    if le.infeasible or not lp_solve(trial + [le], K)[0]:
                        continue
                    dfs(path + [w], trial, le)
                else:
                    dfs(path + [w], trial, left_ext)

        for s in range(ctx.n_nodes):
            if dp_good(s) + 1 >= target:
                dfs([s], [], None)
        if explanations:
            return {
                "lower": lower, "upper": upper, "upper_inf": upper_inf,
                "explanations": explanations,
                "explanations_truncated": truncated,
                "pair_union": pair_union,
                "pair_inter": pair_inter if pair_inter is not None else set(),
                "max_anchors": target,
            }
        target -= 1
    return None


def _finalize(ctx: _Ctx, result: dict) -> dict:
    K = ctx.K
    concs = ctx.concentrations
    components = []
    for k in range(K):
        lo = result["lower"][k]
        if not np.isfinite(lo):
            lo = 0.0  # alpha = 0 is always feasible in any feasible matching
        hi_raw = result["upper"][k]
        # None when explicitly unbounded, or when no upper-binding constraint
        # ever involves this component (its template is supported nowhere on
        # the sampled/saturated points).
        unbounded = result["upper_inf"][k] or not np.isfinite(hi_raw)
        hi = None if unbounded else float(hi_raw)
        components.append({
            "index": k,
            "concentration_lower": float(max(0.0, lo) * concs[k]),
            "concentration_upper": (None if hi is None
                                    else float(hi * concs[k])),
        })

    def pair_key(p):
        return (p[0], p[1])

    common = sorted(result["pair_inter"], key=pair_key)
    possible = sorted(result["pair_union"], key=pair_key)
    evidence = {
        "common_anchors": [
            {"reference_index": i, "candidate_index": j} for i, j in common],
        "possible_anchors": [
            {"reference_index": i, "candidate_index": j} for i, j in possible],
        "explanations": [
            [{"reference_index": i, "candidate_index": j} for i, j in path]
            for path in result["explanations"]
        ],
        "explanations_truncated": result["explanations_truncated"],
        "num_explanations": len(result["explanations"]),
    }
    if result["explanations_truncated"]:
        evidence["num_explanations_note"] = (
            f"more than {EXPLANATION_CAP} tied explanations exist; the list "
            "is capped, but concentration extrema and anchor union/intersection "
            "are computed over all explanations and remain exact")
    if result["explanations_truncated"]:
        evidence["num_explanations_note"] = (
            f"more than {EXPLANATION_CAP} tied explanations; "
            "concentration bounds and anchor unions remain exact")
    return {
        "status": result.get("status", "ok"),
        "max_anchors": result["max_anchors"],
        "structural_max_anchors": result["structural_max_anchors"],
        "components": components,
        "evidence": evidence,
    }
