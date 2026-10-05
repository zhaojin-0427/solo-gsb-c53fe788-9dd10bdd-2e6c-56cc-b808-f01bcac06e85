"""Correctness tests for overlap ambiguity, null upper bound, missed ISs,
distractors, and tolerance/saturation semantics."""
import numpy as np
from app.solver import quantify


def gauss(t, mu, sig, amp):
    return amp * np.exp(-0.5 * ((t - mu) / sig) ** 2)


def base_payload(ref_times, cand_times, sample_y, templates,
                 st=None, saturation=1e9, tol=0.05, slope=(0.5, 2.0),
                 concs=None):
    t0 = templates[0] if isinstance(templates[0], np.ndarray) else None
    n = t0.size if t0 is not None else 2001
    ref_grid = np.linspace(0, 100, n)
    if st is None:
        st = ref_grid
    return {
        "standards": [
            {"times": ref_grid.tolist(),
             "intensities": (tt if isinstance(tt, np.ndarray)
                             else gauss(ref_grid, *tt)).tolist(),
             "concentration": 1.0 if concs is None else concs[k]}
            for k, tt in enumerate(templates)],
        "reference_times": list(map(float, ref_times)),
        "candidate_times": list(map(float, cand_times)),
        "sample_times": st.tolist(),
        "sample_intensities": np.clip(sample_y, 0, None).tolist(),
        "saturation": saturation,
        "tolerance": tol,
        "slope_range": list(slope),
    }


def test_overlapping_peaks_not_unique():
    """Two identical-shape co-eluting components => no unique attribution;
    bounds must span a range and never force one concentration."""
    g = np.linspace(0, 100, 4001)
    shape = gauss(g, 50, 2.0, 1.0)
    # mixture alpha0 + alpha1 = 1 within tolerance; individual unknown
    y = 1.0 * shape
    p = base_payload([20, 80], [20, 80], y, [shape, shape])
    r = quantify(p)
    assert r["status"] == "ok", r
    c0, c1 = r["components"]
    # sum constrained, each can range between ~0 and ~1
    assert c0["concentration_lower"] < 0.05
    assert c0["concentration_upper"] is not None
    assert c0["concentration_upper"] > 0.95
    assert c1["concentration_lower"] < 0.05
    assert c1["concentration_upper"] > 0.95
    print("OVERLAP OK", (round(c0["concentration_lower"], 3),
                         round(c0["concentration_upper"], 3)),
          (round(c1["concentration_lower"], 3),
           round(c1["concentration_upper"], 3)))


def test_null_upper_bound_when_unsupported():
    """A component whose template is absent everywhere it could be needed
    can have unbounded concentration => upper bound null."""
    g = np.linspace(0, 100, 4001)
    # comp 0 is a peak at 50; comp 1 is a peak at 95 (outside any constraint
    # binding region effectively => its alpha never bounded above where it
    # could hide). Use a sample whose signal is zero everywhere: then any
    # component template nonzero would be forced to 0; instead make comp 1
    # live only where sample grid ends (never sampled) -> unbounded.
    a0 = gauss(g, 50, 1.5, 1.0)
    a1 = gauss(g, 99.9, 0.02, 1.0)  # within grid but effectively a spike
    y = 0.8 * a0
    # tighten so a1 spike at a sampled-zero point would force alpha1=0; to get
    # a null upper we instead provide NO sample coverage near a1: restrict
    # sample time grid to [0, 90]
    st = np.linspace(0, 90, 3000)
    p = base_payload([20, 80], [20, 80], 0.8 * np.interp(st, g, a0),
                     [a0, a1], st=st, tol=0.02)
    r = quantify(p)
    assert r["status"] == "ok", r
    c1 = r["components"][1]
    assert c1["concentration_upper"] is None, c1
    assert c1["concentration_lower"] == 0.0
    print("NULL UPPER OK (component supported nowhere on grid)")


def test_missed_internal_standard():
    """One reference IS has no matching candidate (missed detection); the
    longest feasible chain must skip it and still align the rest."""
    g = np.linspace(0, 100, 3001)
    shape = gauss(g, 50, 2.0, 1.0)
    ref = [10, 30, 70, 90]
    # reference 30 is missed: no candidate near 30
    cand = [10.0, 70.2, 90.3]
    y = np.interp(np.linspace(0, 100, 3001), g,
                  0.7 * shape)  # aligned (slope ~1)
    p = base_payload(ref, cand, y, [shape], slope=(0.9, 1.1))
    r = quantify(p)
    assert r["status"] == "ok", r
    assert r["max_anchors"] == 3, r["max_anchors"]
    used_refs = {a["reference_index"]
                 for a in r["evidence"]["possible_anchors"]}
    assert 1 not in used_refs  # missed reference index 1 (time 30)
    print("MISSED IS OK, anchors", r["max_anchors"], "refs", sorted(used_refs))


def test_distractor_peaks():
    """Extra candidate peaks (impurities) must not be forced into anchors."""
    g = np.linspace(0, 100, 3001)
    shape = gauss(g, 50, 2.0, 1.0)
    ref = [20, 80]
    cand = [20.0, 33.3, 51.0, 80.0]  # two distractors
    y = 0.6 * shape
    p = base_payload(ref, cand, y, [shape], slope=(0.9, 1.1))
    r = quantify(p)
    assert r["status"] == "ok", r
    common = [(a["reference_index"], a["candidate_index"])
              for a in r["evidence"]["common_anchors"]]
    assert (0, 0) in common and (1, 3) in common, common
    print("DISTRACTOR OK", common)


def test_tolerance_boundary():
    """Signal equal to prediction + exactly tol is feasible; + more is not."""
    g = np.linspace(0, 100, 2001)
    shape = gauss(g, 50, 2.0, 1.0)
    # within tolerance (allow floating-point/interpolation epsilon)
    p = base_payload([20, 80], [20, 80], 1.0 * shape + 0.05 - 1e-9,
                     [shape], tol=0.05)
    assert quantify(p)["status"] == "ok"
    p2 = base_payload([20, 80], [20, 80], 1.0 * shape + 0.051,
                      [shape], tol=0.05)
    assert quantify(p2)["status"] == "infeasible"
    print("TOLERANCE BOUNDARY OK")


def test_saturation_semantics():
    """Saturated point needs prediction >= sat-tol (one-sided); a prediction
    far above is allowed, but below sat-tol is rejected."""
    g = np.linspace(0, 100, 4001)
    shape = gauss(g, 50, 1.0, 1.0)
    sat, tol = 1.0, 0.05
    # peak clipped onto the saturation plateau; elsewhere the normal trace
    y = np.minimum(1.2 * shape, sat)
    p = base_payload([20, 80], [20, 80], y, [shape],
                     saturation=sat, tol=tol)
    r = quantify(p)
    assert r["status"] == "ok", r  # max template 1.0 >= sat-tol 0.95
    # weak template cannot reach the saturation threshold at the plateau
    weak = gauss(g, 50, 1.0, 0.5)
    y2 = np.minimum(1.2 * weak, sat)  # no actual plateau; build real one:
    y2 = np.where(np.abs(g - 50) < 0.5, sat, 1.2 * weak)
    p2 = base_payload([20, 80], [20, 80], y2, [weak],
                      saturation=sat, tol=tol)
    assert quantify(p2)["status"] == "infeasible"
    print("SATURATION SEMANTICS OK")


def test_slope_range_enforced():
    """Pairing whose piecewise slope lies outside the range must be rejected;
    feasible solution exists inside it."""
    g = np.linspace(0, 100, 4001)
    shape = gauss(g, 50, 2.0, 1.0)

    # (a) A candidate second anchor whose segment slope is outside the
    # allowed range: the corresponding pairing must never form a 2-chain.
    p = base_payload([20.0, 80.0], [20.0, 40.0], 0.7 * shape, [shape],
                     st=g, slope=(0.95, 1.05))  # 60/20 = 3.0 rejected
    r = quantify(p)
    assert r["structural_max_anchors"] == 1, r
    assert r["status"] == "insufficient_anchors", r

    # (b) Slope in range, but the in-range pairing warps the time axis so the
    # peak is displaced beyond tolerance -> feasible structure, infeasible
    # signal.  Two candidate anchors: a wrong in-range pairing (second anchor
    # too early) is the only 2-chain and misaligns the peak.
    p2 = base_payload([20.0, 80.0], [20.0, 78.0], 0.7 * shape, [shape],
                      st=g, slope=(0.95, 1.05))
    r2 = quantify(p2)
    assert r2["structural_max_anchors"] == 2, r2
    assert r2["status"] == "infeasible", r2
    print("SLOPE RANGE ENFORCED OK")


if __name__ == "__main__":
    test_overlapping_peaks_not_unique()
    test_null_upper_bound_when_unsupported()
    test_missed_internal_standard()
    test_distractor_peaks()
    test_tolerance_boundary()
    test_saturation_semantics()
    test_slope_range_enforced()
    print("ALL CORRECTNESS TESTS PASSED")
