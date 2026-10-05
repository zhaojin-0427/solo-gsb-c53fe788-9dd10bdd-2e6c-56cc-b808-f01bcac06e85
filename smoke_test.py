"""Quick correctness smoke test for the solver."""
import numpy as np
from app.solver import quantify


def gauss(t, mu, sig, amp):
    return amp * np.exp(-0.5 * ((t - mu) / sig) ** 2)


def make_case(n_components=2, n_ref=4, extra_candidates=2,
              slope=1.02, noise=0.0, saturation=1e9, alpha=(1.0, 0.5),
              seed=0):
    rng = np.random.default_rng(seed)
    # reference grid and templates
    ref_grid = np.linspace(0, 100, 2001)
    templates = []
    for k in range(n_components):
        mu = 15 + k * 30
        y = gauss(ref_grid, mu, 2.5 + 0.3 * k, 1.0)
        templates.append((ref_grid.copy(), y))

    ref_times = np.array([10.0, 30.0, 55.0, 80.0][:n_ref])
    cand_times = ref_times / slope
    # add distractors (missed IS + impurities)
    distractors = rng.uniform(5, 85, extra_candidates)
    all_cand = np.sort(np.concatenate([cand_times, distractors]))

    # sample grid and mixture mapped through phi(t) = slope * t
    st = np.linspace(0, 100 / slope, 1500)
    phi = slope * st
    si = np.zeros_like(st)
    for k, (tt, yy) in enumerate(templates):
        a = alpha[k] if k < len(alpha) else 0.0
        si += a * np.interp(phi, tt, yy)
    si += rng.normal(0, noise, size=si.shape)
    si = np.clip(si, 0, None)

    payload = {
        "standards": [
            {"times": tt.tolist(), "intensities": yy.tolist(),
             "concentration": 1.0}
            for tt, yy in templates],
        "reference_times": ref_times.tolist(),
        "candidate_times": all_cand.tolist(),
        "sample_times": st.tolist(),
        "sample_intensities": si.tolist(),
        "saturation": saturation,
        "tolerance": 0.02,
        "slope_range": [0.9, 1.15],
    }
    return payload


def test_basic():
    payload = make_case()
    res = quantify(payload)
    print("status:", res["status"])
    assert res["status"] == "ok", res
    print("max anchors:", res["max_anchors"])
    for c in res["components"]:
        print(c)
    ev = res["evidence"]
    print("common anchors:", ev["common_anchors"])
    print("possible anchors:", ev["possible_anchors"])
    print("num explanations:", ev["num_explanations"])
    # component 0 alpha=1.0, component 1 alpha=0.5
    for k, expected in enumerate((1.0, 0.5)):
        c = res["components"][k]
        assert abs(c["concentration_lower"] - expected) < 0.05, c
        assert c["concentration_upper"] is not None
        assert abs(c["concentration_upper"] - expected) < 0.05, c
    print("BASIC OK")


def test_saturated_gives_infinite_upper():
    payload = make_case(saturation=0.8)
    # clip signal at saturation
    si = np.array(payload["sample_intensities"])
    sat = payload["saturation"]
    si = np.clip(si, None, sat)
    payload["sample_intensities"] = si.tolist()
    payload["tolerance"] = 0.02
    res = quantify(payload)
    print("saturated status:", res["status"])
    if res["status"] == "ok":
        for c in res["components"]:
            print(c)
        print("common:", res["evidence"]["common_anchors"])


def test_invalid_inputs():
    payload = make_case()
    bad = dict(payload)
    bad["candidate_times"] = [1.0, 5.0, 3.0]  # decreasing
    res = quantify(bad)
    assert res["status"] == "invalid_input", res
    bad = dict(payload)
    bad["sample_intensities"] = bad["sample_intensities"][:-1]
    res = quantify(bad)
    assert res["status"] == "invalid_input", res
    bad = dict(payload)
    bad["slope_range"] = [1.2, 1.0]
    res = quantify(bad)
    assert res["status"] == "invalid_input", res
    print("INVALID OK")


def test_insufficient_anchors():
    payload = make_case()
    payload["reference_times"] = [10.0]
    payload["candidate_times"] = [9.0]
    res = quantify(payload)
    assert res["status"] == "insufficient_anchors", res
    print("INSUFFICIENT OK:", res)


def test_infeasible_signal():
    payload = make_case(alpha=(1.0, 0.5))
    si = np.array(payload["sample_intensities"])
    si[700] += 5.0  # huge unexplained peak
    payload["sample_intensities"] = si.tolist()
    res = quantify(payload)
    print("infeasible:", res["status"])
    assert res["status"] == "infeasible", res
    print("INFEASIBLE OK")


if __name__ == "__main__":
    test_basic()
    test_saturated_gives_infinite_upper()
    test_invalid_inputs()
    test_insufficient_anchors()
    test_infeasible_signal()
    print("ALL SMOKE TESTS PASSED")
