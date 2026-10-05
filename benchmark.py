"""Online-injection benchmark: 12 reference ISs, 24 candidates, 8192 points.

Builds a piecewise-linearly drifted, 6-component synthetic chromatogram with
12 true internal-standard anchors, 12 distractor candidate peaks (some
co-eluting) and detector saturation, then certifies the result and measures
wall-clock latency.
"""
import time
import numpy as np

from app.solver import quantify


def gauss(t, mu, sig, amp):
    return amp * np.exp(-0.5 * ((t - mu) / sig) ** 2)


def build_payload(seed=1, saturation=1.6, tol=0.01,
                  n_ref=12, n_cand_extra=12, n_points=8192, n_comp=6):
    rng = np.random.default_rng(seed)

    t0, t1 = 0.0, 120.0
    ref_grid = np.linspace(t0, t1, 4000)

    # six well-separated (slightly overlapping) component templates; all
    # peaks lie strictly inside the internal-standard anchor range.
    mus = np.linspace(15, 105, n_comp)
    amps = rng.uniform(1.8, 2.4, n_comp)  # strong enough for real saturation
    sigs = rng.uniform(1.6, 2.6, n_comp)
    templates = []
    for k in range(n_comp):
        y = gauss(ref_grid, mus[k], sigs[k], amps[k])
        templates.append((ref_grid.copy(), y))
    pure_conc = np.full(n_comp, 100.0)
    alpha_true = np.array([0.8, 0.0, 0.55, 0.0, 1.1, 0.3])  # comp 1,3 absent

    # 12 numbered reference internal standards spanning the peaks
    ref_is = np.sort(rng.uniform(10, 110, n_ref))
    ref_is.sort()

    # Piecewise-linear timing drift defined *exactly* by the anchor chain:
    # edge slopes in [0.985, 1.02], candidate times cumulative; outside the
    # outermost anchors the mapping is linearly extrapolated with the first /
    # last edge slope (identical to the solver's calibration).
    edge_slopes = rng.uniform(0.985, 1.02, n_ref - 1)
    true_cand = np.empty(n_ref)
    true_cand[0] = ref_is[0] / rng.uniform(0.99, 1.01)
    for i in range(1, n_ref):
        true_cand[i] = true_cand[i - 1] + (
            ref_is[i] - ref_is[i - 1]) / edge_slopes[i - 1]

    def inv(c):
        """Sample time -> reference time (piecewise linear + extrapolation)."""
        r = np.interp(c, true_cand, ref_is)
        left = c < true_cand[0]
        if np.any(left):
            s = (ref_is[1] - ref_is[0]) / (true_cand[1] - true_cand[0])
            r = np.where(left, ref_is[0] + s * (c - true_cand[0]), r)
        right = c > true_cand[-1]
        if np.any(right):
            s = (ref_is[-1] - ref_is[-2]) / (true_cand[-1] - true_cand[-2])
            r = np.where(right, ref_is[-1] + s * (c - true_cand[-1]), r)
        return r

    # 12 distractor candidates, away from true times
    distractors = []
    while len(distractors) < n_cand_extra:
        d = rng.uniform(2, true_cand[-1] + 2)
        if np.all(np.abs(d - true_cand) > 1.5):
            distractors.append(d)
    all_cand = np.sort(np.concatenate([true_cand, distractors]))

    st = np.linspace(0, true_cand[-1] + 2, n_points)
    phi = inv(st)
    si = np.zeros(n_points)
    for k, (tt, yy) in enumerate(templates):
        si += alpha_true[k] * np.interp(phi, tt, yy)
    si = np.clip(si, 0.0, saturation)  # saturate the biggest peak

    payload = {
        "standards": [
            {"times": tt.tolist(), "intensities": yy.tolist(),
             "concentration": float(pure_conc[k])}
            for k, (tt, yy) in enumerate(templates)],
        "reference_times": ref_is.tolist(),
        "candidate_times": all_cand.tolist(),
        "sample_times": st.tolist(),
        "sample_intensities": si.tolist(),
        "saturation": saturation,
        "tolerance": tol,
        "slope_range": [0.95, 1.05],
    }
    truth = {"alpha": alpha_true, "true_pairs_ref_idx":
             [(k, int(np.where(np.isclose(all_cand, true_cand[k],
                                         atol=1e-6))[0][0]))
              for k in range(n_ref)]}
    return payload, truth


def main():
    payload, truth = build_payload()
    t0 = time.perf_counter()
    res = quantify(payload)
    dt = time.perf_counter() - t0
    print(f"elapsed: {dt:.3f} s")
    print("status:", res["status"])
    if res["status"] != "ok":
        print(res)
        raise SystemExit(1)
    print("max anchors:", res["max_anchors"],
          "structural:", res["structural_max_anchors"])
    for c, a in zip(res["components"], truth["alpha"]):
        print(f"comp {c['index']}: true alpha={a:.2f}  "
              f"c=[{c['concentration_lower']:.3f}, "
              f"{c['concentration_upper'] if c['concentration_upper'] is None else round(c['concentration_upper'],3)}]")
    ev = res["evidence"]
    print("common anchors:", len(ev["common_anchors"]))
    print("possible anchors:", len(ev["possible_anchors"]))
    print("num explanations:", ev["num_explanations"],
          "truncated:", ev["explanations_truncated"])
    assert dt <= 10.0, f"latency {dt:.2f}s exceeds 10 s budget"
    print("WITHIN 10 s BUDGET")


if __name__ == "__main__":
    main()
