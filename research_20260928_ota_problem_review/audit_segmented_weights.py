"""Exact, CPU-only algebra audit. Not a SegOTA reproduction or FL benchmark.

Four clients, two scalar coordinates, groups {0}, {1,2,3}; one
group per coordinate per transmission. Frozen innovations are (+1,-1,-1,-1)
in each coordinate. Channels: h0=(1,0), h1=h2=h3=(0,0.2).
All clients use unit peak power. Real receiver noise variance is 0.1.
Only payload uses/energy are counted here; control, CSI and downlink are absent.
"""

from fractions import Fraction as F
from pathlib import Path
import json


def segmented(lam, repeats=1, balanced_cycle=False, cross_gain=F(0)):
    v0, vr = 1 - lam / 2, F(1, 3) + lam / 6
    # Transmit b_k=gamma_group*v_k/h_own = 1 under these channels.
    gamma0, gamma1 = 1 / v0, F(1, 5) / vr
    # Cross channels: h0=(1,e), others=(0.2e,0.2), fixed unit beams.
    mean0 = v0 - cross_gain * (gamma1 / gamma0) * 3 * vr
    mean1 = -3 * vr + cross_gain * (gamma0 / gamma1) * v0
    mean = (mean0 + mean1) / 2
    target = -F(1, 2)
    bias_sq = (mean - target) ** 2
    sampling = ((mean0 - mean) ** 2 + (mean1 - mean) ** 2) / 2
    noise = F(1, 20) * (1 / gamma0**2 + 1 / gamma1**2)
    sampling_after = F(0) if balanced_cycle else sampling / repeats
    return {
        "lambda": float(lam),
        "expected_coefficient": [float(v0 / 2)] + [float(vr / 2)] * 3,
        "mean": float(mean),
        "bias_squared": float(bias_sq),
        "sampling_variance": float(sampling_after),
        "receiver_noise_variance": float(noise / repeats),
        "mse_per_coordinate": float(bias_sq + sampling_after + noise / repeats),
        "payload_real_uses": repeats,
        "sum_tx_energy_unit_symbol_duration": 4 * repeats,
        "selection": "fixed lambda; no learned scheduler",
    }


def main():
    rows = []
    for lam in [F(0), F(1, 2), F(1)]:
        for repeats, cycle in [(1, False), (2, False), (2, True)]:
            row = segmented(lam, repeats, cycle)
            row["method"] = "segmented_balanced_cycle" if cycle else "segmented_random"
            rows.append(row)

    # Minimize the exact quadratic for this deliberately chosen frozen signal.
    # This is an oracle diagnostic, not a deployable lambda-selection rule.
    for repeats, cycle in [(1, False), (2, True)]:
        # MSE = sampling + (1-lambda)^2/4
        #       + (17/90 + 4*lambda/45 + 17*lambda^2/360)/repeats.
        linear = -F(1, 2) + F(4, 45) / repeats
        quadratic = F(1, 4) + F(17, 360) / repeats
        lam = -linear / (2 * quadratic)
        row = segmented(lam, repeats, cycle)
        row["method"] = "oracle_lambda_balanced_cycle" if cycle else "oracle_lambda_random"
        row["selection"] = "oracle: exact frozen-signal MSE, NOT a proposed algorithm"
        rows.append(row)

    # Full-model unbiased OTA: unit beam (1,5)/sqrt(26), scale 4/sqrt(26).
    # Two coordinates cost two real uses, the same as a two-slot segmented cycle.
    rows.append({
        "method": "full_model_unbiased_OTA", "mean": -0.5,
        "bias_squared": 0.0, "sampling_variance": 0.0,
        "receiver_noise_variance": float(F(13, 80)),
        "mse_per_coordinate": float(F(13, 80)),
        "payload_real_uses": 2, "sum_tx_energy_unit_symbol_duration": 8,
    })
    # Standard linear MMSE receiver, unit independent source covariance assumed:
    # w=(H H^T + .1 I)^-1 H p = (5/22,15/22).
    # Evaluate it on the same fixed correlated innovations as every other row.
    w0, w1 = F(5, 22), F(15, 22)
    mean = w0 - F(3, 5) * w1
    noise = F(1, 10) * (w0**2 + w1**2)
    rows.append({
        "method": "full_model_standard_LMMSE_OTA",
        "mean": float(mean), "bias_squared": float((mean + F(1, 2))**2),
        "sampling_variance": 0.0, "receiver_noise_variance": float(noise),
        "mse_per_coordinate": float((mean + F(1, 2))**2 + noise),
        "payload_real_uses": 2, "sum_tx_energy_unit_symbol_duration": 8,
        "selection": "standard unit-independent-source LMMSE; NOT oracle signal tuning",
    })

    # Checks target specific possible mathematical errors, not just implementation.
    assert segmented(F(0))["expected_coefficient"] == [0.5] + [float(F(1, 6))] * 3
    assert segmented(F(1))["expected_coefficient"] == [0.25] * 4
    assert segmented(F(1), 2, True)["mse_per_coordinate"] == float(F(13, 80))
    cross = segmented(F(1), cross_gain=F(1, 10))
    assert abs(cross["mean"] - (-0.39)) < 1e-14  # own-weight correction does not erase leakage
    # The usual full-model baseline beats even the oracle partial correction here.
    cycle_oracle = next(r for r in rows if r["method"] == "oracle_lambda_balanced_cycle")
    assert rows[-1]["mse_per_coordinate"] < cycle_oracle["mse_per_coordinate"]

    result = {
        "kind": "exact rational algebra; CPU; no dataset, GPU, Monte Carlo, or RF measurement",
        "scope": "frozen update estimation; not multi-round neural-network training",
        "K": 4, "S": 2, "D_real": 2, "sigma_squared": 0.1,
        "target_weights": [0.25] * 4, "rows": rows,
        "interference_counterexample": cross,
        "quadratic_learning_example": {
            "local_objective": "f_k(theta)=0.5*(theta-c_k)^2; c=(1,-1,-1,-1)",
            "true_minimizer": -0.5, "uncorrected_mean_dynamics_fixed_point": 0.0,
            "objective_excess_at_uncorrected_fixed_point": 0.125,
            "qualification": "mean drift fixed point; not almost-sure noisy SGD convergence",
        },
        "checks_passed": 5,
    }
    dest = Path(__file__).with_name("math_audit.json")
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        print(row["method"], "lambda=", row.get("lambda"), "uses=", row["payload_real_uses"],
              "MSE=", round(row["mse_per_coordinate"], 8))
    print("Checks passed: 5; wrote", dest.name)


if __name__ == "__main__":
    main()
