"""Diagnostic: does the same regularizer help ordinary digital split learning?

Added after the fashion20 core results; NOT a preregistered hypothesis test.
All original seeds/settings retained. Required norm reports are charged.
"""
import argparse
import copy
import json
from pathlib import Path
import run_sfl as base
import run_channel_width as width


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layout", choices=["views", "width"], default="views")
    ap.add_argument("--seeds", default="311,312,313")
    ap.add_argument("--snr", type=float, default=20.)
    ap.add_argument("--rounds", type=int, default=80)
    ap.add_argument("--bits", default="4,8")
    args = ap.parse_args()
    ParentLedger = width.InputLedger if args.layout == "width" else base.Ledger

    class ControlLedger(ParentLedger):
        def frame(self, config, batch, feature_dim, h, snr, ota_energy=None, signal_dim=None):
            super().frame(config, batch, feature_dim, h, snr, ota_energy, signal_dim)
            uses = base.CP*float(sum(32/base.rates(h, snr)))
            self.x["control_uses"] += uses
            self.x["ul_energy"] += uses

    base.Ledger = ControlLedger
    if args.layout == "width":
        base.Model = width.WidthModel
    for seed in map(int, args.seeds.split(",")):
        ds = base.data(args.data, seed)
        for bits in map(int, args.bits.split(",")):
            name = f"Latent{bits}-Reg-Q16-W8"
            file = args.out/f"{name}_seed{seed}.json"
            if file.exists():
                print("SKIP completed", file.name, flush=True)
                continue
            cfg = copy.deepcopy(base.CONFIGS[f"Latent{bits}-Q16-W8"])
            cfg["correction"] = "fixed"
            base.run_one(name, cfg, ds, seed, args.rounds, args.out, args.snr)
            result = json.loads(file.read_text(encoding="utf-8"))
            result["layout"] = args.layout
            result["control_wrapper_sha"] = base.sha(__file__)
            if args.layout == "width":
                result["wrapper_sha"] = base.sha(width.__file__)
            result["purpose"] = "post-core diagnostic; same regularizer on a digital channel"
            base.save(file, result)
