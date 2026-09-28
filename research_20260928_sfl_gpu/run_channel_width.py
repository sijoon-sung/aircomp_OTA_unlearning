"""Actual same-input channel branches, reusing the frozen cohort training loop.

Each sensor computes distinct CNN channels for the SAME full 28x28 image.
Raw input broadcast is charged; this is not ordinary horizontal SFL.
"""
import argparse
from pathlib import Path
import json
import torch
from torch import nn
import run_sfl as base


class WidthModel(base.Model):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cs = [392*w for w in self.ws]
        self.projections = nn.ModuleList([nn.Linear(c, cfg["q"], bias=False) for c in self.cs])

    def forward(self, x, seed, ledger=None, snr=20., condition="rician", csi=0.):
        hs = [enc(x.contiguous()) for enc in self.encoders]
        kind, bits = self.cfg["kind"], self.cfg["bits"]
        h, he = base.channels(seed, condition, csi)
        if kind == "feature":
            hs = [base.Wire.apply(z, bits, 8) for z in hs]
        us = [p(z) for p, z in zip(self.projections, hs)]
        energy = None
        if kind == "ota":
            z, energy = base.ota(us, h, he, snr, seed+817, ledger)
        elif kind == "digital":
            z = sum(base.Wire.apply(u, bits, 32) for u in us)
        else:
            z = sum(us)
        if kind not in ["feature", "ideal"]:
            z = base.Wire.apply(z, 32, 8)
        self.latest_z, self.latest_us, self.latest_h = z, us, h
        if ledger is not None:
            ledger.frame(self.cfg, len(x), self.cs, h, snr, energy)
        return self.head(self.shared(z)+self.bias)

    def compute(self):
        out = super().compute()
        encs = [4*v for v in out["encoder_MAC"]]
        macs = [a+b for a, b in zip(encs, out["projection_MAC"])]
        out.update(encoder_MAC=encs, client_MACs=macs, client_forward_MAC_per_sample=max(macs))
        return out


class InputLedger(base.Ledger):
    def __init__(self):
        super().__init__()
        self.x["raw_input_dl_uses"] = 0.

    def frame(self, config, batch, feature_dim, h, snr, ota_energy=None, signal_dim=None):
        super().frame(config, batch, feature_dim, h, snr, ota_energy, signal_dim)
        if config["kind"] != "ideal":
            # Shared image originates at an edge coordinator: uint8, no entropy coding.
            uses = base.CP*(batch*28*28*8+96)/float(base.rates(h, snr).min())
            self.x["raw_input_dl_uses"] += uses
            self.x["dl_energy"] += uses


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent/"width20")
    ap.add_argument("--seeds", default="311,312,313")
    ap.add_argument("--rounds", type=int, default=80)
    ap.add_argument("--snr", type=float, default=20.)
    ap.add_argument("--methods", default="Latent4-Q16-W8,OTA-Q16-W8,OTA-Stein-Q16-W8,OTA-Reg-Q16-W8")
    args = ap.parse_args()
    base.Model, base.Ledger = WidthModel, InputLedger
    for seed in map(int, args.seeds.split(",")):
        ds = base.data(args.data, seed)
        for name in args.methods.split(","):
            file = args.out/f"{name}_seed{seed}.json"
            if file.exists():
                print("SKIP completed", file.name, flush=True)
                continue
            base.run_one(name, base.CONFIGS[name], ds, seed, args.rounds, args.out, args.snr)
            result = json.loads(file.read_text(encoding="utf-8"))
            result["layout"] = "same_full_input_channel_branches"
            result["wrapper_sha"] = base.sha(__file__)
            result["input_broadcast"] = "uint8 full image every training batch; included in ledger"
            base.save(file, result)
