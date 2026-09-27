"""Client gradient relations + one weighted AirComp for FedOSD projection.

FedOSD UCE/projection formulation: Pan et al., AAAI 2025, MIT source at
third_party/FedOSD, commit 41cc10635c6f2396795d0d9c6239ea181b7c9ee6.
The server solve accepts only sketches. Dense client vectors are used by the
client-side sketch/transmitter simulator and by explicitly named oracle audits.
"""
from dataclasses import dataclass, asdict
import math
import torch


def uce(logits, labels):
    py = logits.softmax(-1).gather(1, labels[:, None]).squeeze(1)
    return -torch.log1p(-py / 2).mean()


def normalized_rows(rows):
    norms = rows.norm(dim=1).clamp_min(1e-20)
    return rows / norms[:, None], norms


def srht(rows, size, seed):
    """Independent public signed Hadamard probes; rows are client vectors."""
    k, d = rows.shape
    padded = 1 << (d - 1).bit_length()
    if size > padded:
        raise ValueError('sketch dimension exceeds padded dimension')
    generator = torch.Generator(device=rows.device).manual_seed(seed)
    signs = torch.randint(0, 2, (padded,), generator=generator,
                          device=rows.device, dtype=torch.int32).to(rows.dtype) * 2 - 1
    indices = torch.randperm(padded, generator=generator, device=rows.device)[:size]
    values = torch.nn.functional.pad(rows, (0, padded - d)) * signs
    h = 1
    while h < padded:
        pairs = values.reshape(k, -1, 2, h)
        left, right = pairs[:, :, 0], pairs[:, :, 1]
        values = torch.stack((left + right, left - right), dim=2).reshape(k, padded)
        h *= 2
    return values[:, indices] / math.sqrt(size)


def quantize_rows(rows, bits):
    if bits >= 32:
        return rows.clone()
    levels = 2 ** (bits - 1) - 1
    scale = rows.abs().amax(dim=1, keepdim=True).clamp_min(1e-30) / levels
    return (rows / scale).round().clamp(-levels, levels) * scale


def relation_coefficients(sketches, rtol=1e-7):
    """Server API: last row belongs to the target; no dense vectors available."""
    retained = sketches[:-1].double().T
    target = sketches[-1].double()
    # GPU SVD avoids squaring the condition number in normal equations.
    left, singular, right_h = torch.linalg.svd(retained, full_matrices=False)
    cutoff = singular.max().clamp_min(1e-30) * rtol
    inv = torch.where(singular > cutoff, singular.reciprocal(), 0)
    c = right_h.T @ (inv * (left.T @ target))
    return torch.cat((-c, torch.ones(1, device=c.device, dtype=c.dtype))).to(sketches.dtype)


def oracle_projection(rows):
    unit, norms = normalized_rows(rows)
    weights = relation_coefficients(unit)
    residual = weights @ unit
    return residual, weights, norms[-1]


def rescale_fedosd(residual, target_norm):
    norm = residual.norm()
    if float(norm) < 1e-12:
        return torch.zeros_like(residual)
    return residual * (target_norm / norm)


@dataclass
class Channel:
    snr_db: float = 20.0
    weak_db: float = 0.0
    csi_std: float = 0.0
    repeats: int = 1
    peak_ratio: float = 4.0
    pilot_symbols_per_client: int = 64
    digital_peak_symbols: int = 0
    weak_index: int = 0


def aircomp_transmit(unit_rows, weights, channel, seed):
    """Physical simulator, not server inference. Average P<=1, peak P<=4.

    Two real entries per complex symbol, real positive fading after coherent
    phase compensation. Complex AWGN power N0=10**(-SNR/10). Estimated fading
    controls inversion; its relative Gaussian error is given by csi_std.
    Metadata required for eta consists of each weighted payload RMS and peak.
    """
    k, d = unit_rows.shape
    rng = torch.Generator(device=unit_rows.device).manual_seed(seed)
    h = torch.ones(k, device=unit_rows.device, dtype=unit_rows.dtype)
    # Worst link fixed independently of gradient values and selected coefficients.
    h[channel.weak_index] = 10 ** (-channel.weak_db / 20)
    csi = torch.randn(k, generator=rng, device=h.device, dtype=h.dtype) * channel.csi_std
    hhat = (h * (1 + csi)).clamp_min(1e-4)
    payload = unit_rows * weights[:, None]
    repair = torch.zeros_like(payload)
    repair_bits = 0
    if channel.digital_peak_symbols:
        # Known hybrid residual coding idea: digital sparse large coordinates,
        # analog remainder. This is not claimed as a novel coding mechanism.
        paired = torch.nn.functional.pad(payload, (0, d % 2)).reshape(k, -1, 2)
        count = min(channel.digital_peak_symbols, paired.shape[1])
        indices = paired.square().sum(-1).topk(count, dim=1).indices
        selected = paired.gather(1, indices[:, :, None].expand(-1, -1, 2))
        decoded = quantize_rows(selected.reshape(k, -1), 16).reshape(k, count, 2)
        digital_payload = torch.zeros_like(paired)
        digital_payload.scatter_(1, indices[:, :, None].expand(-1, -1, 2), decoded)
        repair = digital_payload.reshape(k, -1)[:, :d]
        repair_bits = k * (count * (math.ceil(math.log2(paired.shape[1])) + 32) + 32)
        payload = payload - repair
    padded = torch.nn.functional.pad(payload, (0, d % 2))
    complex_power = padded.reshape(k, -1, 2).square().sum(-1)
    avg = complex_power.mean(1)
    peak = complex_power.amax(1)
    limits_avg = hhat.square() / avg.clamp_min(1e-30)
    limits_peak = channel.peak_ratio * hhat.square() / peak.clamp_min(1e-30)
    eta2 = torch.minimum(limits_avg, limits_peak).min()
    gain_error = h / hhat
    mean = (payload * gain_error[:, None]).sum(0)
    n0 = 0.0 if math.isinf(channel.snr_db) else 10 ** (-channel.snr_db / 10)
    variance = n0 / (2 * float(eta2) * channel.repeats)
    noise = torch.randn(d, generator=rng, device=payload.device, dtype=payload.dtype)
    received = mean + noise * math.sqrt(variance) + repair.sum(0)
    tx_avg = eta2 * avg / hhat.square()
    tx_peak = eta2 * peak / hhat.square()
    n_symbols = (d + 1) // 2 * channel.repeats
    stats = {
        **asdict(channel), 'analog_symbols': n_symbols,
        'eta_squared': float(eta2), 'noise_variance_per_real': variance,
        'predicted_awgn_norm_rms': math.sqrt(d * variance),
        'max_average_tx_power': float(tx_avg.max()),
        'max_peak_tx_power': float(tx_peak.max()),
        'normalized_rf_energy': float(tx_avg.sum()) * n_symbols,
        'digital_peak_repair_bits': repair_bits,
    }
    return received, stats


def cost_ledger(d, k, method, sketch_size=512, sketch_bits=16, vector_bits=16,
                spectral_efficiency=2.0, repeats=1, pilots=64, digital_peak_symbols=0,
                topk_elements=8192):
    """Information-bit abstraction; counts all task-specific transmissions.

    One broadcast of model parameters is common to all methods. No coding gain,
    ARQ behavior, or measured hardware runtime is inferred from this ledger.
    """
    b = spectral_efficiency
    model_dl_bits = 32 * d
    base_metadata = k * (64 + 32)  # norm/peak and vector scale/header
    control_dl_bits = 64
    analog = 0
    if method.startswith('relation'):
        uplink_bits = k * (sketch_size * sketch_bits + 32) + base_metadata
        control_dl_bits += 32 * k + 32  # coefficients and common MAC amplitude eta
        analog = (d + 1) // 2 * repeats
        if digital_peak_symbols:
            uplink_bits += k * (digital_peak_symbols * (math.ceil(math.log2((d+1)//2)) + 32) + 32)
    elif method == 'compressed8':
        uplink_bits = k * topk_elements * (8 + math.ceil(math.log2(d))) + base_metadata
    elif method == 'aggregate_only':
        # Optimistic mean-only comparator: target vector digital, exact retained MAC.
        uplink_bits = d * 16 + base_metadata
        analog = (d + 1) // 2
        control_dl_bits += 32  # common MAC amplitude eta
    else:
        uplink_bits = k * d * vector_bits + base_metadata
    pilot_uses = k * pilots
    return {
        'digital_uplink_bits': uplink_bits,
        'model_downlink_bits': model_dl_bits,
        'control_downlink_bits': control_dl_bits,
        'pilot_channel_uses': pilot_uses,
        'analog_channel_uses': analog,
        'uplink_channel_uses': uplink_bits / b + analog + pilot_uses,
        'total_channel_uses': (uplink_bits + model_dl_bits + control_dl_bits) / b + analog + pilot_uses,
        'spectral_efficiency_bits_per_complex_use': b,
    }


def projected_direction(rows, method, seed, sketch_size=512, sketch_bits=16,
                        channel=None, topk_elements=8192):
    """Federated simulator. 'rows' reside at clients; target is last row."""
    unit, norms = normalized_rows(rows)
    stats = {}
    if method in ('full', 'digital16', 'digital8'):
        bits = {'full': 32, 'digital16': 16, 'digital8': 8}[method]
        received_individual = quantize_rows(unit, bits)
        weights = relation_coefficients(received_individual)
        residual = weights @ received_individual
    elif method == 'compressed8':
        count = min(topk_elements, unit.shape[1])
        indices = unit.abs().topk(count, dim=1).indices
        selected = unit.gather(1, indices)
        decoded = quantize_rows(selected, 8)
        received_individual = torch.zeros_like(unit).scatter_(1, indices, decoded)
        weights = relation_coefficients(received_individual)
        residual = weights @ received_individual
    elif method.startswith('relation'):
        packets = quantize_rows(srht(unit, sketch_size, seed), sketch_bits)
        weights = relation_coefficients(packets)
        if method == 'relation_ideal':
            residual = weights @ unit  # ideal MAC, no individual decoding
        else:
            residual, stats = aircomp_transmit(unit, weights, channel or Channel(), seed + 9176)
        stats['sketch_residual_norm'] = float((weights @ packets).norm())
        stats['coefficient_norm'] = float(weights.norm())
    elif method == 'aggregate_only':
        # Target available separately; retained individual relationships lost.
        aggregate = rows[:-1].mean(0)
        unit_aggregate = aggregate / aggregate.norm().clamp_min(1e-20)
        residual = unit[-1] - (unit[-1] @ unit_aggregate) * unit_aggregate
    else:
        raise ValueError(method)
    stats['pre_rescale_norm'] = float(residual.norm())
    return rescale_fedosd(residual, norms[-1]), stats


def recovery_gradients(rows, displacement):
    """FedOSD paper Algorithm 1: conditional local post-training projection."""
    norm2 = displacement.square().sum()
    if float(norm2) < 1e-20:
        return rows
    norms = rows.norm(dim=1, keepdim=True)
    dots = rows @ displacement
    projected = rows - (dots.clamp_min(0) / norm2)[:, None] * displacement
    return projected / projected.norm(dim=1, keepdim=True).clamp_min(1e-20) * norms
