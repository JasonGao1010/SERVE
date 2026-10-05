"""Selected SERVE computation: measured-return evidence and same-class readout."""

import math

import torch


MIN_RETURN_SCALE = .001
LOG_RETURN_PEAK = math.log(2 / (math.pi * math.sqrt(3) * MIN_RETURN_SCALE))


def return_density(log_range, mean, scale, log_weight, supported):
    """Evaluate a predicted Student-t mixture at measured log range.

    log_range: [N]; mean, scale, log_weight: [N, C, K]; supported: [N].
    Mixture parameters come from target-excluded context. Scales are at least
    MIN_RETURN_SCALE; log_weight contains normalized log mixture weights.
    The density is defined with respect to d(log r), with three degrees of freedom.
    """
    standardized = (log_range[:, None, None] - mean) / scale
    log_compatibility = -2 * torch.log1p(standardized.square() / 3)
    log_prob = -math.log(math.pi * math.sqrt(3) / 2) - scale.log() + log_compatibility
    return dict(log_prob=log_prob, weight=log_weight, supported=supported)


def geometry_energy(prediction):
    """Normalize return density by the peak of a minimum-scale Student-t.

    The scale penalty remains in each component. A cell without neighboring
    context contributes zero return energy, leaving the appearance evidence.
    """
    energy = LOG_RETURN_PEAK - torch.logsumexp(
        prediction["weight"] + prediction["log_prob"], -1
    )
    return torch.where(prediction["supported"][:, None], energy.clamp_min(0), 0.)


def same_class_readout(appearance_energy, prediction):
    """Combine appearance and measured-return evidence within each class.

    appearance_energy has shape [N, C]. Lower energy means greater support.
    The least joint energy determines the semantic class and the raw unknown
    score; larger raw scores indicate weaker best-class support.
    """
    energy = appearance_energy + geometry_energy(prediction)
    return dict(
        energy=energy,
        semantic=energy.argmin(-1),
        semantic_only=appearance_energy.argmin(-1),
        raw_score=energy.amin(-1),
        confidence=(-energy).softmax(-1).amax(-1),
    )


def main():
    """Trace three measured returns through fixed, illustrative class densities."""
    torch.set_num_threads(1)
    measured = torch.tensor([12., 20., 25.], dtype=torch.float64)
    centers = torch.tensor([12., 20., 32.], dtype=torch.float64).log()
    offsets = torch.tensor([-.02, 0., .02], dtype=torch.float64)
    mean = (centers[None, :, None] + offsets).expand(3, -1, -1)
    scale = torch.full_like(mean, .025)
    log_weight = torch.zeros_like(mean).log_softmax(-1)
    supported = torch.ones(3, dtype=torch.bool)
    appearance = measured.new_tensor([[.1, .9, 1.6]]).expand(3, -1)
    prediction = return_density(measured.log(), mean, scale, log_weight, supported)
    result = same_class_readout(appearance, prediction)

    print("Measured ranges (m):", measured.tolist())
    print("Appearance classes:", result["semantic_only"].tolist())
    print("Joint classes:", result["semantic"].tolist())
    print("Raw unknown scores:", [round(x, 4) for x in result["raw_score"].tolist()])


if __name__ == "__main__":
    main()
