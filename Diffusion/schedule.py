"""Noise schedule for absorbing-state discrete diffusion.

Continuous-time (MDLM-style) formulation: time t in [0, 1] with a survival
function alpha_bar(t) = P(a token is still its clean value at time t). The
per-token probability of having been absorbed to [mask] by time t is therefore
    mask_prob(t) = 1 - alpha_bar(t).

Linear schedule (the project default):
    alpha_bar(t) = 1 - t   ->   mask_prob(t) = t
so at t=0 nothing is masked and at t=1 everything is. Masking is independent per
token, so a training batch can jump straight to any t in one shot.

Note there is NO square root here -- that term is a Gaussian/DDPM variance
artifact. Discrete probabilities combine linearly: alpha_bar + (1 - alpha_bar) = 1.

alpha_bar/mask_prob are plain arithmetic, so they accept either Python floats or
torch tensors of t. Only "linear" is implemented; the class leaves room for
cosine / log-linear schedules later.
"""


class NoiseSchedule:
    """Maps continuous time t in [0, 1] to absorbing-state masking probabilities."""

    def __init__(self, kind: str = "linear"):
        if kind != "linear":
            raise NotImplementedError(
                f"noise schedule {kind!r} not implemented; only 'linear' is available")
        self.kind = kind

    def alpha_bar(self, t):
        """Survival probability: P(token still clean at time t). t in [0, 1]."""
        return 1.0 - t

    def mask_prob(self, t):
        """Absorption probability: P(token is [mask] by time t) = 1 - alpha_bar(t)."""
        return t


if __name__ == "__main__":
    schedule = NoiseSchedule("linear")
    print("linear schedule: mask probability over time")
    print(f"  {'t':>5}  {'alpha_bar':>10}  {'mask_prob':>10}")
    for t in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
        print(f"  {t:>5.2f}  {schedule.alpha_bar(t):>10.3f}  {schedule.mask_prob(t):>10.3f}")
