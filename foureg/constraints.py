from dataclasses import dataclass, field

nan = float("nan")


@dataclass
class Constraints:
    angle: tuple[float, float] = field(default=(0, nan))
    scale: tuple[float, float] = field(default=(1, nan))
    tx: tuple[float, float] = field(default=(0, nan))
    ty: tuple[float, float] = field(default=(0, nan))
