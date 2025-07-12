from typing import List, Tuple, Dict, Optional
def get_rect_center(rect: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Compute the center of a rectangle."""
    xs = [p[0] for p in rect]
    ys = [p[1] for p in rect]
    return (sum(xs) / 4, sum(ys) / 4)