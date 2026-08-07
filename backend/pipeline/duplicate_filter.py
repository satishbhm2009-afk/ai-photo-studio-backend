import numpy as np
from backend.utils import compute_hash
from backend.config import settings

class DuplicateFilter:
    def __init__(self, hash_size: int = None):
        self.hash_size = hash_size or settings.DUPLICATE_HASH_SIZE
        self.seen_hashes = set()

    def is_duplicate(self, image: np.ndarray) -> bool:
        h = compute_hash(image, self.hash_size)
        if h in self.seen_hashes:
            return True
        self.seen_hashes.add(h)
        return False

    def reset(self):
        self.seen_hashes.clear()