# ising_converter.py
"""
QUBO → Ising Hamiltonian conversion

Converts binary variables x ∈ {0,1}
to spin variables s ∈ {-1, +1}

Mapping: s = 2x - 1
"""

import numpy as np


class IsingModel:
    def __init__(self, h, J, offset):
        """
        h: dict {i: coefficient}
        J: dict {(i, j): coefficient}
        offset: constant energy shift
        """
        self.h = h
        self.J = J
        self.offset = offset


def qubo_to_ising(Q):
    """
    Convert QUBO matrix Q to Ising parameters.

    Q: dict {(i, j): value} or 2D numpy array

    Returns:
        IsingModel
    """

    h = {}
    J = {}
    offset = 0.0

    # Convert QUBO → Ising
    for (i, j), value in Q.items():
        if i == j:
            # Linear term
            h[i] = h.get(i, 0) + value / 2
            offset += value / 2
        else:
            # Quadratic term
            J[(i, j)] = value / 4
            h[i] = h.get(i, 0) + value / 4
            h[j] = h.get(j, 0) + value / 4
            offset += value / 4

    return IsingModel(h, J, offset)


if __name__ == "__main__":
    # Example usage (toy QUBO)
    Q = {
        (0, 0): 1,
        (1, 1): 1,
        (0, 1): -2
    }

    ising = qubo_to_ising(Q)

    print("h:", ising.h)
    print("J:", ising.J)
    print("offset:", ising.offset)