"""Independent float64 checks of scalar bounds and exact-preservation geometry.

Standard library only. These random fixtures are not the unavailable original
124/200 trajectory, a trained architecture, or a general constrained solver.
"""
from __future__ import annotations

import json
import math
import random


def dot(a, b):
    assert len(a) == len(b)
    return math.fsum(x * y for x, y in zip(a, b))


def norm(a):
    return math.sqrt(dot(a, a))


def scalar_step(a, b, slack):
    if not all(math.isfinite(value) for value in (a, b, slack)):
        raise ValueError('Nonfinite constraint')
    if a < 0 or slack < 0 or (a == 0 and b != 0):
        raise ValueError('Invalid or infeasible constraint')
    if a == 0:
        return 1.0
    discriminant = math.sqrt(b * b + a * slack)
    root = slack / (discriminant + b) if b >= 0 and discriminant + b > 0 else (-b + discriminant) / a
    return min(1.0, max(0.0, root))


def complement(vector, basis):
    result = vector[:]
    for _ in range(2):
        for direction in basis:
            coefficient = dot(result, direction)
            result = [x - coefficient * y for x, y in zip(result, direction)]
    return result


def projected_update(residual, projected_key, budget):
    if budget < 0 or not math.isfinite(budget):
        raise ValueError('Invalid norm budget')
    en, pn = norm(residual), norm(projected_key)
    if en == 0 or pn == 0 or budget == 0:
        return [[0.0 for _ in projected_key] for _ in residual]
    magnitude = min(budget, en / pn)
    return [[magnitude * (x / en) * (y / pn) for y in projected_key] for x in residual]


def main():
    rng = random.Random(73489)
    overshoot, root_gap = 0.0, 0.0
    for _ in range(10000):
        residual = [rng.uniform(-3, 3) for _ in range(4)]
        change = [rng.uniform(-3, 3) for _ in range(4)]
        a, b, slack = dot(change, change), dot(residual, change), rng.uniform(0, 3)
        ceiling = dot(residual, residual) + slack
        step = scalar_step(a, b, slack)
        after = math.fsum((x + step * y) ** 2 for x, y in zip(residual, change))
        overshoot = max(overshoot, after - ceiling)
        if a + 2 * b <= slack:
            oracle = 1.0
        else:
            low, high = 0.0, 1.0
            for _ in range(60):
                middle = (low + high) / 2
                if a * middle * middle + 2 * b * middle <= slack:
                    low = middle
                else:
                    high = middle
            oracle = (low + high) / 2
        root_gap = max(root_gap, abs(step - oracle))
        assert after <= ceiling + 1e-12
        assert abs(step - oracle) < 1e-12
    assert scalar_step(0, 0, 0) == 1
    assert scalar_step(1, 0, 0) == 0
    assert scalar_step(1, -1, 0) == 1
    for invalid in [(1, 0, -1), (0, 1, 0), (math.nan, 0, 0)]:
        try:
            scalar_step(*invalid)
        except ValueError:
            continue
        raise AssertionError('Invalid constraint accepted')

    rng = random.Random(9184)
    leakage, objective_gap, budget_excess = 0.0, 0.0, 0.0
    for _ in range(1000):
        basis = []
        for _ in range(8):
            vector = complement([rng.gauss(0, 1) for _ in range(12)], basis)
            length = norm(vector)
            basis.append([x / length for x in vector])
        key = [rng.gauss(0, 1) for _ in range(12)]
        length = norm(key)
        key = [x / length for x in key]
        p = complement(key, basis)
        residual = [rng.gauss(0, 1) for _ in range(6)]
        budget = rng.random() * 3 * norm(residual)
        update = projected_update(residual, p, budget)
        actual = norm([value - dot(row, key) for value, row in zip(residual, update)]) ** 2
        optimum = max(norm(residual) - budget * norm(p), 0.0) ** 2
        old_change = max(abs(dot(row, query)) for row in update for query in basis)
        used = math.sqrt(math.fsum(x * x for row in update for x in row))
        leakage = max(leakage, old_change)
        objective_gap = max(objective_gap, abs(actual - optimum))
        budget_excess = max(budget_excess, used - budget)
        assert old_change < 1e-12
        assert abs(actual - optimum) < 1e-10
        assert used <= budget + 1e-12

    assert projected_update([1, 2], [0, 0], 10) == [[0, 0], [0, 0]]
    assert projected_update([0, 0], [1, 0], 10) == [[0, 0], [0, 0]]
    assert projected_update([1, 2], [1, 0], 0) == [[0, 0], [0, 0]]
    key = [1 / math.sqrt(2), 1 / math.sqrt(2)]
    assert scalar_step(key[0] ** 2, 0, 0) == 0
    exact = projected_update([1.0], [0.0, key[1]], math.sqrt(2))
    assert exact[0][0] == 0 and abs(dot(exact[0], key) - 1) < 1e-15
    capped = projected_update([1.0], [0.0, key[1]], 1.0)
    assert 0 < dot(capped[0], key) < 1

    print(json.dumps({'status': 'passed', 'scalar_random_checks': 10000,
                      'scalar_max_ceiling_overshoot': overshoot,
                      'scalar_max_bisection_step_difference': root_gap,
                      'projection_random_checks': 1000, 'projection_shape': [6, 12],
                      'projection_witness_rank': 8, 'max_witness_prediction_change': leakage,
                      'max_closed_form_objective_difference': objective_gap,
                      'max_norm_budget_excess': budget_excess,
                      'boundary_checks': 'zero residual, zero projected key, zero budget, invalid constraints, repair, and blocked-but-representable example',
                      'scope': 'Independent fixtures; not a replay of the reported 124/200 trajectory or evidence of trained-model quality.'}, indent=2))


if __name__ == '__main__':
    main()
