import numpy as np


def create_grid(n, F, q, seed=None, rng=None):
    if n < 1 or F < 1 or q < 1:
        raise ValueError("Parametry n, F i q muszą być dodatnie.")

    if rng is None:
        rng = np.random.default_rng(seed)

    grid = rng.integers(0, q, size=(n, n, F))

    return grid


def get_neighbors(row, col, n):
    neighbors = []

    if row > 0:
        neighbors.append((row - 1, col))

    if row < n - 1:
        neighbors.append((row + 1, col))

    if col > 0:
        neighbors.append((row, col - 1))

    if col < n - 1:
        neighbors.append((row, col + 1))

    return neighbors


def interaction_step(grid, rng, return_details=False):
    n = grid.shape[0]
    F = grid.shape[2]

    row = rng.integers(0, n)
    col = rng.integers(0, n)

    neighbors = get_neighbors(row, col, n)
    neighbor_index = rng.integers(0, len(neighbors))
    neighbor_row, neighbor_col = neighbors[neighbor_index]

    agent = grid[row, col]
    neighbor = grid[neighbor_row, neighbor_col]

    common = int(np.sum(agent == neighbor))
    probability = common / F

    details = {
        "agent_row": int(row + 1),
        "agent_col": int(col + 1),
        "neighbor_row": int(neighbor_row + 1),
        "neighbor_col": int(neighbor_col + 1),
        "common_features": common,
        "probability": probability,
        "changed": False,
        "feature": None,
        "value_before": None,
        "value_after": None
    }

    if 0 < common < F:
        if rng.random() < probability:
            different_features = np.where(agent != neighbor)[0]
            feature = rng.choice(different_features)

            old_value = int(agent[feature])
            new_value = int(neighbor[feature])

            agent[feature] = new_value

            details["changed"] = True
            details["feature"] = int(feature + 1)
            details["value_before"] = old_value
            details["value_after"] = new_value

    if return_details:
        return details

    return details["changed"]


def is_stable(grid):
    n = grid.shape[0]
    F = grid.shape[2]

    for row in range(n):
        for col in range(n):
            agent = grid[row, col]

            for neighbor_row, neighbor_col in get_neighbors(row, col, n):
                neighbor = grid[neighbor_row, neighbor_col]

                common = np.sum(agent == neighbor)

                if 0 < common < F:
                    return False

    return True


def run_simulation(grid, rng, max_steps=100000, save_history=False):
    steps = 0
    changes = 0
    history = []

    if save_history:
        history.append(grid.copy())

    while not is_stable(grid):
        if steps >= max_steps:
            raise RuntimeError(
                "Osiagnieto limit prob przed stabilizacja."
            )

        changed = interaction_step(grid, rng)
        steps += 1

        if changed:
            changes += 1

            if save_history:
                history.append(grid.copy())

    result = {
        "grid": grid,
        "steps": steps,
        "changes": changes
    }

    if save_history:
        result["history"] = history

    return result


def count_regions(grid):
    n = grid.shape[0]

    visited = set()
    regions = 0

    for row in range(n):
        for col in range(n):

            if (row, col) in visited:
                continue

            regions += 1
            culture = grid[row, col].copy()

            stack = [(row, col)]
            visited.add((row, col))

            while stack:
                current_row, current_col = stack.pop()

                neighbors = get_neighbors(
                    current_row, current_col, n
                )

                for nr, nc in neighbors:

                    if (nr, nc) in visited:
                        continue

                    if np.array_equal(grid[nr, nc], culture):
                        visited.add((nr, nc))
                        stack.append((nr, nc))

    return regions


def count_active_pairs(grid):
    n = grid.shape[0]
    F = grid.shape[2]
    active_pairs = 0

    for row in range(n):
        for col in range(n):
            agent = grid[row, col]

            if row < n - 1:
                neighbor = grid[row + 1, col]
                common = np.sum(agent == neighbor)

                if 0 < common < F:
                    active_pairs += 1

            if col < n - 1:
                neighbor = grid[row, col + 1]
                common = np.sum(agent == neighbor)

                if 0 < common < F:
                    active_pairs += 1

    return active_pairs


def is_active_pair(grid, row1, col1, row2, col2):
    agent = grid[row1, col1]
    neighbor = grid[row2, col2]

    common = np.sum(agent == neighbor)
    F = grid.shape[2]

    return 0 < common < F


def update_active_pairs(grid, row, col, old_culture, active_pairs):
    n = grid.shape[0]
    F = grid.shape[2]

    for neighbor_row, neighbor_col in get_neighbors(row, col, n):
        neighbor = grid[neighbor_row, neighbor_col]

        common_before = np.sum(old_culture == neighbor)
        common_after = np.sum(grid[row, col] == neighbor)

        active_before = 0 < common_before < F
        active_after = 0 < common_after < F

        active_pairs += int(active_after) - int(active_before)

    return active_pairs


def run_simulation_optimized(grid, rng, max_steps=100000, save_history=False):
    steps = 0
    changes = 0
    history = []

    active_pairs = count_active_pairs(grid)

    if save_history:
        history.append(grid.copy())

    while active_pairs > 0:
        if steps >= max_steps:
            raise RuntimeError("Osiagnieto limit prob przed stabilizacja.")

        details = interaction_step(grid, rng, return_details=True)
        steps += 1

        if details["changed"]:
            changes += 1

            row = details["agent_row"] - 1
            col = details["agent_col"] - 1
            feature = details["feature"] - 1

            old_culture = grid[row, col].copy()
            old_culture[feature] = details["value_before"]

            active_pairs = update_active_pairs(
                grid, row, col, old_culture, active_pairs
            )

            if save_history:
                history.append(grid.copy())

    result = {
        "grid": grid,
        "steps": steps,
        "changes": changes
    }

    if save_history:
        result["history"] = history

    return result


