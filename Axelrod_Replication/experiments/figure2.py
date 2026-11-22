import csv
from pathlib import Path
import numpy as np

from model import create_grid, run_simulation_optimized, count_regions, is_stable

F = 5
q = 15

n_values = [2, 5, 10, 12, 15, 20, 25]
repetitions = 40
max_steps = 10000000

output_dir = Path("data/figure2")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "figure2_runs.csv"

fieldnames = [
    "run", "n", "F", "q",
    "seed_grid", "seed_interactions",
    "steps", "changes", "stable", "regions"
]

completed = set()

if output_file.exists():
    with open(output_file, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames != fieldnames:
            raise RuntimeError("Nieprawidłowa struktura istniejącego pliku CSV.")

        for row in reader:
            if row["stable"] == "True":
                key = (int(row["n"]), int(row["run"]))

                if key in completed:
                    raise RuntimeError("Wykryto powtórzony przebieg w pliku CSV.")

                completed.add(key)

file_exists = output_file.exists()

with open(output_file, "a", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)

    if not file_exists:
        writer.writeheader()
        file.flush()

    for n in n_values:
        for run in range(1, repetitions + 1):
            key = (n, run)

            if key in completed:
                print(f"Pomijam ukończony przebieg: n={n}, run={run}")
                continue

            experiment_id = n * 1000 + run
            seed_grid = 100000 + experiment_id * 2
            seed_interactions = seed_grid + 1

            rng_grid = np.random.default_rng(seed_grid)
            rng_interactions = np.random.default_rng(seed_interactions)

            grid = create_grid(n, F, q, rng=rng_grid)

            result = run_simulation_optimized(
                grid,
                rng_interactions,
                max_steps=max_steps
            )

            stable = is_stable(result["grid"])

            if not stable:
                raise RuntimeError(
                    f"Nie osiągnięto stabilności: n={n}, run={run}, "
                    f"steps={result['steps']}"
                )

            regions = count_regions(result["grid"])

            writer.writerow({
                "run": run,
                "n": n,
                "F": F,
                "q": q,
                "seed_grid": seed_grid,
                "seed_interactions": seed_interactions,
                "steps": result["steps"],
                "changes": result["changes"],
                "stable": stable,
                "regions": regions
            })

            file.flush()
            completed.add(key)

            print(
                f"n={n} | run={run}/{repetitions} | "
                f"steps={result['steps']} | "
                f"changes={result['changes']} | "
                f"regions={regions}"
            )

print()
print("Zakończono zaplanowane symulacje.")
print("Wyniki zapisano do:", output_file)
