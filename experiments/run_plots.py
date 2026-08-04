import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.reproduce_figures import run_reproduce_figures
from scripts.reproduce_tables import run_reproduce_tables


def run():
    print("=" * 80)
    print("Generating paper figures")
    print("=" * 80)
    run_reproduce_figures()
    print("\nFigure generation completed.")

    print("=" * 80)
    print("Generating paper figures")
    print("=" * 80)
    run_reproduce_tables()
    print("\nFigure generation completed.")


if __name__ == "__main__":
    run()
