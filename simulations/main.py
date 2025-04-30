import argparse
from simulation_runner import simulate_evs
from utils.initializers import clear_old_results
from config import EV_PATHS


def main():
    parser = argparse.ArgumentParser(description="Run smart and/or non-smart EV simulations.")
    parser.add_argument(
        "--mode",
        choices=["smart", "non_smart", "both", "non_smart_no_public", "smart_oracle"],
        default="both",
        help="Choose which simulation mode to run."
    )
    args = parser.parse_args()

    clear_old_results()
    simulate_evs(EV_PATHS, args.mode)


if __name__ == "__main__":
    main()


