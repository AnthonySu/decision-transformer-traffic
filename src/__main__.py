"""Entry point for running EV-DT as a module: python -m src"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="ev-dt",
        description="EV-DT: Return-Conditioned Emergency Corridor Optimization",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("demo", help="Run quick environment demo")
    sub.add_parser("smoke", help="Run end-to-end smoke test")
    sub.add_parser("profile", help="Profile environment performance")
    sub.add_parser("figures", help="Generate demo figures for paper")

    train_p = sub.add_parser("train", help="Train Decision Transformer")
    train_p.add_argument("--config", default="configs/default.yaml")
    train_p.add_argument("--method", choices=["dt", "madt", "ppo", "dqn"], default="dt")

    eval_p = sub.add_parser("eval", help="Evaluate all methods")
    eval_p.add_argument("--config", default="configs/default.yaml")

    args = parser.parse_args()

    if args.command == "demo":
        from scripts.quick_demo import main as demo_main
        demo_main()
    elif args.command == "smoke":
        from scripts.smoke_test import main as smoke_main
        sys.exit(smoke_main())
    elif args.command == "profile":
        from scripts.profile_env import main as profile_main
        profile_main()
    elif args.command == "figures":
        from scripts.generate_demo_figures import main as fig_main
        fig_main()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
