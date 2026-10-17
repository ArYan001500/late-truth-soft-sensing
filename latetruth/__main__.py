"""Command-line entry point."""
import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('reproduce')
    run.add_argument('--tier', choices=['paper','quick','main','full'], required=True)
    run.add_argument('--data-dir', default=os.environ.get('DATA_DIR', './data'))
    run.add_argument('--jobs', type=int, choices=range(1,9), default=8)
    run.add_argument('--offline', action='store_true', help='Use existing archives without a network request')
    args = parser.parse_args()
    try:
        from latetruth.reproduce import run
        run(args.tier, args.data_dir, args.jobs, args.offline)
    except Exception as exc:
        print(f'FAIL: {args.tier}: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
