import numpy as np
import argparse
import contextlib
import json
from pathlib import Path


parser = argparse.ArgumentParser(
    description="Print reults")
parser.add_argument("-i", '--path',
                    type=str,
                    default=None,
                    help='Results path')
parser.add_argument("-b", '--baseline_path',
                    type=str,
                    default=None,
                    help='Baseline path')
parser.add_argument("-a", '--add_header',
                    action="store_true",
                    default=False,
                    help='print header and exit')
args = parser.parse_args()
if args.add_header:
    print("| %20s |" % "PATH", "| %20s |" % "BASELINE", "| %12s |" % "ASR", "| %12s |" % "ACC", "| %12s |" %
          "ACC_LIT", "| %12s |" % "ASR_DIFF", "| %12s |" % "ACC_DIFF", "| %12s |" % "ACC_LIT_DIFF")
    exit(0)


def parse(path: Path):
    asr, acc, acc_lit = np.nan, np.nan, np.nan
    with contextlib.suppress(Exception):
        with open(path/"utility.json", "r") as f:
            data = json.load(f)["results"]
            acc = (data["arc_easy"]["acc,none"]+data["boolq"]
                   ["acc,none"]+data["hellaswag"]["acc,none"])/3.0*100
    with contextlib.suppress(Exception):
        with open(path/"utility_LitGPT.json", "r") as f:
            data = json.load(f)["results"]
            acc_lit = (data["arc_easy"]["acc"]+data["boolq"]
                       ["acc"]+data["hellaswag"]["acc"])/3.0*100
    with contextlib.suppress(Exception):
        with open(path/"harmfulness.json", "r") as f:
            asr = json.load(f)["ASR"]
    return asr, acc, acc_lit

if not args.path:
    print("path required")
    exit(-1)
args.path = Path(args.path)
asr, acc, acc_lit = parse(args.path)
asr_b, acc_b, acc_lit_b = np.nan, np.nan, np.nan
args.path = str(args.path.resolve())[-20:]
if args.baseline_path:
    args.baseline_path = Path(args.baseline_path)
    asr_b, acc_b, acc_lit_b = parse(args.baseline_path)
    args.baseline_path = str(args.baseline_path.resolve())[-20:]
asr_b = asr - asr_b
acc_b = acc - acc_b
acc_lit_b = acc_lit - acc_lit_b
fmt = "| %+10.3f%%  |"

print("| %20s |" % args.path, "| %20s |" % args.baseline_path, fmt %
      asr, fmt % acc, fmt % acc_lit, fmt % asr_b, fmt % acc_b, fmt % acc_lit_b)
