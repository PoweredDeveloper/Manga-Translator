#!/usr/bin/env python3

import sys
import argparse
from manga_processor import MT_Processor, RunMode

parser = argparse.ArgumentParser()
parser.add_argument("--height", help = "Sets height for window", type = int, default = 720)
parser.add_argument("--width", help = "Sets width for window", type = int, default = 1280)
args = parser.parse_args()

if __name__ == "__main__":
    processor = MT_Processor(sys.argv)
    processor.run(args.width, args.height, mode = RunMode.MT_MODE_WINDOW)
