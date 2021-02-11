import os
import argparse
from statistics import mean

from utils.metrics import *

parser = argparse.ArgumentParser()

parser.add_argument("--hypothesis_file", type=str, required=True)
parser.add_argument("--source_file", type=str, required=True)
parser.add_argument("--reference_file", type=str, required=True)

args = parser.parse_args()
    
def main() -> None:
    with open(args.hypothesis_file, 'r') as f:
        hypothesis = [line.strip() for line in f] 
    with open(args.source_file, 'r') as f:
        source = [line.strip() for line in f] 
    with open(args.reference_file, 'r') as f:
        reference = [line.strip() for line in f] 

    bleu4_score = mean(bleu([r], h) for r, h in zip(reference, hypothesis))
    bleu3_score = mean(bleu([r], h, weights=(1/3, 1/3, 1/3)) for r, h in zip(reference, hypothesis))
    ibleu_score = mean(ibleu([r], h, s) for r, h, s in zip(reference, hypothesis, source))

    print('BLEU4:\t%2.2f' % bleu4_score)
    print('BLEU3:\t%2.2f' % bleu3_score)
    print('iBLEU:\t%2.2f' % ibleu_score)

if __name__ == '__main__':
    main()
