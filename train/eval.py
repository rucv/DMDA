from test import test

import argparse

parser = argparse.ArgumentParser(description='PyTorch DA Implementation')
parser.add_argument('--gpu',    type=int,   default=0,       help='Which GPU you want to use')
parser.add_argument('--alg',    type=int,   default=0,       help='Which Algorithm you want to use')
parser.add_argument('--source', type=str,   default='SVHN',  help='Name of SOURCE Domain')
parser.add_argument('--target', type=str,   default='MNIST', help='Name of TARGET Domain')
args = parser.parse_args()



source = args.source
target = args.target
alg = args.alg
gpu = args.gpu

test(source,target,-2,alg,gpu)
# test(source,target,-2,alg,gpu)
# test(source,target,-2,alg,gpu)
# test(source,target,-2,alg,gpu)
# test(source,target,-2,alg,gpu)
