import argparse
from l2r import Rank

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('-b', '--model', default='LambdaRank', help='Backend to use. (Default: grobid)')
    arg_parser.add_argument('-t', '--train', default='./train.npy', help='Training data store in .npy')
    arg_parser.add_argument('-n', '--test', default='./train.npy', help='Testing data store in .npy')

    arg_parser.add_argument('-k', '--topk', default=3, help='NDCG@k for ranking test')

    arg_parser.add_argument('-nf', '--n_feature', default=6, help='number of feature')
    arg_parser.add_argument('-h1', '--h1_units', default=512, help='number od cell layer 1')
    arg_parser.add_argument('-n2', '--h2_units', default=256, help='number od cell layer 2')
    arg_parser.add_argument('-e', '--epoch', default=10, help='Training epoch')
    arg_parser.add_argument('-l', '--lr', default=0.0001, help='learning rate')
    arg_parser.add_argument('-r', '--nt', default=20, help='number of tree')

    args = arg_parser.parse_args()
    rank = Rank(args.model)
    rank.handler(args.train, args.n_feature, args.h1_units, args.h2_units, args.epoch, args.lr, args.nt)