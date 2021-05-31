
import argparse


class Parser(object):
    def getParser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--logfile', type=str, default='BioEE-RL', help="Filename of log file")
        parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
        parser.add_argument('--epochPRE', type=int, default=50, help="Number of epoch on pretraining")
        parser.add_argument('--epochRL', type=int, default=50, help="Number of epoch on training with RL")
        parser.add_argument('--dim', type=int, default=100, help="Dimension of action and argument detection embedding")
        parser.add_argument('--statedim', type=int, default=300, help="Dimension of state")
        parser.add_argument('--hidden_dim', type=int, default=300, help="Dimension of hidden state in LSTM")
        parser.add_argument('--batchsize', type=int, default=16, help="Batch size on training")
        parser.add_argument('--batchsize_test', type=int, default=32, help="Batch size on testing")
        parser.add_argument('--print_per_batch', type=int, default=100, help="Print results every XXX batches")
        parser.add_argument('--sampleround', type=int, default=5, help="Sample round in RL")
        parser.add_argument('--numprocess', type=int, default=2, help="Number of process")
        parser.add_argument('--start', type=str, default='', help="Directory to load model")
        parser.add_argument('--test', type=bool, default=False, help="Set to True to inference")
        parser.add_argument('--pretrain', type=bool, default=False, help="Set to True to pretrain")
        parser.add_argument('--datapath', type=str, default='../data/MLEE/', help="Data directory")
        parser.add_argument('--testfile', type=str, default='test', help="Filename of test file")
        return parser
