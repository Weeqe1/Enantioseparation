import argparse

def parse_args(TRANSFER_TARGET):
    if TRANSFER_TARGET == 'Cyclosil_B':
        parser = argparse.ArgumentParser(description='Graph data mining with GAT')
        parser.add_argument('--devices', type=int, nargs='+', default=[0], help='Which GPUs to use')
        parser.add_argument('--batch_size', type=int, default=8, help='Input batch size for training')
        parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
        parser.add_argument('--heads', type=int, default=16, help='Number of attention heads in GAT')
        parser.add_argument('--hidden_dim', type=int, default=128,
                            help='Dimensionality of hidden units in GAT')
        parser.add_argument('--l1_lambda', type=float, default=0.0001, help='l1_lambda')
        parser.add_argument('--l2_lambda', type=float, default=1e-05, help='l2_lambda')
        parser.add_argument('--lr', type=float, default=1e-05, help='Learning rate')
        parser.add_argument('--num_layers', type=int, default=4, help='Number of GAT layers')
        parser.add_argument('--output_dim', type=int, default=32,
                            help='Dimensionality of output units in GAT')
        parser.add_argument('--weight_decay', type=float, default=1e-05, help='Weight decay')
        parser.add_argument('--epochs', type=int, default=50000, help='Number of epochs to train')
        args = parser.parse_args()
        return args

    if TRANSFER_TARGET == 'Cyclodex_B':
        parser = argparse.ArgumentParser(description='Graph data mining with GAT')
        parser.add_argument('--devices', type=int, nargs='+', default=[0], help='Which GPUs to use')
        parser.add_argument('--batch_size', type=int, default=8, help='Input batch size for training')
        parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
        parser.add_argument('--heads', type=int, default=8, help='Number of attention heads in GAT')
        parser.add_argument('--hidden_dim', type=int, default=128,
                            help='Dimensionality of hidden units in GAT')
        parser.add_argument('--l1_lambda', type=float, default=0.001, help='l1_lambda')
        parser.add_argument('--l2_lambda', type=float, default=0.0001, help='l2_lambda')
        parser.add_argument('--lr', type=float, default=1e-05, help='Learning rate')
        parser.add_argument('--num_layers', type=int, default=16, help='Number of GAT layers')
        parser.add_argument('--output_dim', type=int, default=12,
                            help='Dimensionality of output units in GAT')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay')
        parser.add_argument('--epochs', type=int, default=50000, help='Number of epochs to train')
        args = parser.parse_args()
        return args

    if TRANSFER_TARGET == 'HP_chiral_20β':
        parser = argparse.ArgumentParser(description='Graph data mining with GAT')
        parser.add_argument('--devices', type=int, nargs='+', default=[0], help='Which GPUs to use')
        parser.add_argument('--batch_size', type=int, default=8, help='Input batch size for training')
        parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
        parser.add_argument('--heads', type=int, default=8, help='Number of attention heads in GAT')
        parser.add_argument('--hidden_dim', type=int, default=32,
                            help='Dimensionality of hidden units in GAT')
        parser.add_argument('--l1_lambda', type=float, default=1e-05, help='l1_lambda')
        parser.add_argument('--l2_lambda', type=float, default=0.01, help='l2_lambda')
        parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
        parser.add_argument('--num_layers', type=int, default=14, help='Number of GAT layers')
        parser.add_argument('--output_dim', type=int, default=8,
                            help='Dimensionality of output units in GAT')
        parser.add_argument('--weight_decay', type=float, default=1e-05, help='Weight decay')
        parser.add_argument('--epochs', type=int, default=50000, help='Number of epochs to train')
        args = parser.parse_args()
        return args

    if TRANSFER_TARGET == 'CP_Cyclodextrin_β_2,3,6_M_19':
        parser = argparse.ArgumentParser(description='Graph data mining with GAT')
        parser.add_argument('--devices', type=int, nargs='+', default=[0], help='Which GPUs to use')
        parser.add_argument('--batch_size', type=int, default=8, help='Input batch size for training')
        parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate')
        parser.add_argument('--heads', type=int, default=6, help='Number of attention heads in GAT')
        parser.add_argument('--hidden_dim', type=int, default=128,
                            help='Dimensionality of hidden units in GAT')
        parser.add_argument('--l1_lambda', type=float, default=0.001, help='l1_lambda')
        parser.add_argument('--l2_lambda', type=float, default=0.001, help='l2_lambda')
        parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
        parser.add_argument('--num_layers', type=int, default=12, help='Number of GAT layers')
        parser.add_argument('--output_dim', type=int, default=32,
                            help='Dimensionality of output units in GAT')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay')
        parser.add_argument('--epochs', type=int, default=50000, help='Number of epochs to train')
        args = parser.parse_args()
        return args

    if TRANSFER_TARGET == 'CP_Chirasil_D_Val':
        parser = argparse.ArgumentParser(description='Graph data mining with GAT')
        parser.add_argument('--devices', type=int, nargs='+', default=[0], help='Which GPUs to use')
        parser.add_argument('--batch_size', type=int, default=8, help='Input batch size for training')
        parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
        parser.add_argument('--heads', type=int, default=2, help='Number of attention heads in GAT')
        parser.add_argument('--hidden_dim', type=int, default=128,
                            help='Dimensionality of hidden units in GAT')
        parser.add_argument('--l1_lambda', type=float, default=1e-05, help='l1_lambda')
        parser.add_argument('--l2_lambda', type=float, default=0.01, help='l2_lambda')
        parser.add_argument('--lr', type=float, default=1e-05, help='Learning rate')
        parser.add_argument('--num_layers', type=int, default=8, help='Number of GAT layers')
        parser.add_argument('--output_dim', type=int, default=16,
                            help='Dimensionality of output units in GAT')
        parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
        parser.add_argument('--epochs', type=int, default=50000, help='Number of epochs to train')
        args = parser.parse_args()
        return args

    if TRANSFER_TARGET == 'CP_Chirasil_Dex_CB':
        parser = argparse.ArgumentParser(description='Graph data mining with GAT')
        parser.add_argument('--devices', type=int, nargs='+', default=[0], help='Which GPUs to use')
        parser.add_argument('--batch_size', type=int, default=8, help='Input batch size for training')
        parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
        parser.add_argument('--heads', type=int, default=2, help='Number of attention heads in GAT')
        parser.add_argument('--hidden_dim', type=int, default=32,
                            help='Dimensionality of hidden units in GAT')
        parser.add_argument('--l1_lambda', type=float, default=0.01, help='l1_lambda')
        parser.add_argument('--l2_lambda', type=float, default=0.01, help='l2_lambda')
        parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
        parser.add_argument('--num_layers', type=int, default=2, help='Number of GAT layers')
        parser.add_argument('--output_dim', type=int, default=8,
                            help='Dimensionality of output units in GAT')
        parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
        parser.add_argument('--epochs', type=int, default=50000, help='Number of epochs to train')
        args = parser.parse_args()
        return args

    if TRANSFER_TARGET == 'CP_Chirasil_L_Val':
        parser = argparse.ArgumentParser(description='Graph data mining with GAT')
        parser.add_argument('--devices', type=int, nargs='+', default=[0], help='Which GPUs to use')
        parser.add_argument('--batch_size', type=int, default=8, help='Input batch size for training')
        parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
        parser.add_argument('--heads', type=int, default=2, help='Number of attention heads in GAT')
        parser.add_argument('--hidden_dim', type=int, default=32,
                            help='Dimensionality of hidden units in GAT')
        parser.add_argument('--l1_lambda', type=float, default=0.01, help='l1_lambda')
        parser.add_argument('--l2_lambda', type=float, default=1e-05, help='l2_lambda')
        parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
        parser.add_argument('--num_layers', type=int, default=2, help='Number of GAT layers')
        parser.add_argument('--output_dim', type=int, default=8,
                            help='Dimensionality of output units in GAT')
        parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
        parser.add_argument('--epochs', type=int, default=50000, help='Number of epochs to train')
        args = parser.parse_args()
        return args
