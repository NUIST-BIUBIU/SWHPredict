import argparse
from experiments.transformer_exp import Forecasting
from experiments.tcn_exp import TCNForecasting
import torch

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, required=True)
parser.add_argument('--station_id', type=int, required=True)
parser.add_argument('--seq_len', type=int, required=True)
parser.add_argument('--label_len', type=int, required=False)
parser.add_argument('--pred_len', type=int, required=True)
parser.add_argument('--enc_in', type=int, required=False)
parser.add_argument('--dec_in', type=int, default=1)
parser.add_argument('--c_out', type=int, default=1)
parser.add_argument('--d_model', type=int, default=32)
parser.add_argument('--n_heads', type=int, default=4)
parser.add_argument('--e_layers', type=int, default=2)
parser.add_argument('--d_layers', type=int, default=1)
parser.add_argument('--d_ff', type=int, default=16)
parser.add_argument('--factor', type=int, default=1)
parser.add_argument('--features', type=str, required=False)
parser.add_argument('--years', type=str, required=False)
parser.add_argument('--batch_size', type=int, required=False)
parser.add_argument('--learning_rate', type=float, required=False)
parser.add_argument('--activation', type=str, required=True, default='gelu')
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--train_epochs', type=int, required=False)
parser.add_argument('--data_rootpath', type=str, required=True)

args = parser.parse_args()
print(args)

if args.model in ['Transformer']:
    exp = Forecasting(args)
else:
    exp = TCNForecasting(args)
print('>>>>>>>start training')
exp.train()

torch.cuda.empty_cache()





