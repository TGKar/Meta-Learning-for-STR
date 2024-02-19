from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

parser = argparse.ArgumentParser(description="Softmax loss classification")

# Data
parser.add_argument('--train_data_dir', nargs='+', type=str, metavar='PATH',
                    default=[None])

parser.add_argument('--train_data_gt', nargs='+', type=str, metavar='PATH',
                    default=[None])
parser.add_argument('--test_data_dir', type=str, metavar='PATH',
                    default=None)
parser.add_argument('--test_data_gt', type=str, metavar='PATH',
                    default=None)
parser.add_argument('-b', '--train_batch_size', type=int, default=32)
parser.add_argument('-v', '--val_batch_size', type=int, default=32)
parser.add_argument('-j', '--workers', type=int, default=2)
parser.add_argument('-g', '--gpus', type=str, default='1')
parser.add_argument('--height', type=int, default=48, help="input height")
parser.add_argument('--width', type=int, default=160, help="input width")
parser.add_argument('--aug', type=bool, default=True, help="using data augmentation or not")
parser.add_argument('--keep_ratio', action='store_true', default=True,
                    help='length fixed or lenghth variable.')
parser.add_argument('--voc_type', type=str, default='Italian',
                    choices=['LOWERCASE', 'ALLCASES', 'ALLCASES_SYMBOLS', 'Italian', 'Arabic', 'Bangla', 'English', 'French', 'German', 'Hindi', 'Symbols'])
parser.add_argument('--max_voc_len', type=int, default=137,
                    help="the dim of hidden layer in encoder.")
parser.add_argument('--num_train', type=int, default=-1)
parser.add_argument('--num_test', type=int, default=-1)

# Model
parser.add_argument('--max_len', type=int, default=30)
parser.add_argument('--encoder_sdim', type=int, default=512,
                    help="the dim of hidden layer in encoder.")
parser.add_argument('--encoder_layers', type=int, default=2,
                    help="the num of layers in encoder lstm.")
parser.add_argument('--decoder_sdim', type=int, default=512,
                    help="the dim of hidden layer in decoder.")
parser.add_argument('--decoder_layers', type=int, default=2,
                    help="the num of layers in decoder lstm.")
parser.add_argument('--decoder_edim', type=int, default=512,
                    help="the dim of embedding layer in decoder.")

# Optimizer
parser.add_argument('--lr', type=float, default=0.0008,
                    help="learning rate of new parameters, for pretrained ")
parser.add_argument('--inner_lr', type=float, default=0.003,
                    help="learning rate inner loop ")                    
parser.add_argument('--reptile_lr', type=float, default=0.001,
                    help="learning rate outer loop ")                    
parser.add_argument('--weight_decay', type=float, default=0.9)
parser.add_argument('--decay_iter', type=int, default=80)
parser.add_argument('--decay_end', type=float, default=0.00001)
parser.add_argument('--grad_clip', type=float, default=1.0)
parser.add_argument('--inner_iters', type=int, default=51)  # Meta learning
parser.add_argument('--outer_iters', type=int, default=10000)  # Meta learning
parser.add_argument('--iters', type=int, default=20002) #only relevnet for not meta-learning.was 30002
parser.add_argument('--decode_type', type=str, default='greed')

parser.add_argument('--resume', type=bool, default=False)
#parser.add_argument('--pretrained', type=str, default='../meta_trained_models/meta_no_italian', metavar='PATH')  # Uncomment to train from meta-initialized weights
#parser.add_argument('--pretrained', type=str, default='./trained_models/', metavar='PATH') # Uncomment to continue training from checkpoint
parser.add_argument('--pretrained', type=str, default='./sar_synall_lmdb_checkpoints_2epochs/', metavar='PATH')  # Uncomment to train from article's pretrained model or to load backbone (if resume==False)
parser.add_argument('--log_iter', type=int, default=50)
parser.add_argument('--summary_iter', type=int, default=500)
parser.add_argument('--eval_iter', type=int, default=500)
parser.add_argument('--save_iter', type=int, default=2000)  # applies to outer loop and to non-meta-training
parser.add_argument('--vis_dir', type=str, metavar='PATH')

parser.add_argument('--languages', type=list, default=['Arabic', 'Bangla', 'English', 'French', 'German', 'Hindi', 'Symbols'])  # Removed Italian

parser.add_argument('--languages_p', type=list, default=[0.1, 0.1, 0.4, 0.1, 0.1, 0.1, 0.1])


parser.add_argument('--train_checkpoints', type=str, default='./trained_models/italian_no_meta_bar')  # Training model save folder
parser.add_argument('--checkpoints', type=str, default='../meta_trained_models/meta_no_italian')  # Meta training model save folder
parser.add_argument('--meta_train_data_dir', type=str, default='../data_for_the_net')

# Edit distance arguments
parser.add_argument('--test_batches', type=int, default=50)
parser.add_argument('--test_pretrained', type=str, default='./trained_models/italian_with_meta/', metavar='PATH')
parser.add_argument('--test_voc_type', type=str, default='Italian',
                    choices=['LOWERCASE', 'ALLCASES', 'ALLCASES_SYMBOLS', 'Italian', 'Arabic', 'Bangla', 'English', 'French', 'German', 'Hindi', 'Symbols'])
parser.add_argument('--test_batch_size', type=int, default=32)

def get_args(sys_args):
  global_args = parser.parse_args(sys_args)
  return global_args