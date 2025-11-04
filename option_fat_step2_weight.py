import argparse
import template

parser = argparse.ArgumentParser(description='EDSR and MDSR')

# VDSR
parser.add_argument("--is_y", action='store_true', default=True,
                    help='evaluate on y channel, if False evaluate on RGB channels')
parser.add_argument("--upscale_factor", type=int, default=4,
                    help='upscaling factor')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='use cuda')


parser.add_argument('--debug', action='store_true',
                    help='Enables debug mode')
parser.add_argument('--template', default='.',
                    help='You can set various templates in option.py')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=1,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

# Data specifications
parser.add_argument('--dir_data', type=str, default=r'D:\yxd\code\ESRT-RS\npydata\AID\train',
                    help='dataset directory')
# parser.add_argument('--dir_demo', type=str, default=r'D:\yxd\code\ESRT-RS\npydata\UCM',
#                     help='demo image directory')
parser.add_argument('--dir_demo', type=str, default=r'D:\yxd\code\ESRT-RS\npydata\AID\val',
                    help='demo image directory')
parser.add_argument('--data_train', type=str, default='AID',
                    help='train dataset name')
parser.add_argument('--data_test', type=str, default='AID',
                    help='test dataset name')
parser.add_argument('--benchmark_noise', action='store_true',
                    help='use noisy benchmark sets')
parser.add_argument('--n_train', type=int, default=7884,
                    help='number of training set')
# parser.add_argument('--n_val', type=int, default=2056,
#                     help='number of validation set')
# parser.add_argument('--n_val', type=int, default=150,
#                     help='number of validation set')
parser.add_argument('--offset_val', type=int, default=0,
                    help='validation index offest')
parser.add_argument('--ext', type=str, default='img',
# parser.add_argument('--ext', type=str, default='sep_reset',
# parser.add_argument('--ext', type=str, default='sep',
                    help='dataset file extension')
parser.add_argument('--scale', default='4',
                    help='super resolution scale')
parser.add_argument('--patch_size', type=int, default=200,
                    help='output patch size')

parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--noise', type=str, default='.',
                    help='Gaussian noise std.')
parser.add_argument('--chop', action='store_true',
                    help='enable memory-efficient forward')

# Model specifications
parser.add_argument('--model', default='fat_step2_weight',
                    help='model name')
parser.add_argument('--act', type=str, default='relu',
                    help='activation function')
parser.add_argument('--pre_train', type=str, default=r'D:\yxd\code\nips_uncertainy\experiment\X4\FAT_fat_s_x4_step2_weight\model\model_latest1103.pt',
                    help='pre-trained model directory')
# parser.add_argument('--pre_train', type=str, default=r'.',
#                     help='pre-trained model directory')
parser.add_argument('--pre_train_step1', type=str, default=r'D:\yxd\code\nips_uncertainy\experiment\X2\FAT_fat_s_x2_step1_weight\model\model_best.pt',
                    help='pre-trained model directory of step1')
parser.add_argument('--extend', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--n_resblocks', type=int, default=8,
                    help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=0.1,
                    help='residual scaling')
parser.add_argument('--shift_mean', default=True,
                    help='subtract pixel mean from the input')
parser.add_argument('--precision', type=str, default='single',
                    choices=('single', 'half'),
                    help='FP precision for test (single | half)')

# Training specifications
parser.add_argument('--reset', action='store_true',
                    help='reset the training')
parser.add_argument('--test_every', type=int, default=1000,
                    help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=50,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=8,
                    help='input batch size for training')
parser.add_argument('--split_batch', type=int, default=1,
                    help='split the batch into smaller chunks')
parser.add_argument('--self_ensemble', action='store_true',
                    help='use self-ensemble method for test')
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')
parser.add_argument('--gan_k', type=int, default=1,
                    help='k value for adversarial loss')

# Optimization specifications
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--lr_decay', type=int, default=5,
                    help='learning rate decay per N epochs')
parser.add_argument('--decay_type', type=str, default='step',
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='ADAM beta2')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')

# Loss specifications
parser.add_argument('--loss', type=str, default='1*L1',
                    help='loss function configuration')
parser.add_argument('--stage', type=str, default='step2',
                    help='loss function stage')
parser.add_argument('--skip_threshold', type=float, default='1e6',
                    help='skipping batch that has large error')

# Log specifications
parser.add_argument('--save', type=str, default='../experiment/X4/FAT_fat_s_x4_step2_weight',
                    help='file name to save')
parser.add_argument('--load', type=str, default='.',
                    help='file name to load')
parser.add_argument('--resume', type=int, default=0,
                    help='resume from specific checkpoint')
parser.add_argument('--print_model', action='store_true',
                    help='print model')
parser.add_argument('--save_models', action='store_true',
                    help='save all intermediate models')
parser.add_argument('--print_every', type=int, default=400,
                    help='how many batches to wait before logging training status')
parser.add_argument('--repeat', type=int, default=10,
                    help='how many time to repeat dataset')
parser.add_argument('--save_results', type=bool, default=True,
                    help='save output results')

# options for residual group and feature channel reduction
parser.add_argument('--n_resgroups', type=int, default=8,
                    help='number of residual groups')
parser.add_argument('--reduction', type=int, default=16,
                    help='number of feature maps reduction')
# options for test
# parser.add_argument('--testpath', type=str, default=r'D:\yxd\code\ESRT-RS\npydata\UCM',
#                     help='dataset directory for testing')
# parser.add_argument('--testset', type=str, default='UCM',
#                     help='dataset name for testing')
parser.add_argument('--testpath', type=str, default=r'D:\yxd\code\ESRT-RS\npydata\UCM',
                    help='dataset directory for testing')
parser.add_argument('--testset', type=str, default='UCM',
                    help='dataset name for testing')
parser.add_argument('--n_val', type=int, default=2056,
                    help='number of validation set')


# parser.add_argument('--testpath', type=str, default=r'D:\yxd\code\ESRT-RS\npydata\AID\test',
#                     help='dataset directory for testing')
# parser.add_argument('--testset', type=str, default='AID_test',
#                     help='dataset name for testing')
# parser.add_argument('--n_val', type=int, default=1966,
#                     help='number of validation set')

args = parser.parse_args()
template.set_template(args)

args.scale = list(map(lambda x: int(x), args.scale.split('+')))

if args.epochs == 0:
    args.epochs = 1e8

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False
