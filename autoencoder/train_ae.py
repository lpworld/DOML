import argparse, json, os, random, logging, sys
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from models import Seq2Seq

parser = argparse.ArgumentParser('Train an autoencoder.')
# Path Arguments
parser.add_argument('--data_file', type=str, default='data.json',
                    help='location of the data corpus')
parser.add_argument('--outf', type=str, default='output',
                    help='output directory name')
# Training Arguments
parser.add_argument('--batch_size', type=int, default=1024, metavar='N',
                    help='batch size')
parser.add_argument('--epochs', type=int, default=200,
                    help='maximum number of epochs')
parser.add_argument('--split', type=float, default=0.1,
                    help='the ratio of validation data.'
                         'set it to 0 to switch off validating')
# Model Arguments
parser.add_argument('--seqlen', type=int, default=9,
                    help='length of input feature sequence')
parser.add_argument('--emsize', type=int, default=8,
                    help='size of word embeddings')
parser.add_argument('--emsize_len', type=int, default=5)
parser.add_argument('--nhidden', type=int, default=8,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=1,
                    help='autoencoder learning rate')
parser.add_argument('--noise_radius', type=float, default=0.2,
                    help='stdev of noise for autoencoder (regularizer)')
parser.add_argument('--noise_anneal', type=float, default=0.995,
                    help='anneal noise_radius exponentially by this'
                         'every 100 iterations')
parser.add_argument('--hidden_init', action='store_true',
                    help="initialize decoder hidden state with encoder's")
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--clip', type=float, default=1,
                    help='gradient clipping, max norm')
parser.add_argument('--temp', type=float, default=1,
                    help='softmax temperature (lower --> more discrete)')

# other
parser.add_argument('--seed', type=int, default=625,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log_interval', type=int, default=10)

args = parser.parse_args()
print(vars(args))

# create output directory
out_dir = './{}'.format(args.outf)
os.makedirs(out_dir, exist_ok=True)

# set random seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, "
              "so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
fh = logging.FileHandler('logs.txt')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
log.addHandler(fh)
log.addHandler(ch)

class AutoEncoder:
    def __init__(self):
        self.autoencoder = Seq2Seq(emsize=args.emsize,
                                   nhidden=args.nhidden,
                                   ntokens=args.ntokens,
                                   seqlen=args.seqlen,
                                   nlayers=args.nlayers,
                                   noise_radius=args.noise_radius,
                                   hidden_init=args.hidden_init,
                                   dropout=args.dropout,
                                   gpu=args.cuda)
        self.optimizer = optim.SGD(self.autoencoder.parameters(), lr=args.lr)
        #self.optimizer = optim.Adam(self.autoencoder.parameters())
        self.criterion = nn.CrossEntropyLoss()
        if args.cuda:
            self.autoencoder = self.autoencoder.cuda()
            self.criterion = self.criterion.cuda()

    def update(self, batch):
        self.autoencoder.train()
        self.autoencoder.zero_grad()
        source, target = batch
        source = to_gpu(args.cuda, Variable(source))
        target = to_gpu(args.cuda, Variable(target))
        output = self.autoencoder(source)
        loss = self.criterion(output, target.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), args.clip)
        self.optimizer.step()
        return loss.data.numpy()

    def anneal(self):
        '''exponentially decaying noise on autoencoder'''
        self.autoencoder.noise_radius = self.autoencoder.noise_radius * args.noise_anneal

    def save(self, dirname, filename):
        with open(os.path.join(dirname, filename), 'wb') as f:
            torch.save(self.autoencoder.state_dict(), f)

def batchify(data, bsz, shuffle=True, gpu=False):
    if shuffle:
        random.shuffle(data)
    nbatch = len(data) // bsz
    batches = []
    for i in range(nbatch):
        batch = data[i*bsz:(i+1)*bsz]
        source = [x for x in batch]
        target = [x for x in batch]
        source = torch.LongTensor(source)
        target = torch.LongTensor(target)
        if gpu:
            source = source.cuda()
            target = target.cuda()
        batches.append((source, target))
    return batches

def to_gpu(gpu, var):
    if gpu:
        return var.cuda()
    return var

# prepare corpus
with open('book_user_index') as f:
    train = json.load(f)
# save arguments
ntokens = len(set([x for y in train for x in y]))+1
args.ntokens = ntokens
print(ntokens)
autoencoder = AutoEncoder()
for epoch in range(1, args.epochs + 1):
    # shuffle train data in each epoch
    batches = batchify(train, args.batch_size, shuffle=True)
    global_iters = 0
    start_time = datetime.now()
    for i, batch in enumerate(batches):
        loss = autoencoder.update(batch)
        if i % args.log_interval == 0 and i > 0:
            log.info(('[Epoch {} {}/{} Loss {:.5f} ETA {}]').format(
                epoch, i, len(batches), loss,
                str((datetime.now() - start_time) / (i + 1) * (len(batches) - i - 1)).split('.')[0]))
        global_iters += 1
        if global_iters % 100 == 0:
            autoencoder.anneal()
autoencoder.save(out_dir, 'book_user_{}.pt'.format(epoch))