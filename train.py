import argparse
import pandas as pd
from engine import Engine
from data import SampleGenerator, OverlapGenerator, MetricGenerator
import json

parser = argparse.ArgumentParser('Unsupervised Recommendation Training')
# Path Arguments
parser.add_argument('--num_epoch', type=int, default=100,help='number of epoches')
parser.add_argument('--batch_size', type=int, default=1024,help='batch size')
parser.add_argument('--lr', type=int, default=1e-3,help='learning rate')
parser.add_argument('--latent_dim', type=int, default=8,help='latent dimensions')
parser.add_argument('--cuda', action='store_true',help='use of cuda')
args = parser.parse_args()

config = {'batch_size': args.batch_size,
              'optimizer': 'sgd',
              'lr': args.lr,
              'latent_dim': args.latent_dim,
              'nlayers':1,
              'layers': [2*args.latent_dim,64,args.latent_dim],  # layers[0] is the concat of latent user vector & latent item vector
              'use_cuda': args.cuda}

#Load Datasets
book = pd.read_csv('book.csv')
movie = pd.read_csv('movie.csv')
music = pd.read_csv('music.csv')
book['user_embedding'] = book['user_embedding'].map(eval)
book['item_embedding'] = book['item_embedding'].map(eval)
movie['user_embedding'] = movie['user_embedding'].map(eval)
movie['item_embedding'] = movie['item_embedding'].map(eval)
music['user_embedding'] = music['user_embedding'].map(eval)
music['item_embedding'] = music['item_embedding'].map(eval)

book_user = list(set(book['userId']))     #1005 users
movie_user = list(set(movie['userId']))   #2007 users
music_user = list(set(music['userId']))   #160 users

book_movie_overlap = list(set(book['userId']).intersection(movie['userId']))     # 195 users
movie_music_overlap = list(set(movie['userId']).intersection(music['userId']))   # 40 users
book_music_overlap = list(set(music['userId']).intersection(book['userId']))     # 23 users
    
sample_book_generator = SampleGenerator(ratings=book)
evaluate_book_data = sample_book_generator.evaluate_data
sample_movie_generator = SampleGenerator(ratings=movie)
evaluate_movie_data = sample_movie_generator.evaluate_data
sample_music_generator = SampleGenerator(ratings=music)
evaluate_music_data = sample_music_generator.evaluate_data

engine = Engine(config)
train_book_loader = sample_book_generator.instance_a_train_loader(config['batch_size'])
train_music_loader = sample_music_generator.instance_a_train_loader(config['batch_size'])
train_movie_loader = sample_movie_generator.instance_a_train_loader(config['batch_size'])

with open('overlap_movie_music_index','r') as f:
    overlap = json.load(f)
movie_overlap = overlap['movie']
music_overlap = overlap['music']

#book_music_overlap = list(np.random.choice(book_music_overlap,64,replace=True))
overlap_generator = OverlapGenerator(rating1=movie, rating2=music, users=movie_music_overlap)
overlap_movie_loader, overlap_music_loader, movie_user_embeddings, music_user_embeddings = overlap_generator.instance_a_train_loader(config['batch_size'])

metric_generator = MetricGenerator(rating1=movie, rating2=music, metric1=movie_overlap, metric2=music_overlap)
metric_movie_loader, metric_music_loader, movie_item_embeddings, music_item_embeddings = overlap_generator.instance_a_train_loader(config['batch_size'])

for epoch in range(args.num_epoch):
    print('Epoch {} starts !'.format(epoch))
    print('-' * 80)
    engine.train_an_epoch(train_movie_loader, train_music_loader, epoch_id=epoch)
    loss = engine.train_user_overlap(movie_user_embeddings, music_user_embeddings)
    engine.train_an_epoch(overlap_movie_loader, overlap_music_loader, epoch_id=epoch, user_overlap=True)
    #loss = engine.train_item_overlap(movie_item_embeddings, music_item_embeddings)
    #engine.train_an_epoch(overlap_movie_loader, overlap_music_loader, epoch_id=epoch, item_overlap=True)
    movie_RMSE, movie_MAE, music_RMSE, music_MAE = engine.evaluate(evaluate_movie_data, evaluate_music_data, epoch_id=epoch)

