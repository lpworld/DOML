import torch
from torch.autograd import Variable
from mlp import MLP, ShareLayer
from utils import use_optimizer, use_cuda
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import numpy as np
import math

class Engine(object):
    """Meta Engine for training & evaluating NCF model

    Note: Subclass should implement self.model !
    """

    def __init__(self, config):
        self.config = config  # model configuration
        self.share_layer_A = torch.nn.Linear(config['latent_dim'], config['latent_dim'])
        self.share_layer_B = torch.nn.Linear(config['latent_dim'], config['latent_dim'])
        self.metric_layer_A = torch.nn.Linear(config['latent_dim'], config['latent_dim'])
        self.metric_layer_B = torch.nn.Linear(config['latent_dim'], config['latent_dim'])
        self.modelA = MLP(config)
        self.modelB = MLP(config)
        self.sharelayer = ShareLayer(config)
        if config['use_cuda'] is True:
            self.modelA.cuda()
            self.modelB.cuda()
            self.sharelayer.cuda()
        self.optA = use_optimizer(self.modelA, config)
        self.optB = use_optimizer(self.modelB, config)
        self.optshare = torch.optim.SGD(self.sharelayer.parameters(), lr=1e-1)
        self.optmetric_A = torch.optim.SGD(self.metric_layer_A.parameters(), lr=1e-1)
        self.optmetric_B = torch.optim.SGD(self.metric_layer_B.parameters(), lr=1e-1)
        self.crit = torch.nn.MSELoss()

    def train_single_batch(self, book_user_embeddings, book_item_embeddings, book_rating,
                          movie_user_embeddings, movie_item_embeddings, movie_rating, user_overlap, item_overlap):
        self.optA.zero_grad()
        self.optB.zero_grad()
        book_user_embeddings = self.share_layer_A(book_user_embeddings)
        movie_user_embeddings = self.share_layer_B(movie_user_embeddings)
        book_item_embeddings = self.metric_layer_A(book_item_embeddings)
        movie_item_embeddings = self.metric_layer_B(movie_item_embeddings)
        book_ratings_pred = self.modelA(book_user_embeddings, book_item_embeddings)
        lossA = self.crit(book_ratings_pred.squeeze(1), book_rating)
        movie_ratings_pred = self.modelB(movie_user_embeddings, movie_item_embeddings)
        lossB = self.crit(movie_ratings_pred.squeeze(1), movie_rating)
        '''
        if user_overlap:
            self.optshare.zero_grad()
            book_overlap_embeddings, movie_overlap_embeddings = self.sharelayer(book_user_embeddings, movie_user_embeddings)
            book_ratings_pred = self.modelB(movie_overlap_embeddings, book_item_embeddings)
            movie_ratings_pred = self.modelB(book_overlap_embeddings, movie_item_embeddings)
            lossA_overlap = self.crit(book_ratings_pred.squeeze(1), book_rating)
            lossB_overlap = self.crit(movie_ratings_pred.squeeze(1), movie_rating)
            lossA_overlap.backward(retain_graph=True)
            lossB_overlap.backward(retain_graph=True)
            self.optshare.step()
        '''
        lossA.backward()
        lossB.backward()
        self.optA.step()
        self.optB.step()
        if self.config['use_cuda'] is True:
            lossA = lossA.data.cpu().numpy()
            lossB = lossB.data.cpu().numpy()
        else:
            lossA = lossA.data.numpy()
            lossB = lossB.data.numpy()
        return lossA + lossB
    
    def train_user_overlap(self, book_user_embeddings, movie_user_embeddings):
        self.optshare.zero_grad()
        book_user_embeddings = self.share_layer_A(book_user_embeddings)
        movie_user_embeddings = self.share_layer_B(movie_user_embeddings)
        book_overlap_embeddings, movie_overlap_embeddings = self.sharelayer(book_user_embeddings, movie_user_embeddings)
        map_loss_A = self.crit(book_overlap_embeddings, movie_user_embeddings)
        map_loss_B = self.crit(movie_overlap_embeddings, book_user_embeddings)
        map_loss_A.backward(retain_graph=True)
        map_loss_B.backward(retain_graph=True)
        reg = 1e-6
        with torch.enable_grad():
            orth_loss_A, orth_loss_B = torch.zeros(1), torch.zeros(1)
            for name, param in self.sharelayer.bridge1.named_parameters():
                if 'bias' not in name:
                    param_flat = param.view(param.shape[0], -1)
                    sym = torch.mm(param_flat, torch.t(param_flat))
                    sym -= torch.eye(param_flat.shape[0])
                    orth_loss_A = orth_loss_A + (reg * sym.abs().sum())
            orth_loss_A.backward()
            for name, param in self.sharelayer.bridge2.named_parameters():
                if 'bias' not in name:
                    param_flat = param.view(param.shape[0], -1)
                    sym = torch.mm(param_flat, torch.t(param_flat))
                    sym -= torch.eye(param_flat.shape[0])
                    orth_loss_B = orth_loss_B + (reg * sym.abs().sum())
            orth_loss_B.backward()
        self.optshare.step()
        if self.config['use_cuda'] is True:
            map_loss_A = map_loss_A.data.cpu().numpy()
            map_loss_B = map_loss_B.data.cpu().numpy()
            orth_loss_A = orth_loss_A.data.cpu().numpy()
            orth_loss_B = orth_loss_B.data.cpu().numpy()
        else:
            map_loss_A = map_loss_A.data.numpy()
            map_loss_B = map_loss_B.data.numpy()
            orth_loss_A = orth_loss_A.data.numpy()
            orth_loss_B = orth_loss_B.data.numpy()
        return map_loss_A + map_loss_B + orth_loss_A + orth_loss_B

    def train_item_overlap(self, book_item_embeddings, movie_item_embeddings):
        self.optmetric_A.zero_grad()
        self.optmetric_B.zero_grad()
        book_item_embeddings = self.metric_layer_A(book_item_embeddings)
        movie_item_embeddings = self.metric_layer_B(movie_item_embeddings)
        map_loss = self.crit(book_item_embeddings, movie_item_embeddings)
        map_loss.backward(retain_graph=True)
        self.optmetric_A.step()
        self.optmetric_B.step()
        if self.config['use_cuda'] is True:
            map_loss = map_loss.data.cpu().numpy()
        else:
            map_loss = map_loss.data.numpy()
        return map_loss

    def train_an_epoch(self, train_book_loader, train_movie_loader, epoch_id, user_overlap=False, item_overlap=False):
        self.modelA.train()
        self.modelB.train()
        self.sharelayer.train()
        total_loss = 0
        for book_batch, movie_batch in zip(train_book_loader, train_movie_loader):
            assert isinstance(book_batch[0], torch.LongTensor)
            book_rating, book_user_embeddings, book_item_embeddings = Variable(book_batch[2]), Variable(book_batch[3]), Variable(book_batch[4])
            movie_rating, movie_user_embeddings, movie_item_embeddings = Variable(movie_batch[2]), Variable(movie_batch[3]), Variable(movie_batch[4])
            book_rating = book_rating.float()
            movie_rating = movie_rating.float()
            if self.config['use_cuda'] is True:
                book_rating = book_rating.cuda()
                movie_rating = movie_rating.cuda()
                book_user_embeddings = book_user_embeddings.cuda()
                book_item_embeddings = book_item_embeddings.cuda()
                movie_user_embeddings = movie_user_embeddings.cuda()
                movie_item_embeddings = movie_item_embeddings.cuda()
            loss = self.train_single_batch(book_user_embeddings, book_item_embeddings, book_rating,
                                           movie_user_embeddings, movie_item_embeddings, movie_rating, user_overlap=user_overlap, item_overlap=item_overlap)
            total_loss += loss

    def evaluate(self, evaluate_book_data, evaluate_movie_data, epoch_id):
        self.modelA.eval()
        self.modelB.eval()
        book_user, book_item, book_user_embeddings, book_item_embeddings, \
            book_golden = evaluate_book_data[0], evaluate_book_data[1], \
                Variable(evaluate_book_data[2]), Variable(evaluate_book_data[3]), evaluate_book_data[4]
        movie_user, movie_item, movie_user_embeddings, movie_item_embeddings, \
            movie_golden = evaluate_movie_data[0], evaluate_movie_data[1], \
                Variable(evaluate_movie_data[2]), Variable(evaluate_movie_data[3]), evaluate_movie_data[4]
        if self.config['use_cuda'] is True:
            book_user_embeddings = book_user_embeddings.cuda()   
            book_item_embeddings = book_item_embeddings.cuda()
            movie_user_embeddings = movie_user_embeddings.cuda()   
            movie_item_embeddings = movie_item_embeddings.cuda()
        book_scores = self.modelA(book_user_embeddings, book_item_embeddings)
        book_scores = book_scores.detach().numpy()
        movie_scores = self.modelB(movie_user_embeddings, movie_item_embeddings)
        movie_scores = movie_scores.detach().numpy()
        book_RMSE = math.sqrt(mean_squared_error(book_golden, book_scores))
        book_MAE = mean_absolute_error(book_golden, book_scores)
        movie_RMSE = math.sqrt(mean_squared_error(movie_golden, movie_scores))
        movie_MAE = mean_absolute_error(movie_golden, movie_scores)
        
        unique_book_user = list(set(book_user))
        unique_movie_user = list(set(movie_user))
        book_recommend, movie_recommend = [], []
        book_precision, movie_precision, book_recall, movie_recall = [], [], [], []
        for index in range(len(book_user)):
            book_recommend.append((book_user[index],book_item[index],book_golden[index],book_scores[index]))
        for index in range(len(movie_user)):
            movie_recommend.append((movie_user[index],movie_item[index],movie_golden[index],movie_scores[index]))      
        for user in unique_book_user:
            user_ratings = [x for x in book_recommend if x[0]==user]
            user_ratings.sort(key=lambda x:x[3], reverse=True)
            user_ratings = user_ratings[:5]
            n_rel = sum((true_r >= 0.5) for (_, _, true_r, _) in user_ratings)
            n_rec_k = sum((est >= 0.5) for (_, _, _, est) in user_ratings)
            n_rel_and_rec_k = sum(((true_r >= 0.5) and (est >= 0.5))
                            for (_, _, true_r, est) in user_ratings)
            book_precision.append(n_rel_and_rec_k / n_rec_k if n_rec_k!=0 else 1)
            book_recall.append(n_rel_and_rec_k / n_rel if n_rel!=0 else 1)
        book_precision = np.mean(book_precision)
        book_recall = np.mean(book_recall)
        for user in unique_movie_user:
            user_ratings = [x for x in movie_recommend if x[0]==user]
            user_ratings.sort(key=lambda x:x[3], reverse=True)
            user_ratings = user_ratings[:5]
            n_rel = sum((true_r >= 0.5) for (_, _, true_r, _) in user_ratings)
            n_rec_k = sum((est >= 0.5) for (_, _, _, est) in user_ratings)
            n_rel_and_rec_k = sum(((true_r >= 0.5) and (est >= 0.5))
                              for (_, _, true_r, est) in user_ratings)
            movie_precision.append(n_rel_and_rec_k / n_rec_k if n_rec_k!=0 else 0)
            movie_recall.append(n_rel_and_rec_k / n_rel if n_rel!=0 else 1)
        movie_precision = np.mean(movie_precision)
        movie_recall = np.mean(movie_recall)
        
        print('[Domain A Evluating Epoch {}] RMSE = {:.4f}, MAE = {:.4f}, Precision = {}, Recall = {}'.format(epoch_id, book_RMSE, book_MAE, book_precision, book_recall))
        print('[Domain B Evluating Epoch {}] RMSE = {:.4f}, MAE = {:.4f}, Precision = {}, Recall = {}'.format(epoch_id, movie_RMSE, movie_MAE, movie_precision, movie_recall))
        
        return book_RMSE, book_MAE, movie_RMSE, movie_MAE

    def save(self, dirname, filename):
        with open(os.path.join(dirname, filename)+'A', 'wb') as f:
            torch.save(self.modelA.state_dict(), f)
        with open(os.path.join(dirname, filename)+'B', 'wb') as f:
            torch.save(self.modelB.state_dict(), f)