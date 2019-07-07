import os
import requests
import json
import numpy as np
import pandas as pd

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

import MeCab

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset, DataLoader

dict_path = '-d /home/b2018yniki/local/lib/mecab/dic/mecab-ipadic-neologd/'

def get_tango(sen):
    word_list = []
    tagger = MeCab.Tagger(dict_path)
    for word_line in tagger.parse(sen).split("\n"):
        if word_line.strip() == "EOS":
            break
        (word, temp) = word_line.split("\t")
        temps = temp.split(',')
        if "記号" == temps[0]:
            continue
        if "数" == temps[1]:
            continue
        word_list.append(word)
    return word_list

def df2input(df, vocab_idx):
    data = []
    for text in df.values:
        words = get_tango(text)
        data.append([vocab_idx[word] for word in words if word in vocab_idx.keys()])
    return data

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data, tags):
        super(MyDataset, self).__init__()
        assert len(data) == len(tags)
        # npに変換し、0埋めを行う
        max_length = max([len(d) for d in data])
        self.data = np.zeros((len(tags), max_length))
        for i, d1 in enumerate(data):
            for l, d2 in enumerate(d1):
                self.data[i][l] = d2
        self.tags = tags

    def __len__(self):
        return len(self.tags)

    def __getitem__(self, index):
        return self.data[index], self.tags[index]

class LSTM(nn.Module):
    def __init__(self, batch_size, vocab_size, emb_dim, hidden_dim, dropout_rate=0.0, activate='tanh', bidirectional=False, device='cpu'):
        super(LSTM, self).__init__()
        
        self.vocab_size = vocab_size
        self.emb_dim    = emb_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.activate   = activate
        
        self.emb  = nn.Embedding(self.vocab_size, self.emb_dim)
        self.lstm = nn.LSTM(self.emb_dim, self.hidden_dim, batch_first=True, bidirectional=self.bidirectional)
        
        self.fc0 = nn.Linear(hidden_dim * 2, 100)
        self.fc1 = nn.Linear(100, 2)
        self.do  = nn.Dropout(dropout_rate)
        self.device = device
        self.hidden = self.init_hidden()

    def forward(self, x):

        x = self.emb(x)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        y = self.fc0(torch.cat([self.hidden[0][-1], self.hidden[0][-2]], 1))
        y = self.do(y)
        if self.activate == 'tanh':
            y = self.fc1(F.tanh(y))
        elif self.activate == 'relu':
            y = self.fc1(F.relu(y))
        tag_scores = F.log_softmax(y)
        return tag_scores

    def init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        num = 2 if self.bidirectional else 1    # bidirectionalのとき2
        h0 = torch.zeros(num, self.batch_size, self.hidden_dim).to(self.device)
        c0 = torch.zeros(num, self.batch_size, self.hidden_dim).to(self.device)
        return (h0, c0)
    
def training(net, train_loader, epoch_num, optimizer, criterion, device):

    for epoch in range(epoch_num):

        train_loss = 0.0
        train_acc  = 0.0
        valid_loss = 0.0
        valid_acc  = 0.0

        # train====================
        net.train()
        for xx, yy in train_loader:
            xx, yy = xx.long().to(device), yy.to(device)

            net.batch_size = len(yy)
            net.hidden = net.init_hidden()

            optimizer.zero_grad()    # 勾配の初期化

            output = net(xx)
            loss   = criterion(output, yy)

            train_loss += loss.item()
            train_acc += (output.max(1)[1] == yy).sum().item()

            loss.backward(retain_graph=True)     # 逆伝播の計算
            optimizer.step()    # 勾配の更新

def test(net, test_loader, y_test, device):
    net.eval()
    y_pred = []
    with torch.no_grad():
        for xx, yy in test_loader:
            xx, yy = xx.long().to(device), yy.to(device)

            net.batch_size = len(yy)
            net.hidden = net.init_hidden()

            output = net(xx)
            y_pred += output.data.max(1, keepdim=True)[1].to('cpu').numpy()[:,0].tolist()

    acc = (y_pred == y_test).sum().item() / len(y_test)
    result = precision_recall_fscore_support(y_test, y_pred, average='macro')
    return [acc, result[0], result[1], result[2]]

def slack_notification(arg_str):
    requests.post('https://hooks.slack.com/services/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', data = json.dumps({
        'text': arg_str, # 投稿するテキスト
        'username': u'yniki', # 投稿のユーザー名
    }))

def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    train_df = pd.read_csv('/home/b2018yniki/data/nikkei/train.txt', sep='\t', header=None, names=['target', 'time', 'body']).drop('time', axis=1)
    test_df  = pd.read_csv('/home/b2018yniki/data/nikkei/test.txt', sep='\t', header=None, names=['target', 'time', 'body']).drop('time', axis=1)
    total_df = pd.concat([train_df, test_df], ignore_index=True).drop_duplicates()

    X = total_df.body
    y = total_df.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2019)
    del train_df, test_df, total_df, X, y
    
    vocab = []
    for text in X_train.values:
        vocab.extend(get_tango(text))
    vocab = list(set(vocab))
    print('vocabulaly size: {}'.format(len(vocab)))
    vocab_idx = dict(zip(vocab, range(len(vocab))))
    del vocab
    
    X_train = df2input(train_df, vocab_idx)
    X_test  = df2input(test_df, vocab_idx)

    train_ds = MyDataset(X_train, y_train.values)
    test_ds  = MyDataset(X_test, y_test.values)
    del X_train, X_test
    
    vocab_size = len(vocab_idx)      # 語彙数
    emb_list   = [100, 200, 300]     # 分散表現の次元数

    epoch_list   = [100, 200, 300]   # エポック数
    batch_list   = [64, 128, 256]    # バッチサイズ
    hidden_list  = [100, 200]        # 隠れ層の次元数
    dropout_list = [0.0, 0.5]  # Dropout率
    activate_list = ['tanh', 'relu'] # 活性化関数
    lr_list = [0.1, 0.01]     # 学習率
    l2_list = [0, 1e-3]              # l2正則化

    i = 0
    for batch_size in tqdm(batch_list):
        np.random.seed(2019)
        np.random.RandomState(2019)
        torch.manual_seed(2019)

        # dataloader(注意：validloader, testloaderのshuffleはFalse)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
        test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

        for emb_dim in emb_list:
            for hidden_dim in hidden_list:
                for dropout_rate in dropout_list:
                    for activate in activate_list:
                        for lr in lr_list:
                            for l2 in l2_list:
                                for epoch in epoch_list:

                                    df = pd.read_csv(os.path.join(os.getcwd(), 'results/gridsearch_lstm_end2end.csv'))
                                    if len(df.query('epoch == @epoch & batch_size == @batch_size & embedding_dim == @emb_dim & hidden_dim == @hidden_dim & activate_func == @activate & learning_rate == @lr & l2_regular == @l2 & dropout_rate == @dropout_rate')) == 1:
                                        continue
                                    del df
                                    
                                    np.random.seed(2019)
                                    np.random.RandomState(2019)
                                    torch.manual_seed(2019)

                                    net = LSTM(batch_size, vocab_size, emb_dim, hidden_dim, dropout_rate, activate, bidirectional=True, device=device).to(device)
                                    criterion = nn.NLLLoss()
                                    optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=l2)

                                    # Train Network
                                    training(net, train_loader, epoch, optimizer, criterion, device)
                                    # Test Network
                                    result = test(net, test_loader, y_test, device)

                                    df = pd.read_csv(os.path.join(os.getcwd(), 'results/gridsearch_lstm_end2end.csv'))
                                    write_ser = pd.Series([
                                        epoch, batch_size, emb_dim, hidden_dim, activate, lr, l2, dropout_rate, result[0], result[1], result[2], result[3]
                                    ], index = df.columns)
                                    df.append(write_ser, ignore_index=True).to_csv(os.path.join(os.getcwd(), 'results/gridsearch_lstm_end2end.csv'), index=False)
                                    del df, net, criterion, optimizer
                                    
        del train_loader, test_loader
        i += 1
        notice = '{}/{} gridsearch is finished!!!'.format(i, len(batch_list))
        slack_notification()
                                    
if __name__ == '__main__':
    main()