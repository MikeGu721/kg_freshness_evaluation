from pytorch_transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import gzh


def one_hot(labels, size=2):
    i = []
    for label in labels:
        i.append([0] * size)
        i[-1][label] = 1
    return i


class predictor(nn.Module):
    def __init__(self, triple=False, max_len=24, device='cuda', language='Chinese'):
        super(predictor, self).__init__()
        if language == 'Chinese':
            name = 'bert-base-chinese'
            path = gzh.bert_model
        if language == 'English':
            name = 'bert-base-uncased'
            path = gzh.eng_bert_model
        print('导入%s bert' % language)
        print('bert name: %s' % name)
        print('bert path: %s' % path)
        self.tokenizer = BertTokenizer.from_pretrained(name, cache_dir=path)
        self.bert = BertModel.from_pretrained(name, cache_dir=path)
        if triple:
            self.fn = nn.Linear(768 * 3, 2)
        else:
            self.fn = nn.Linear(768, 2)
        self.triple = triple
        self.max_len = max_len
        self.to_device(device)

    def to_device(self, device):
        self.device = device
        self.bert.to(self.device)
        self.fn.to(self.device)

    def forward(self, triples: list):
        if self.triple:
            x = []
            for triple in triples:
                triple = list(triple)
                tokens = torch.tensor(
                    [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(i))[:self.max_len] + [
                        0] * max(0, self.max_len - len(self.tokenizer.tokenize(i))) for i in triple]).to(self.device)
                x.append(self.bert(tokens)[1])
            x1, x2, x3 = x
            x = torch.cat((x1, x2, x3), dim=1)
            y = self.fn(x)
            return F.softmax(y, dim=1)
        else:
            x = \
                self.bert(
                    torch.tensor([self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(i))[:self.max_len] + [
                        0] * max(0, self.max_len - len(self.tokenizer.tokenize(i))) for i in triples]).to(self.device))[
                    1]
            y = self.fn(x)
            return F.softmax(y, dim=1)

    def get_embedding(self,triples: list):
        if self.triple:
            x = []
            for triple in triples:
                triple = list(triple)
                tokens = torch.tensor(
                    [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(i))[:self.max_len] + [
                        0] * max(0, self.max_len - len(self.tokenizer.tokenize(i))) for i in triple]).to(self.device)
                x.append(self.bert(tokens)[1])
            x1, x2, x3 = x
            print(x1.shape)
            x = torch.cat((x1, x2, x3), dim=1)
            return x
        else:
            x = \
                self.bert(
                    torch.tensor([self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(i))[:self.max_len] + [
                        0] * max(0, self.max_len - len(self.tokenizer.tokenize(i))) for i in triples]).to(self.device))[
                    1]
            return x


class DealDataset(Dataset):
    def __init__(self, X, Y=None):
        self.X = X
        self.Y = Y

    def __getitem__(self, item):
        if self.Y:
            return self.X[item], self.Y[item]
        else:
            return self.X[item]

    def __len__(self):
        return len(self.X)


if __name__ == '__main__':
    import random
    import config

    lr = 0.001
    trainRate = 0.7
    batch_size = 16
    max_len = 64
    data_size = 1000
    EPOCH = 10
    name = '0601'

    # 生成数据集
    fresh_triples, unfresh_triples = config.get_triple_data(name=name,num=data_size)
    fresh_entities, unfresh_entities = config.get_entity_data(name=name,num=data_size)

    print('fresh triples', len(fresh_triples))
    print('unfresh triples', len(unfresh_triples))
    print('fresh entities', len(fresh_entities))
    print('unfresh entities', len(unfresh_entities))

    fresh_triples, unfresh_triples = random.sample(fresh_triples, k=min(len(fresh_triples), len(unfresh_triples))), \
                                     random.sample(unfresh_triples, k=min(len(fresh_triples), len(unfresh_triples)))
    fresh_triples = [[i[0], i[1], i[2], 1] for i in fresh_triples]
    unfresh_triples = [[i[0], i[1], i[2], 0] for i in unfresh_triples]

    fresh_entities, unfresh_entities = random.sample(fresh_entities, k=min(len(fresh_entities), len(unfresh_entities))), \
                                       random.sample(unfresh_entities,
                                                     k=min(len(fresh_entities), len(unfresh_entities)))
    fresh_entities = [[i, 1] for i in fresh_entities]
    unfresh_entities = [[i, 0] for i in unfresh_entities]

    # -------------------------------------------------------------------------------------------------- #
    # -------------------------------------------------------------------------------------------------- #
    # 三元组预测
    predictor_triple = predictor(triple=True, max_len=max_len, language='English')
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(predictor_triple.parameters(), lr=lr)

    data = [i for i in fresh_triples]
    data.extend([i for i in unfresh_triples])
    data = random.sample(data, k=min(len(data), data_size))

    train = random.sample(data, k=int(len(data) * trainRate))
    trainX, trainY = [i[:3] for i in train], [i[3] for i in train]

    test = [i for i in data if i not in train]
    test = random.sample(test, k=len(test))
    testX, testY = [i[:3] for i in test], [i[3] for i in test]
    # for i in [trainX, trainY, testX, testY]:
    #     print(i[:10])
    print('训练集大小:%d' % len(train))
    print('测试集大小:%d' % len(test))
    del (train)
    del (test)
    del (data)
    del (fresh_triples)
    del (unfresh_triples)

    train_dataset = DealDataset(trainX, trainY)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    for epoch in range(EPOCH):
        epoch_loss = 0
        for inputs, labels in train_loader:
            results = predictor_triple.forward(inputs)
            labels = torch.FloatTensor(one_hot(labels)).to('cuda')
            loss = criterion(results, labels)
            optimizer.zero_grad()
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
        print(epoch_loss)
    try:
        import os

        os.mkdir('./pkl')
    except:
        pass
    torch.save(predictor_triple.state_dict(), './pkl/predictor_triples.pkl')

    test_dataset = DealDataset(testX, testY)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    with torch.no_grad():
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        mse = 0
        for inputs, labels in test_loader:
            results = predictor_triple.forward(inputs)
            _, predicted = torch.max(results.data, 1)
            predicted = list(predicted.detach().cpu().numpy())
            labels = list(labels.detach().cpu().numpy())
            mse += sum([(i - j) ** 2 for i, j in zip(list(results.detach().cpu().numpy()), one_hot(labels))])

            tp += len([i for i, j in zip(predicted, labels) if ((i == j) and (i == 1))])
            fp += len([i for i, j in zip(predicted, labels) if ((i != j) and (i == 1))])
            tn += len([i for i, j in zip(predicted, labels) if ((i == j) and (i == 0))])
            fn += len([i for i, j in zip(predicted, labels) if ((i != j) and (i == 0))])
    acc, pre, recall, f1 = gzh.getMetrics(tp, fp, tn, fn)
    mse / sum([tp, fp, tn, fn])
    print('三元组实时性预测结果：')
    print('MSE', mse[0])
    print('accuracy', acc)
    print('precision', pre)
    print('recall', recall)
    print('f1', f1)
    print('--' * 20)
    # -------------------------------------------------------------------------------------------------- #
    # -------------------------------------------------------------------------------------------------- #
    # 实体预测
    lr = 0.001
    trainRate = 0.7
    batch_size = 4
    max_len = 24
    EPOCH = 10

    predictor_entities = predictor(triple=False, max_len=max_len, language='English')
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(predictor_triple.parameters(), lr=lr)

    data = [i for i in fresh_entities]
    data.extend([i for i in unfresh_entities])
    data = random.sample(data, k=min(len(data), data_size))

    train = random.sample(data, k=int(len(data) * trainRate))
    trainX, trainY = [i[0] for i in train], [i[1] for i in train]

    test = [i for i in data if i not in train]
    test = random.sample(test, k=len(test))
    testX, testY = [i[0] for i in test], [i[1] for i in test]

    del (train)
    del (test)
    del (data)
    del (fresh_entities)
    del (unfresh_entities)

    train_dataset = DealDataset(trainX, trainY)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    for epoch in range(EPOCH):
        epoch_loss = 0
        for inputs, labels in train_loader:
            results = predictor_entities.forward(inputs)
            labels = torch.FloatTensor(one_hot(labels)).to('cuda')
            loss = criterion(results, labels)
            optimizer.zero_grad()
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
        print(epoch_loss)
    try:
        import os

        os.mkdir('./pkl')
    except:
        pass
    torch.save(predictor_entities.state_dict(), './pkl/predictor_entities.pkl')

    test_dataset = DealDataset(testX, testY)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    with torch.no_grad():
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        mse = 0
        for inputs, labels in test_loader:
            results = predictor_entities.forward(inputs)
            _, predicted = torch.max(results.data, 1)
            predicted = list(predicted.detach().cpu().numpy())
            labels = list(labels.detach().cpu().numpy())

            mse += sum([(i - j) * (i - j) for i, j in zip(list(results.detach().cpu().numpy()), one_hot(labels))])

            tp += len([i for i, j in zip(predicted, labels) if ((i == j) and (i == 1))])
            fp += len([i for i, j in zip(predicted, labels) if ((i != j) and (i == 1))])
            tn += len([i for i, j in zip(predicted, labels) if ((i == j) and (i == 0))])
            fn += len([i for i, j in zip(predicted, labels) if ((i != j) and (i == 0))])
    acc, pre, recall, f1 = gzh.getMetrics(tp, fp, tn, fn)
    mse / sum([tp, fp, tn, fn])
    print('实体实时性预测结果：')
    print('MSE', mse[0])
    print('accuracy', acc)
    print('precision', pre)
    print('recall', recall)
    print('f1', f1)
    print('--' * 20)
