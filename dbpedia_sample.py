from dbpedia_freshness import predictor, DealDataset
import tqdm
import torch
from torch.utils.data import DataLoader
import random

if __name__ == '__main__':
    import numpy as np
    import config

    max_len = 24
    data_size = 100000
    batch_size = 128

    predictor_triples = predictor(triple=True, max_len=max_len, language='English')
    # predictor_entities = predictor(triple=False, max_len=max_len, language='English')

    predictor_triples.load_state_dict(torch.load('./pkl/predictor_triples.pkl'))
    # predictor_entities.load_state_dict(torch.load('./pkl/predictor_entities.pkl'))

    f = open('dbpedia_0601.raw.txt', encoding='utf-8')
    data = []
    for index, line in tqdm.tqdm(enumerate(f)):
        if index >= data_size:
            break
        try:
            data.append(line.strip().split('\t'))
        except:
            pass

    # del (fresh_triples)
    # del (unfresh_triples)

    data = random.sample(data, k=min(100000, len(data)))
    X, Y = [i[:3] for i in data], [1 for i in data]
    train_dataset = DealDataset(X, Y)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    f = open('dbpedia_unfresh.txt', 'w', encoding='utf-8')
    num = 0
    for inputs, _ in train_loader:
        results = predictor_triples.forward(inputs)
        _, predicted = torch.max(results.data, 1)
        predicted = predicted.detach().cpu().numpy()

        ss = np.array(inputs[0])[predicted == 0]
        ps = np.array(inputs[1])[predicted == 0]
        os = np.array(inputs[2])[predicted == 0]
        for s, p, o in zip(ss, ps, os):
            num += 1
            f.write('\t'.join([s, p, o]) + '\n')
    print('数据集总数：', data_size)
    print('MLP unfresh个数', num)
