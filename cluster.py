from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
import tqdm
from dbpedia_freshness import predictor, DealDataset

if __name__ == '__main__':
    f = open('dbpedia_0601.raw.txt', encoding='utf-8')

    language = 'English'
    max_len = 64
    triple = True
    database_name = 'dbpedia_raw'
    batch_size = 32
    classes = 10

    predictor_tiples = predictor(triple=triple, max_len=max_len, language=language)

    getDataset = getDataset(database_name=database_name)
    fresh_triples, unfresh_triples = getDataset.get_triples()
    getDataset.clear_cache()

    data = []

    data.extend(fresh_triples)
    data.extend(unfresh_triples)
    print(data[:10])
    train_dataset = DealDataset(data)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    for triple in fresh_triples:
        data.extend(predictor_tiples.get_embedding(triple).detach().cpu().numpy())
    classes = KMeans(n_clusters=classes).predict(data)
    print(len(classes))
