# How to run the code
## dbpedia
1. put dbpedia zip file into ./dbpedia_freshness/dbpedia_xxxx
   - xxxx means the version of dbpedia, such as "0601" etc.
   
2. run preprocess.py
3. run find_unfresh_triples&entities.py
4. run dbpedia_freshness.py
5. run dbpedia_sample.py

## cndbpedia
1. We have already prepared the training data in ./cndbpedia_freshness/dataset
2. run predictor.py to show the prediction of unfreshness by the indicator we proposed in the paper
3. run sample.py to show the result of stratfication sampling 

