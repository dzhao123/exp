import numpy as np

class MQ2008NoTargetException(Exception):
    pass

class MQ2008:
    def __init__(self, dataset_dir='/work/ececis_research/Manning/'):#'drive/My Drive/MQ2008-semi'):
    #def __init__(self, dataset_dir='/drive/My Drive/MQ2008-semi'):

        self.pool = {}
        self.pool['train'] = self._load_data(dataset_dir + 'irgan_train.txt')#'/train.txt')
        self.pool['test'] = self._load_data(dataset_dir + 'irgan_dev2.txt')#'/test.txt')
        #self.pool['train'] = self._load_data(dataset_dir + '/train.txt')
        #self.pool['test'] = self._load_data(dataset_dir + '/test.txt')

        
    def get_queries(self, target='train'):
        if target in self.pool.keys():
            return list(self.pool[target].keys())
        else:
            raise MQ2008NoTargetException()

    def get_docs(self, query, target='train'):
        if target in self.pool.keys():
            return list(self.pool[target][query].keys())
        else:
            raise MQ2008NoTargetException()
    
    def get_features(self, query, doc, target='train'):
        if target in self.pool.keys():
            return self.pool[target][query][doc]['f']
        else:
            raise MQ2008NoTargetException()
    
    def get_rank(self, query, doc, target='train'):
        if target in self.pool.keys():
            return self.pool[target][query][doc]['r']
        else:
            raise MQ2008NoTargetException()
 
    def get_pos_queries(self, target='train'):
        if target in self.pool.keys():
            return list({query for query in self.get_queries(target=target)
                         for doc in self.get_docs(query, target=target) if self.get_rank(query, doc, target=target) > 0.0})
        else:
            raise MQ2008NoTargetException()
            
    def get_pos_docs(self, query, target='train'):
        if target in self.pool.keys():
            return list({doc for doc in self.get_docs(query, target=target) if self.get_rank(query, doc, target=target) > 0.0})
        else:
            raise MQ2008NoTargetException()

    # load docs and features for a query.
    def _load_data(self, file, feature_size=46):#46
        query_doc_feature = {}
        with open(file) as f:
            for line in f:
                cols = line.strip().split(' ')
                #print('cols:', cols)
                rank = cols[0]
                #print('rank:', rank)
                #query = cols[1].split(':')[1]
                query = cols[1]#.split(' ')[1]
                #print('query:', query)
                doc = cols[-2]#cols[-7]
                #print('doc:', doc)
                #doc = cols[-7]
                feature = []
                for i in range(2, 2 + feature_size):
                    #feature.append(float(cols[i].split(':')[1]))
                    feature.append(float(cols[i]))#.split(' ')[1]))
                if query in query_doc_feature.keys():
                    query_doc_feature[query][doc] = {'r': float(rank), 'f': np.array(feature)}
                else:
                    query_doc_feature[query] = {doc: {'r': float(rank), 'f': np.array(feature)}}
        return query_doc_feature
