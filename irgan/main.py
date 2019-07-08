from Generator import Generator
from Discriminator import Discriminator
from Dataloader import MQ2008
from Optimizer import Optimizer
import tensorflow as tf
import numpy as np

# size of input vector
feature_size = 784
# size of latent vector
hidden_size = 784
# keep probability
keep_prod = 0.5
# learning_rate
learning_rate = 0.00001
# batch_size
batch_size = 10
# generator training epochs
epochs = 10






class Dataset(MQ2008):
    def __init__(self, batch_size, dataset_dir='/work/ececis_research/Manning/'):#'/content/drive/My Drive/MQ2008-semi'):
    #def __init__(self, batch_size, dataset_dir='/content/drive/My Drive/MQ2008-semi'):

        MQ2008.__init__(self, dataset_dir=dataset_dir)
        self.batch_size = batch_size
        self.docs_pairs = []
    
    def set_docs_pairs(self, sess, generator):
        for query in dataset.get_pos_queries():
            can_docs = dataset.get_docs(query)
            #print(can_docs)
            can_features = [dataset.get_features(query, doc) for doc in can_docs]
            #print(can_features)
            #can_score = sess.run(generator.pred_score, feed_dict={generator.pred_data: can_features})
            print('can_docs:', can_docs)
        
            # softmax for candidate
            #exp_rating = np.exp(can_score)
            #prob = exp_rating / np.sum(exp_rating)
        
            pos_docs = dataset.get_pos_docs(query)
            print('pos_docs:', pos_docs)
            neg_docs = []
            #print(prob)
            #print('pos_docs:',pos_docs)
            for i in range(len(pos_docs)):
                #while True:
                for j in range(len(can_docs)):
                    #doc = np.random.choice(can_docs, p=prob)
                    doc = can_docs[j]
                    #print('doc:', doc)
                    if doc not in pos_docs:
                        neg_docs.append(doc)
                        break
            print('neg_docs:', neg_docs)
            for i in range(len(pos_docs)):              
                self.docs_pairs.append((query, pos_docs[i], neg_docs[i]))
        
    def get_batches(self):
        size = len(self.docs_pairs)
        cut_off = size // self.batch_size
        
        for i in range(0, self.batch_size * cut_off, self.batch_size):
            batch_pairs = self.docs_pairs[i:i+self.batch_size]
            yield np.asarray([self.get_features(p[0], p[1]) for p in batch_pairs]), np.asarray([self.get_features(p[0], p[2]) for p in batch_pairs])


def train_generator(sess, generator, discriminator, optimizer, dataset):
    for query in dataset.get_pos_queries():
        pos_docs = dataset.get_pos_docs(query)
        can_docs = dataset.get_docs(query)
        
        #print('pos_docs:', pos_docs)
        #print('can_docs:', can_docs)
        
        can_features = [dataset.get_features(query, doc) for doc in can_docs]
        #print('can_features:', can_features)
        can_score = sess.run(generator.pred_score, feed_dict={generator.pred_data: can_features})
        
        # softmax for all
        exp_rating = np.exp(can_score)
        prob = exp_rating / np.sum(exp_rating)
        
        # sampling
        neg_index = np.random.choice(np.arange(len(can_docs)), size=[len(pos_docs)], p=prob)
        neg_docs = np.array(can_docs)[neg_index]
        
        pos_features =  [dataset.get_features(query, doc) for doc in pos_docs]
        neg_features = [dataset.get_features(query, doc) for doc in neg_docs]
        
        neg_reward = sess.run(discriminator.reward,
                              feed_dict={discriminator.pos_data: pos_features, discriminator.neg_data: neg_features})
            
        _ = sess.run(optimizer.g_train_opt, 
                     feed_dict={generator.pred_data: can_features, generator.sample_index: neg_index, generator.reward: neg_reward})
            
    return sess.run(generator.opt_loss, 
                    feed_dict={generator.pred_data: can_features, generator.sample_index: neg_index, generator.reward: neg_reward})


def train_discriminator(sess, generator, discriminator, optimizer, dataset):
    dataset.set_docs_pairs(sess, generator)
    
    for input_pos, input_neg in dataset.get_batches():
        _ = sess.run(optimizer.d_train_opt,
                     feed_dict={discriminator.pos_data: input_pos, discriminator.neg_data: input_neg})
        
    return sess.run(discriminator.opt_loss, 
                    feed_dict={discriminator.pos_data: input_pos, discriminator.neg_data: input_neg})


def cal_mrr(r):
    num = 1
    for i in r:
        if i:
            break
        num += 1
    return 1. / num


def MRR(sess, discriminator, dataset):
    rs = []
    for query in dataset.get_pos_queries(target='test'):
        pos_docs = dataset.get_pos_docs(query, target='test')
        pred_docs = dataset.get_docs(query, target='test')
        
        pred_features = np.asarray([dataset.get_features(query, doc, target='test') for doc in pred_docs])

        #pred_list_feature = [query_url_feature[query][url] for url in pred_list]
        #pred_list_feature = np.asarray(pred_list_feature)
        pred_score = sess.run(discriminator.pred_score, feed_dict={discriminator.pred_data: pred_features})
        pred_doc_score = sorted(zip(pred_docs, pred_score), key=lambda x: x[1], reverse=True)
        #pred_url_score = zip(pred_list, pred_list_score)
        #pred_url_score = sorted(pred_url_score, key=lambda x: x[1], reverse=True)

        r = [0.0] * len(pred_score)
        for i in range(0, len(pred_score)):
            (url, score) = pred_doc_score[i]
            if url in pos_docs:
                r[i] = 1.0
        rs.append(r)

    return np.mean([cal_mrr(r) for r in rs])


if __name__ == '__main__':
    dataset = Dataset(batch_size)
    generator = Generator(feature_size, hidden_size, keep_prod)
    discriminator = Discriminator(feature_size, hidden_size, keep_prod)
    optimizer = Optimizer(generator, discriminator, learning_rate)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for e in range(epochs):
            g_loss = train_generator(sess, generator, discriminator, optimizer, dataset)
            d_loss = train_discriminator(sess, generator, discriminator, optimizer, dataset)
            mrr_best = MRR(sess, discriminator, dataset)

        
            print("Epoch {}/{}...".format(e+1, epochs),
              "Generator Loss: {:.4f}".format(g_loss), 
              "Discriminator Loss: {:.4f}".format(d_loss),
              #"NDCG@3: {:.4f}".format(ndcg_at_3),
              #"NDCG@5: {:.4f}".format(ndcg_at_5),
              #"NDCG@10: {:.4f}".format(ndcg_at_10),
              "MRR: {:.4f}".format(mrr_best))                                   












