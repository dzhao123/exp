import tensorflow as tf
import tensorflow_hub as hub

from tqdm import tqdm
import random
import numpy as np

import logging
import torch
import torch.nn as nn
import argparse
import csv

from DataLoader import DataLoader
from DataLoaderTest import DataLoaderTest
from datetime import datetime

#train_data = DataLoader(
#            data='/content/drive/My Drive/msmarco/bert_train.txt',
#            batch_size=100,
#            cuda=False)

valid_data = DataLoaderTest(
            data='/work/ececis_research/Manning/bert_dev.txt',
            batch_size=200,
            cuda=False)


class feature(object):
  
  def __init__(self, bert_module, sess, qrels):
    self.sess = sess
    self.sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
    self.bert_module = bert_module

    self.qd = {}
    self.qrels = self.get_qrels(qrels)
    
    self.train_pack = ''
    self.dev_pack = ''
    
  def get_qrels(self, qrels):
    with open(qrels) as qfile:
      data = qfile.readlines()
    
    for line in data:
      qrel, _, doc, _ = line.strip().split('\t')
      if qrel not in self.qd:
        self.qd[qrel] = [doc]
      else:
        self.qd[qrel].append(doc) 
    
  
  def extractor(self, input_ids, input_mask, segment_ids):#, sess):
    # Feature based (used as embedding)
    #bert_module = hub.Module("https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1", trainable=False)
    # Fine-tuning BERT model
    # bert_module = hub.Module("https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1", trainable=True)
    #sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

    bert_inputs = dict(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids)
    bert_outputs = self.bert_module(bert_inputs, signature="tokens", as_dict=True)
    #pooled_output = bert_outputs["pooled_output"]
    #sequence_output = bert_outputs["sequence_output"] 
    message_embeddings = self.sess.run(bert_outputs)
    pooled_output = message_embeddings["pooled_output"]
    print('extraction finished')
    return pooled_output

  
  def train_data_extract(self, train_data):#, sess):
    counter = 0
    with open('irgan_train.txt', 'w') as file:
      for batch in train_data:
        
        time_start = datetime.now()

        inputs_q, inputs_d_pos, inputs_d_neg, mask_q, mask_d_pos, mask_d_neg = batch

        pos_data = torch.cat([inputs_q, inputs_d_pos], 1)#.numpy()
        neg_data = torch.cat([inputs_q, inputs_d_neg], 1)#.numpy()
        pos_segs = torch.cat([torch.zeros_like(inputs_q), torch.ones_like(inputs_d_pos)], 1)#.numpy()
        neg_segs = torch.cat([torch.zeros_like(inputs_q), torch.ones_like(inputs_d_neg)], 1)#.numpy()
        pos_mask = torch.cat([mask_q, mask_d_pos], 1)#.numpy()
        neg_mask = torch.cat([mask_q, mask_d_neg], 1)#.numpy()
        
        
        data = torch.cat([pos_data, neg_data],0).numpy()
        segs = torch.cat([pos_segs, neg_segs],0).numpy()
        mask = torch.cat([pos_mask, neg_mask],0).numpy()
        
        output = self.extractor(data, mask, segs)
        

        #os_output = self.extractor(pos_data.numpy(), pos_mask.numpy(), pos_segs.numpy())
        
        half = len(output)//2

        for j in range(half):
          self.train_pack += '1' + ' ' + 'q%d'%counter + ' ' + ' '.join(str(i) for i in output[j]) + ' ' + 'pd%d'%counter + ' ' + '1' + '\n'
          self.train_pack += '-1' + ' ' + 'q%d'%counter + ' ' + ' '.join(str(i) for i in output[j+half]) + ' ' + 'nd%d'%counter + ' ' + '-1' + '\n'
          counter += 1
          print('counter:',counter)
        #ounter += half
        file.write(self.train_pack)
        self.train_pack = ''
          
        print('pack finished')

        time = datetime.now() - time_start
        print('time takes:', time)


  def valid_data_extract(self, valid_data):
    with open('/work/ececis_research/Manning/irgan_dev.txt', 'w') as file:
      for batch in valid_data:
        time_start = datetime.now()
        inputs_q, inputs_d, mask_q, mask_d, docid, qid = batch
        #print('docid:', docid)
        #print('qid:', qid)

        data = torch.cat([inputs_q, inputs_d], 1).numpy()
        segs = torch.cat([torch.zeros_like(inputs_q), torch.ones_like(inputs_d)], 1).numpy()
        mask = torch.cat([mask_q, mask_d], 1).numpy()

        output = self.extractor(data, mask, segs)
        #print('output:', output)
        
        #self.pack(data, 'dev')

        for j in range(len(output)):
            if docid[j] in self.qd[qid[j]]:
                #print('docid:', docid[j])
                #print('qid:', qid[j])
                line = '1' + ' ' + qid[j] + ' ' + ' '.join(str(i) for i in output[j]) + ' ' + docid[j] + ' ' + '1' + '\n'
                #line = '1' + ' ' + qid + ' ' + data + ' ' + segs + ' ' + mask + ' ' + docid + ' ' + '1' + '\n'
            elif docid[j] not in self.qd[qid[j]]:
                #print('docid:', docid[j])
                #print('qid:', qid[j])
                line = '-1' + ' ' + qid[j] + ' ' + ' '.join(str(i) for i in output[j]) + ' ' + docid[j] + ' ' + '-1' + '\n'
                #line = '-1' + ' ' + qid + ' ' + data + ' ' + segs + ' ' + mask + ' ' + docid + ' ' + '-1' + '\n'
            self.dev_pack += line
        #print(self.dev_pack)

        file.write(self.dev_pack)
        del self.dev_pack
        del data
        del segs
        del mask
        del output
        del inputs_q
        del inputs_d
        del mask_q
        del mask_d
        del docid
        del qid
        
        self.dev_pack = ''
        print('pack finished')

        time = datetime.now() - time_start
        print('time takes:', time)



if __name__ == '__main__':
    bert_module = hub.Module("https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1", trainable=False)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    qrels = '/work/ececis_research/Manning/qrels.dev.tsv'
    #feature(bert_module, sess, qrels).train_data_extract(train_data)#, sess)
    feature(bert_module, sess, qrels).valid_data_extract(valid_data)#, sess)




