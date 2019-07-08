import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertConfig
from pytorch_pretrained_bert import modeling

from DataLoader import DataLoader
from DataLoaderTest import DataLoaderTest
from datetime import datetime


#output_model_file = '/work/ececis_research/Manning/finetune'
 
#input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
#input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
#token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

#bert_model = 'bert-base-uncased'
#model_state_dict= torch.load(output_model_file, map_location='cpu')
#model = BertModel.from_pretrained(output_model_file)

#all_output, pooled_output = model(input_ids, token_type_ids, input_mask, output_all_encoded_layers=False)

#print(pooled_output.shape)







class feature(object):
  
  def __init__(self, bert_module, qrels):
    #self.sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
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

    #bert_inputs = dict(
    #  input_ids=input_ids,
    #  input_mask=input_mask,
    #  segment_ids=segment_ids)
    #bert_outputs = self.bert_module(bert_inputs, signature="tokens", as_dict=True)
    #pooled_output = bert_outputs["pooled_output"]
    #sequence_output = bert_outputs["sequence_output"] 
    #message_embeddings = self.sess.run(bert_outputs)
    #pooled_output = message_embeddings["pooled_output"]
    all_output, pooled_output = self.bert_module(input_ids, segment_ids, input_mask, output_all_encoded_layers=False)

    print('extraction finished')
    return pooled_output

  
  #def pack(self, pos, neg, mode):
  #  counter = 0
  #  if mode == 'train':
      #pos = data[0]
      #neg = data[1]
      #print(pos.shape)
      #print(neg.shape)
      #print(len(pos))
      #print(len(neg))
  #    with open('irgan_train.txt', 'a') as file:
  #      for j in range(len(pos)):
  #        self.train_pack += '1' + ' ' + 'q%d'%counter + ' ' + ' '.join(str(i) for i in pos[j]) + ' ' + 'pd%d'%counter + ' ' + '1' + '\n'
  #        self.train_pack += '-1' + ' ' + 'q%d'%counter + ' ' + ' '.join(str(i) for i in neg[j]) + ' ' + 'nd%d'%counter + ' ' + '-1' + '\n'
  #        counter += 1
  #      file.write(self.train_pack)
  #      self.train_pack = ''
          
      #open('irgan_train.txt', 'w').write('1' + ' ' + 'q%d'%counter + ' ' + ' '.join(str(i) for i in pos[j]) + ' ' + 'pd%d'%counter + ' ' + '1' + '\n' for j in range(len(pos))) 
      #open('irgan_train.txt', 'w').write('-1' + ' ' + 'q%d'%counter + ' ' + ' '.join(str(i) for i in neg[j]) + ' ' + 'nd%d'%counter + ' ' + '-1' + '\n' for j in range(len(pos))) 
  #    print('pack finished')


  #  elif mode == 'dev':
  #    pass
  
  
  
  
  def train_data_extract(self, train_data):#, sess):
    counter = 0
    with open('/work/ececis_research/Manning/irgan_train2.txt', 'w') as file:
      for batch in train_data:
        
        time_start = datetime.now()

        inputs_q, inputs_d_pos, inputs_d_neg, mask_q, mask_d_pos, mask_d_neg = batch

        pos_data = torch.cat([inputs_q, inputs_d_pos], 1)#.numpy()
        neg_data = torch.cat([inputs_q, inputs_d_neg], 1)#.numpy()
        pos_segs = torch.cat([torch.zeros_like(inputs_q), torch.ones_like(inputs_d_pos)], 1)#.numpy()
        neg_segs = torch.cat([torch.zeros_like(inputs_q), torch.ones_like(inputs_d_neg)], 1)#.numpy()
        pos_mask = torch.cat([mask_q, mask_d_pos], 1)#.numpy()
        neg_mask = torch.cat([mask_q, mask_d_neg], 1)#.numpy()
        
        
        data = torch.cat([pos_data, neg_data],0)#.numpy()
        segs = torch.cat([pos_segs, neg_segs],0)#.numpy()
        mask = torch.cat([pos_mask, neg_mask],0)#.numpy()
        
        output = self.extractor(data, mask, segs)
        output = output.detach().numpy()
        #print(output) 

        #os_output = self.extractor(pos_data.numpy(), pos_mask.numpy(), pos_segs.numpy())
        
        half = len(output)//2
        #rint(output)
        #SAt(pos_output)
       #for i in range(half):
        # try:
        #   assert np.array_equal(output[i], neg_output[i])
            
        # except:
        #   print(output[i])
        #   print(pos_output[i])
            
        #neg_output = self.extractor(neg_data, neg_mask, neg_segs)
        
        #data = [pos_output, neg_output]
        #self.pack(pos_output, neg_output, 'train')
        #alf = len(output)//2
        for j in range(half):
          self.train_pack += '1' + ' ' + 'q%d'%counter + ' ' + ' '.join(str(i) for i in output[j]) + ' ' + 'pd%d'%counter + ' ' + '1' + '\n'
          self.train_pack += '-1' + ' ' + 'q%d'%counter + ' ' + ' '.join(str(i) for i in output[j+half]) + ' ' + 'nd%d'%counter + ' ' + '-1' + '\n'
          counter += 1
          print('counter:',counter)
        #ounter += half
        file.write(self.train_pack)
        self.train_pack = ''
          
      #open('irgan_train.txt', 'w').write('1' + ' ' + 'q%d'%counter + ' ' + ' '.join(str(i) for i in pos[j]) + ' ' + 'pd%d'%counter + ' ' + '1' + '\n' for j in range(len(pos))) 
      #open('irgan_train.txt', 'w').write('-1' + ' ' + 'q%d'%counter + ' ' + ' '.join(str(i) for i in neg[j]) + ' ' + 'nd%d'%counter + ' ' + '-1' + '\n' for j in range(len(pos))) 
        print('pack finished')
        #pos = '1' + ' ' + 'q%d'%counter + ' ' + ' '.join(str(i) for i in pos_output[0]) + ' ' + 'pd%d'%counter +\
        #      ' ' +'1' + '\n'
        #neg = '-1' + ' ' + 'q%d'%counter + ' ' + ' '.join(str(i) for i in neg_output[0]) + ' ' + 'nd%d'%counter +\
        #      ' ' + '-1' + '\n'
        #print(pos_data)
        #print(neg_data)
        #pos = '1' + ' ' + 'q%d'%counter + ' ' + pos_data + ' ' + pos_segs + ' ' + pos_mask + ' ' + 'pd%d'%counter + ' ' + '1' + '\n'
        #neg = '-1' + ' ' + 'q%d'%counter + ' ' + neg_data + ' ' + neg_segs + ' ' + neg_mask + ' ' + 'nd%d'%counter + ' ' + '-1' + '\n'
        time = datetime.now() - time_start
        print('time takes:', time)
        break
        #print(pos)
        #print(neg)
        #ounter += 1
        #file.write(pos)
        #file.write(neg)  


  def valid_data_extract(self, valid_data):
    with open('/work/ececis_research/Manning/irgan_dev8.txt', 'w') as file:
      for index, batch in enumerate(valid_data):
        if index < 40000:
          continue
        if index >= 50000:
          break
        print('batch_index:',index)
        inputs_q, inputs_d, mask_q, mask_d, docid, qid = batch  

        data = torch.cat([inputs_q, inputs_d], 1)#.numpy()
        segs = torch.cat([torch.zeros_like(inputs_q), torch.ones_like(inputs_d)], 1)#.numpy()
        mask = torch.cat([mask_q, mask_d], 1)#.numpy()

        output = self.extractor(data, mask, segs)
        output = output.detach().numpy()
        
        time_start = datetime.now()
        #self.pack(data, 'dev')

        for i in range(len(output)):
          #print('qury:', qid[i])
          #print('docs:', self.qd[qid[i]])
          if docid[i] in self.qd[qid[i]]:
            line = '1' + ' ' + qid[i] + ' ' + ' '.join(str(k) for k in output[i]) + ' ' + docid[i] + ' ' + '1' + '\n'
  #         line = '1' + ' ' + qid + ' ' + data + ' ' + segs + ' ' + mask + ' ' + docid + ' ' + '1' + '\n'
          elif docid[i] not in self.qd[qid[i]]:
            line = '-1' + ' ' + qid[i] + ' ' + ' '.join(str(k) for k in output[i]) + ' ' + docid[i] + ' ' + '-1' + '\n'
  #         line = '-1' + ' ' + qid + ' ' + data + ' ' + segs + ' ' + mask + ' ' + docid + ' ' + '-1' + '\n'
          self.dev_pack += line

        file.write(self.dev_pack)
        self.dev_pack = ''

        time = datetime.now() - time_start
        print('time takes:', time)
    
      

if __name__ == '__main__':
    output_model_file = '/work/ececis_research/Manning/finetune'
    qrels = '/work/ececis_research/Manning/qrels.dev.tsv'
    bert_module = BertModel.from_pretrained(output_model_file)

    train_data = DataLoader(
            data='/work/ececis_research/Manning/bert_train.txt',
            batch_size=2,
            cuda=False)

    valid_data = DataLoaderTest(
            data='/work/ececis_research/Manning/bert_dev.txt',
            batch_size=100,
            cuda=False)

    feat = feature(bert_module, qrels)
    #feat.train_data_extract(train_data)
    feat.valid_data_extract(valid_data)









