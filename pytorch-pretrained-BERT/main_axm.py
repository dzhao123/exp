from tqdm import tqdm
import random

import logging
import torch
import torch.nn as nn
import argparse
import csv

from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification, BertConfig
from pytorch_pretrained_bert.optimization import BertAdam
from torch.autograd import Variable
from DataLoader_axm import DataLoader
from DataLoaderTest import DataLoaderTest
from datetime import datetime



def get_qrels(QRELS_DEV):
    qrels = {}
    with open(QRELS_DEV, mode='r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            qid = row[0]
            did = row[2]
            if qid not in qrels:
                qrels[qid] = []
            qrels[qid].append(did)
    return qrels



def data_evaluate(model, evaluate_data, flag, qrels):
    count = 0
    eval_dict = dict()
    c_1_j = 0
    c_2_j = 0
    reduce_num = 0
    counter = 0
    for batch in evaluate_data:
        count += len(batch)
        inputs_q, inputs_d, mask_q, mask_d, docid, qid = batch
        model.eval()
        data = torch.cat([inputs_q, inputs_d], 1)
        segs = torch.cat([torch.zeros_like(inputs_q), torch.ones_like(inputs_d)], 1)
        mask = torch.cat([mask_q, mask_d], 1)

        outputs = model(data, segs, mask)
        output = outputs.cpu().data.tolist()
        #print(outputs)
        # output = outputs.data.tolist()
        tuples = zip(qid, docid, output)
        for item in tuples:
            if item[0] not in eval_dict: # id not in eval dict
                eval_dict[item[0]] = []
            eval_dict[item[0]].append((item[1], item[2])) # {id: [(docid, output)]}
        #print('count:', count)

    no_label = 0
    for qid, value in eval_dict.items():
        counter += 1
        #print('counter:',counter)
        if qid not in qrels:
            no_label += 1
            continue
        res = sorted(value, key=lambda x: x[1], reverse=True) # reverse sort by value[1] (output), res=[(docid,score)]
        count = 0.0
        score = 0.0
        for i in range(len(res)):
            if res[i][0] in qrels[qid]:#if docid in this qrel[qid]'s docid list(which means it is relevant)
                count += 1
                score += count / (i+1) # + pos doc number/total doc num
        for i in range(len(res)):
            if res[i][0] in qrels[qid]:
                c_2_j += 1 / float(i+1)
                break
        if count != 0:
            c_1_j += score / count
        else: # a question without pos doc
            reduce_num += 1

    print(len(eval_dict), no_label)
    MAP = c_1_j / float(len(eval_dict) - no_label)
    MRR = c_2_j / float(len(eval_dict) - no_label) #
    #print ""
    #print(" evaluate on " + flag + " MAP: %f" % MAP)
    #print(" evaluate on " + flag + ' MRR: %f' % MRR)
    logging.info(" evaluate on " + flag + " MAP: %f" % MAP)
    logging.info(" evaluate on " + flag + ' MRR: %f' % MRR)
    return MAP, MRR


def train(model, opt, crit, optimizer, train_data, dev_data):
    step = 0
    best_map_dev = 0.0
    best_mrr_dev = 0.0
    best_map_test = 0.0
    best_mrr_test = 0.0
    qrels=get_qrels('/work/ececis_research/Manning/qrels.dev.tsv')
    for epoch_i in range(opt.epoch):
        total_loss = 0.0
        time_epstart=datetime.now()
        for batch in train_data:
            # prepare data
            #inputs_q, inputs_d_pos, inputs_d_neg, mask_q, mask_d_pos, mask_d_neg = batch
            #print(inputs_q.shape)
            #print(inputs_d_pos.shape)
            #pos_data = torch.cat([inputs_q, inputs_d_pos], 1)
            #neg_data = torch.cat([inputs_q, inputs_d_neg], 1)
            #pos_segs = torch.cat([torch.zeros_like(inputs_q), torch.ones_like(inputs_d_pos)], 1)
            #neg_segs = torch.cat([torch.zeros_like(inputs_q), torch.ones_like(inputs_d_neg)], 1)
            #pos_mask = torch.cat([mask_q, mask_d_pos], 1)
            #neg_mask = torch.cat([mask_q, mask_d_neg], 1)
            #print(pos_data.shape)
            #print(pos_mask.shape)
            
            
            # forward
            #optimizer.zero_grad()
            #model.train()
            #outputs_pos = model(pos_data, pos_segs, pos_mask)
            #outputs_neg = model(neg_data, neg_segs, neg_mask)
            #label = torch.ones(outputs_pos.size())#[1,1,1,1...]
            inputs_q, inputs_d_pos, inputs_d_neg, inputs_d_pos_axm, inputs_d_neg_axm, inputs_d_pos_adj, inputs_d_neg_adj, mask_q, mask_d_pos, mask_d_neg, mask_d_pos_axm, mask_d_neg_axm, mask_d_pos_adj, mask_d_neg_adj = batch
            #print(torch.max(inputs_q))
            #print(torch.max(inputs_q))
            #print(torch.max(inputs_d_pos))
            #print(torch.max(inputs_d_neg))
            #print(torch.max(inputs_d_pos_axm))
            #print(torch.max(inputs_d_neg_axm))
            #print(torch.max(inputs_d_pos_adj))
            #print(torch.max(inputs_d_neg_adj))
            pos_data = torch.cat([inputs_q, inputs_d_pos], 1)
            neg_data = torch.cat([inputs_q, inputs_d_neg], 1)
            pos_segs = torch.cat([torch.zeros_like(inputs_q), torch.ones_like(inputs_d_pos)], 1)
            neg_segs = torch.cat([torch.zeros_like(inputs_q), torch.ones_like(inputs_d_neg)], 1)
            pos_mask = torch.cat([mask_q, mask_d_pos], 1)
            neg_mask = torch.cat([mask_q, mask_d_neg], 1)

            pos_axm = torch.cat([inputs_q, inputs_d_pos_axm], 1)
            neg_axm = torch.cat([inputs_q, inputs_d_neg_axm], 1)
            pos_axm_segs = torch.cat([torch.zeros_like(inputs_q), torch.ones_like(inputs_d_pos_axm)], 1)
            neg_axm_segs = torch.cat([torch.zeros_like(inputs_q), torch.ones_like(inputs_d_neg_axm)], 1)
            pos_axm_mask = torch.cat([mask_q, mask_d_pos_axm], 1)
            neg_axm_mask = torch.cat([mask_q, mask_d_neg_axm], 1)

            pos_adj = torch.cat([inputs_q, inputs_d_pos_adj], 1)
            neg_adj = torch.cat([inputs_q, inputs_d_neg_adj], 1)
            pos_adj_segs = torch.cat([torch.zeros_like(inputs_q), torch.ones_like(inputs_d_pos_adj)], 1)
            neg_adj_segs = torch.cat([torch.zeros_like(inputs_q), torch.ones_like(inputs_d_neg_adj)], 1)
            pos_adj_mask = torch.cat([mask_q, mask_d_pos_adj], 1)
            neg_adj_mask = torch.cat([mask_q, mask_d_neg_adj], 1)




            # forward
            optimizer.zero_grad()
            model.train()


            outputs_pos = model(pos_data, pos_segs, pos_mask)
            outputs_neg = model(neg_data, neg_segs, neg_mask)

            outputs_pos_axm = model(pos_axm, pos_axm_segs, pos_axm_mask)#model(inputs_q, inputs_d_pos_axm, mask_q, mask_d_pos_axm)
            outputs_neg_axm = model(neg_axm, neg_axm_segs, neg_axm_mask)#model(inputs_q, inputs_d_neg_axm, mask_q, mask_d_neg_axm)

            outputs_pos_adj = model(pos_adj, pos_adj_segs, pos_adj_mask)#model(inputs_q, inputs_d_pos_adj, mask_q, mask_d_pos_adj)
            outputs_neg_adj = model(neg_adj, neg_adj_segs, neg_adj_mask)#model(inputs_q, inputs_d_neg_adj, mask_q, mask_d_neg_adj)



            #outputs_pos = model(inputs_q, inputs_d_pos, mask_q, mask_d_pos)
            #outputs_neg = model(inputs_q, inputs_d_neg, mask_q, mask_d_neg)

            #outputs_pos_axm = model(inputs_q, inputs_d_pos_axm, mask_q, mask_d_pos_axm)
            #outputs_neg_axm = model(inputs_q, inputs_d_neg_axm, mask_q, mask_d_neg_axm)

            #outputs_pos_adj = model(inputs_q, inputs_d_pos_adj, mask_q, mask_d_pos_adj)
            #outputs_neg_adj = model(inputs_q, inputs_d_neg_adj, mask_q, mask_d_neg_adj)

            label = torch.ones(outputs_pos.size())#[1,1,1,1...]

            crit1 = nn.MarginRankingLoss(margin=0.01, size_average=True)
            crit2 = nn.MarginRankingLoss(margin=0.01, size_average=True)

            #label1 = 0.001*label[:]
            #label2 = 0.001*label[:]

            #try:
            if opt.cuda:
                label = label.cuda()
                #label1 = label1.cuda()
                #label2 = label2.cuda()
            batch_loss_origin = crit(outputs_pos, outputs_neg, Variable(label, requires_grad=False))
            batch_loss_pos = crit1(0.01*outputs_pos_axm, 0.01*outputs_pos_adj, Variable(0.01*label, requires_grad=False))
            batch_loss_neg = crit2(0.01*outputs_neg_axm, 0.01*outputs_neg_adj, Variable(0.01*label, requires_grad=False))
            #print(batch_loss_origin)
            #print(batch_loss_pos)
            #print(batch_loss_neg)
            batch_loss = batch_loss_origin + 0.01*batch_loss_pos + 0.01*batch_loss_neg




            #if opt.cuda:
            #    label = label.cuda()
            #batch_loss = crit(outputs_pos, outputs_neg, Variable(label, requires_grad=False))

            # backward
            batch_loss.backward()

            # update parameters
            optimizer.step()
            step += 1
            total_loss += batch_loss.data[0]

            if step % opt.eval_step == 0:
                time_step=datetime.now()-time_epstart
                print(' Epoch %d Training step %d loss %f this epoch time %s' %(epoch_i, step, total_loss,time_step))
                with open(opt.task+".txt",'a') as logf:
                    logf.write(' Epoch %d Training step %d loss %f this epoch time %s\n' %(epoch_i, step, total_loss,time_step))
                map_dev, mrr_dev = data_evaluate(model, dev_data, "dev", qrels)
                #map_test, mrr_test = data_evaluate(model, test_data, "test")
                # lets just use dev first...so modify like this:
                map_test=map_dev
                mrr_test=mrr_dev
                print('map_test:', map_test)
                print('mrr_test:', mrr_test)
                
                report_loss = total_loss
                total_loss = 0
                if map_dev >= best_map_dev:
                    best_map_dev = map_dev
                    best_map_test = map_test
                    best_mrr_dev = mrr_dev
                    best_mrr_test = mrr_test
                    print ("best dev-- mrr %f map %f; test-- mrr %f map %f" % (
                    best_mrr_dev, best_map_dev, best_mrr_test, best_map_test))
                    with open(opt.task+".txt",'a') as logf:
                        logf.write("best dev-- mrr %f map %f; test-- mrr %f map %f\n" % (
                    best_mrr_dev, best_map_dev, best_mrr_test, best_map_test))
                else:
                    print("NOT the best dev-- mrr %f map %f; test-- mrr %f map %f" %(mrr_dev,map_dev,mrr_test,map_test))
                    with open(opt.task+".txt",'a') as logf:
                        logf.write("NOT the best dev-- mrr %f map %f; test-- mrr %f map %f\n" %(mrr_dev,map_dev,mrr_test,map_test))
                    
                if opt.save_model:
                    model_state_dict = model.state_dict()
                    checkpoint = {
                        'model': model_state_dict,
                        'settings': opt,
                        'epoch': epoch_i}
                    if opt.save_mode == 'all':
                        model_name = '/work/ececis_research/Manning/' + opt.save_model + '_step_{}.chkpt'.format(step)
                        torch.save(checkpoint, model_name)
                    elif opt.save_mode == 'best':
                        model_name = '/work/ececis_research/Manning/' + opt.save_model + '.chkpt'
                        if map_dev == best_map_dev:
                            best_map_dev = map_dev
                            best_map_test = map_test
                            best_mrr_dev = mrr_dev
                            best_mrr_test = mrr_test
                            with open(opt.task+".txt",'a') as logf:# record log
                                logf.write(' Epoch %d Training step %d loss %f this epoch time %s' %(epoch_i, step, report_loss,time_step))
                                logf.write("best dev-- mrr %f map %f; test-- mrr %f map %f" %(best_mrr_dev, best_map_dev, best_mrr_test, best_map_test))
                            torch.save(checkpoint, model_name)
                            print('    - [Info] The checkpoint file has been updated.')
                            with open(opt.task+".txt",'a') as logf:
                                logf.write('    - [Info] The checkpoint file has been updated.\n')
        time_epend=datetime.now()
        time_ep=time_epend-time_epstart
        print('train epoch '+str(epoch_i)+' using time: '+ str(time_ep))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode',type=str,choices=['train','forward'],default='train')
    parser.add_argument('-train_data')
    parser.add_argument('-val_data')
    parser.add_argument('-test_data')
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-epoch', type=int, default=1)
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-task', choices=['KNRM', 'CKNRM', 'MAXPOOL', 'AVGPOOL', 'LSTM', 'BERT', 'BERT2', 'BERT3', 'BERTH', 'BERT1A'])
    parser.add_argument('-eval_step', type=int, default=6000)
    parser.add_argument('-save_model', default=None)
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')
    parser.add_argument('-learning_rate', type=float, default=1e-5)

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda


    training_data = DataLoader(
            data=opt.train_data,
            batch_size=opt.batch_size,
            cuda=opt.cuda)

    validation_data = DataLoaderTest(
            data=opt.val_data,
            batch_size=10,#opt.batch_size,
            test=True,
            cuda=opt.cuda)


    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)

    learning_rate = opt.learning_rate#1e-3
    warmup_proportion = 0.1
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10%% of training."
    #num_train_epochs = 10

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    num_train_steps = 39780811#len(train_data_set) * num_train_epochs
    optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if n not in no_decay], 'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if n in no_decay], 'weight_decay_rate': 0.0}
    ]

    optimizer = BertAdam(optimizer_grouped_parameters,
                lr=learning_rate,
                warmup=warmup_proportion,
                t_total=num_train_steps)

    crit = nn.MarginRankingLoss(margin=1, size_average=True)
    
    if opt.cuda:
        model = model.cuda()
        crit = crit.cuda()

    train(model, opt, crit, optimizer, training_data, validation_data)
    
    print("finish")



if __name__ == '__main__':
    main()
