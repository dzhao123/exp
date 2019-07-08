from tqdm import tqdm
import random 

import torch

from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification, BertConfig
from pytorch_pretrained_bert.optimization import BertAdam


def train(model, opt, crit, optimizer, train_data, dev_data, test_data):
    ''' Start training '''
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
            inputs_q, inputs_d_pos, inputs_d_neg, mask_q, mask_d_pos, mask_d_neg = batch    


            
            # forward
            optimizer.zero_grad()
            model.train()
            outputs_pos = model(inputs_q, inputs_d_pos, mask_q, mask_d_pos)
            outputs_neg = model(inputs_q, inputs_d_neg, mask_q, mask_d_neg)
            label = torch.ones(outputs_pos.size())#[1,1,1,1...]
            if opt.cuda:
                label = label.cuda()
            batch_loss = crit(outputs_pos, outputs_neg, Variable(label, requires_grad=False))

            # backward
            batch_loss.backward()

            # update parameters
            optimizer.step()
            step += 1
            total_loss += batch_loss.data[0]
            #if(step>=14000 and opt.eval_step!= 200):
            #    opt.eval_step=200 # make it smaller 2000--->200
            # opt.eval_step= 10
            if opt.is_ensemble:
                if step > 60000:
                    break
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
                        model_name = '../chkpt/' + opt.save_model + '_step_{}.chkpt'.format(step)
                        torch.save(checkpoint, model_name)
                    elif opt.save_mode == 'best':
                        model_name = '../chkpt/' + opt.save_model + '.chkpt'
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
    parser.add_argument('-embed')
    parser.add_argument('-vocab_size', default=400001, type=int)
    parser.add_argument('-load_model',type=str,default=None)# saved model(chkpt) dir
    parser.add_argument('-task', choices=['KNRM', 'CKNRM', 'MAXPOOL', 'AVGPOOL', 'LSTM', 'BERT'])
    parser.add_argument('-eval_step', type=int, default=1000)
    parser.add_argument('-epoch', type=int, default=1)
    parser.add_argument('-d_word_vec', type=int, default=300)
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-save_model', default=None)
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-lr', type=float, default=0.001)
    parser.add_argument('-n_bins', type=int, default=21)
    parser.add_argument('-name', type=int, default=1)
    parser.add_argument('-is_ensemble', type=bool, default=False)

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.mu =  kernal_mus(opt.n_bins)
    opt.sigma = kernel_sigmas(opt.n_bins)
    opt.n_layers = 1
    print (opt)
    #with open(opt.task+".txt",'w') as logf:#log file
    #    logf.write(str(opt)+'\n')
    if opt.mode=='train':
        # ========= Preparing DataLoader =========#
        # data_dir='/data/disk1/private/zhangjuexiao/MSMARCOReranking/'
        # train_filename = data_dir+"marco_train_pair_small.pkl"
        #test_filename = data_dir+"marco_eval.pkl"
        # dev_filename = data_dir+"marco_dev.pkl"
        # train_data = pickle.load(open(train_filename, 'rb'))
        #test_data = pickle.load(open(test_filename, 'rb'))

        training_data = DataLoader(
            data=opt.train_data,
            batch_size=opt.batch_size,
            cuda=opt.cuda)

        validation_data = DataLoaderTest(
            data=opt.val_data,
            batch_size=opt.batch_size,
            test=True,
            cuda=opt.cuda)




    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model.to("cuda")

    learning_rate = 5e-5
    warmup_proportion = 0.1 
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10%% of training."
    num_train_epochs = 10

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    num_train_steps = len(train_data_set) * num_train_epochs
    optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if n not in no_decay], 'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if n in no_decay], 'weight_decay_rate': 0.0}
    ]
    
    optimizer = BertAdam(optimizer_grouped_parameters,
                lr=learning_rate,
                warmup=warmup_proportion,
                t_total=num_train_steps)








#logits = model(input_ids, token_type_ids, input_mask)















#print(logits)
