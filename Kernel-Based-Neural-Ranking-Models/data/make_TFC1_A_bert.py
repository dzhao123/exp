import random


def generate_ooq(qtxt_list, index_vocab):
    while True:
        word = str(random.randint(1,851412))
        if index_vocab[word] not in qtxt_list:
            return index_vocab[word]
        else:
            continue

def generate_position(doc_list):
    posit = random.randint(0,len(doc_list)-1)
    return posit


def TFC1(text, vocab_index, index_vocab):
    qtxt, pos_doc, neg_doc = text.strip().split('\t')
    #print(qtxt)
    wiq = random.choice(qtxt.split(' '))
    ooq = generate_ooq(qtxt.split(' '), index_vocab)

    #print("wiq:", wiq)
    #print("ooq:", ooq)

    pos_doc_position = generate_position(pos_doc.split(' '))
    neg_doc_position = generate_position(neg_doc.split(' '))

    static_pos_doc = pos_doc[:]
    static_neg_doc = neg_doc[:]
    origin_pos_doc = pos_doc[:]
    origin_neg_doc = neg_doc[:]
   

    pos_axm_doc_list = origin_pos_doc.split(' ')
    neg_axm_doc_list = origin_neg_doc.split(' ')
    pos_adj_doc_list = pos_doc.split(' ')
    neg_adj_doc_list = neg_doc.split(' ')
    #print(pos_axm_doc_list)
    #print(neg_axm_doc_list)
    #print(pos_adj_doc_list)
    #print(neg_adj_doc_list)


    pos_axm_doc_list.insert(pos_doc_position, wiq)
    pos_adj_doc_list.insert(pos_doc_position, ooq)
    neg_axm_doc_list.insert(neg_doc_position, wiq)
    neg_adj_doc_list.insert(neg_doc_position, ooq)

    #print(pos_axm_doc_list)
    #print(neg_axm_doc_list)
    #print(pos_adj_doc_list)
    #print(neg_adj_doc_list)

    #print('qtxt:', qtxt)
    #print('static_pos_doc:', static_pos_doc)
    #print('static_neg_doc:', static_neg_doc)
    #print('axiom_pos_doc:', ','.join(pos_axm_doc_list))
    #print('axiom_neg_doc:', ','.join(neg_axm_doc_list))
    #print('adjst_pos_doc:', ','.join(pos_adj_doc_list))
    #print('adjst_neg_doc:', ','.join(neg_adj_doc_list)

    #if qtxt is None:
    #    return
    #if static_pos_doc is None:
    #    return
    #if static_neg_doc is None:
    #    return
    #if ','.join(pos_axm_doc_list) is None:
    #    return
    #if ','.join(neg_axm_doc_list) is None:
    #    return
    #if ','.join(pos_adj_doc_list) is None:
    #    return
    #if ','.join(neg_adj_doc_list) is None:
    #    return None


    text = qtxt + '\t' + static_pos_doc + '\t' + static_neg_doc + '\t' + ' '.join(pos_axm_doc_list) + '\t' +  ' '.join(pos_adj_doc_list) + '\t' + ' '.join(neg_axm_doc_list) + '\t' + ' '.join(neg_adj_doc_list) + '\n'
    #print("text:", text)
    return text
    

def vocabulize(vocab_data):
    vocab_index = {}
    index_vocab = {}
    for line in vocab_data:
        word, index = line.strip().split('\t')
        vocab_index[word] = index
        index_vocab[index] = word
    return vocab_index, index_vocab


if __name__ == '__main__':
    with open('/work/ececis_research/Manning/triples.train.small.tsv') as file:
        data = file.readlines()

    with open('/work/ececis_research/Manning/vocab_gen.tsv') as file:
        vocab_data = file.readlines()
    vocab_index, index_vocab = vocabulize(vocab_data)
    
    with open('/work/ececis_research/Manning/train_TFC1_bert_A.txt','w') as tfile:
        for index, row in enumerate(data):
            line = TFC1(row, vocab_index, index_vocab)
            if line is None:
                continue
            #print(index)
            #print(TFC1(row))
            tfile.write(line)


