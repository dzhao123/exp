import random


def generate_ooq(qtxt_list):
    while True:
        word = str(random.randint(1,315369))
        if word not in qtxt_list:
            return word
        else:
            continue

#def generate_position(doc_list):
#    posit = random.randint(0,len(doc_list)-1)
#    return posit

def replace(wiq, ooq, doc_list):
    return [ooq if ele == wiq else ele for ele in doc_list]



def TFC1(text):
    qtxt, pos_doc, neg_doc = text.strip().split('\t')

    wiq = random.choice(qtxt.split(','))
    ooq = generate_ooq(qtxt.split(','))

    #pos_doc_position = generate_position(pos_doc.split(','))
    #neg_doc_position = generate_position(neg_doc.split(','))

    static_pos_doc = pos_doc[:]
    static_neg_doc = neg_doc[:]
    origin_pos_doc = pos_doc[:]
    origin_neg_doc = neg_doc[:]


    pos_axm_doc_list = origin_pos_doc.split(',')
    neg_axm_doc_list = origin_neg_doc.split(',')
    pos_adj_doc_list = pos_doc.split(',')
    neg_adj_doc_list = neg_doc.split(',')


    pos_adj_doc_list = replace(wiq, ooq, pos_adj_doc_list)#.insert(pos_doc_position, wiq)
    #pos_adj_doc_list#.insert(pos_doc_position, ooq)
    neg_adj_doc_list = replace(wiq, ooq, neg_adj_doc_list)#.insert(neg_doc_position, wiq)
    #neg_adj_doc_list#.insert(neg_doc_position, ooq)

    text = qtxt + '\t' + static_pos_doc + '\t' + static_neg_doc + '\t' + ','.join(pos_axm_doc_list) + '\t' +  ','.join(pos_adj_doc_list) + '\t' + ','.join(neg_axm_doc_list) + '\t' + ','.join(neg_adj_doc_list) + '\n'
    return text


if __name__ == '__main__':
    with open('/work/ececis_research/Manning/triples.train.small.tsv') as file:
        data = file.readlines()

    with open('/work/ececis_research/Manning/train_TFC1_bert_R.txt','w') as tfile:
        for index, row in enumerate(data):
            line = TFC1(row)
            if line is None:
                continue
            #print(index)
            #print(TFC1(row))
            tfile.write(line)

