from pytorch_pretrained_bert import BertTokenizer


tokenizer = BertTokenizer('/work/ececis_research/Manning/uncased_L-12_H-768_A-12/vocab.txt')


def trans(txt):
    #print(txt)
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(txt))

def make_data(line):
    #for line in data:
    #line = line.strip('\n').split('\t')
    print(line[0])
    print(line[1])
    print(line[2])
    print(line[3])
    print(line[4])
    print(line[5])
    print(line[6])

    qury, docp, docn, docpp, docpn, docnp, docnn = trans(line[0]), trans(line[1]), trans(line[2]), trans(line[3]), trans(line[4]), trans(line[5]), trans(line[6])

    return ','.join(str(x) for x in qury) + '\t' + ','.join(str(x) for x in docp) + '\t' + ','.join(str(x) for x in docn) + '\t' + ','.join(str(x) for x in docpp) + '\t' + ','.join(str(x) for x in docpn) + '\t' + ','.join(str(x) for x in docnp) + '\t' + ','.join(str(x) for x in docnn) + '\n'




if __name__ == '__main__':
    with open("/work/ececis_research/Manning/train_TFC1_bert_A.txt") as file:
        data = file.readlines(10)
    
    with open("/work/ececis_research/Manning/b.txt", "w") as file:
        for line in data:

            line = line.strip('\n').split('\t')
            if len(line) < 3:
                continue
            output = make_data(line)
            file.write(output)
    
