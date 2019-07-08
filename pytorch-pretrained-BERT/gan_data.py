from pytorch_pretrained_bert import BertTokenizer


tokenizer = BertTokenizer('/work/ececis_research/Manning/uncased_L-12_H-768_A-12/vocab.txt')


def trans(txt):
    #print(txt)
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(txt))

def make_data(line):
    #for line in data:
    #line = line.strip('\n').split('\t')
    qury, docp, docn = trans(line[0]), trans(line[1]), trans(line[2])
    qury = qury[:20]
    docp = docp[:20]
    docn = docn[:20]
    return ' '.join(str(x) for x in qury) + '\n' + ' '.join(str(x) for x in docp) + '\n' + ' '.join(str(x) for x in docn) + '\n'




if __name__ == '__main__':
    with open("/work/ececis_research/Manning/bert_train.txt") as file:
        print('start reading')
        data = file.readlines()
    print('end reading')

    with open("/work/ececis_research/Manning/gan_train.txt", "w") as file:
        print('start')
        for line in data:

            line = line.strip('\n').split('\t')
            if len(line) < 3:
                continue
            output = make_data(line)
            file.write(output)
        print('end')

