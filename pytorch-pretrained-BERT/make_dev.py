from pytorch_pretrained_bert import BertTokenizer


tokenizer = BertTokenizer('/work/ececis_research/Manning/uncased_L-12_H-768_A-12/vocab.txt')


def trans(txt):
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(txt))

def make_data(line):
    #for line in data:
    #line = line.strip('\n').split('\t')
    qury, docp = trans(line[2]), trans(line[3])#, trans(line[2])
    return line[0] + '\t' + line[1] + '\t' + ','.join(str(x) for x in qury) + '\t' + ','.join(str(x) for x in docp) + '\n'




if __name__ == '__main__':
    with open("/work/ececis_research/Manning/top1000.dev.tsv") as file:
        data = file.readlines()

    with open("/work/ececis_research/Manning/bert_dev_axm.txt", "w") as file:
        for line in data:
            #print(line)
            line = line.strip('\n').split('\t')
            #print(line)
            if len(line) < 4:
                print(line)
                continue
            output = make_data(line)
            file.write(output)

