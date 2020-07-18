import json 
import sys
import pickle
import argparse
from transformers import BertTokenizer, BertModel 
from utils import DataSample
from tqdm import tqdm

def prepro(json_path):
    from transformers import BertTokenizer, BertModel 
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)
    """ 
    @returns data_list: A list contains all the training sample, each entry is a tuple (context, question, answer)
    """
    train_json = json.load(open(json_path, 'r'))
    data_list = []
    for cur_data in train_json['data']:
        for tmp in cur_data['paragraphs']: # tmp: a list of {context:str, qas:[{q:str, a:str}, ...]}
            context = tmp['context']
            for qa in tmp['qas']:
                question = qa['question']
                question_id = qa['id']
                answerable = qa['answerable']
                for ans in qa['answers']:
                    ans_text = ans['text']
                    ans_start = ans['answer_start']
                    data_sample = DataSample(question_id, context, question, ans_text, ans_start, answerable)
                    if data_sample.discard == False:
                        data_list.append(data_sample)
                        print(len(data_list))
                                        
    '''data_cnt = 0
    for cur_data in train_json['data']:
        for tmp in cur_data['paragraphs']:
            data_cnt += len(tmp['qas'])
    assert(data_cnt == len(data_list))'''
    #print(f'data_cnt {data_cnt}; {len(data_list)}')
    return data_list

def truncate(data_list):
    max_q_len,cnt = -1, 0
    sum_ = 0
    l = []
    for sample in data_list:
        sum_ += len(sample.question)
        l.append(len(sample.question))
        if len(sample.context) > 512-118-3: cnt+=1
        if len(sample.context)+len(sample.ans_text) > max_q_len:
            max_q_len = len(sample.context)+len(sample.ans_text)
    print(sum_, len(data_list))
    print(sum_/len(data_list))
    import numpy as np
    a = np.array(l)
    print(np.sort(a)[-1000:])

def load_data_list(PATH):
    data_list = None
    with open(PATH, 'rb') as f:
        data_list = pickle.load(f)
    return data_list

def make_data_list(PATH, SAVE):
    train_data_list = prepro(PATH)
    with open(SAVE, 'wb') as f:
        pickle.dump(train_data_list, f)
    return train_data_list
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--MAKE', action='store_true', default=False)
    parser.add_argument('-f', dest="PATH", default='./train_data_list')
    parser.add_argument('-s', dest="SAVE", default='./train_data_list')
    args = parser.parse_args()
    
    if args.MAKE:
        train_data_list = make_data_list(args.PATH, args.SAVE)
    else:
        train_data_list = load_data_list(args.PATH)

    '''dev_data_list = prepro('dev.json')
    with open('./dev_data_list', 'wb') as f:
        pickle.dump(dev_data_list, f)

    test_data_list = prepro('test.json')
    with open('./test_data_list', 'wb') as f:
        pickle.dump(test_data_list, f)'''

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)
    
    ############################### Todo: Assert all input tokens has a length of 512 ###############################
    for train_data in tqdm(train_data_list):
        pt = tokenizer(train_data.context, train_data.question, return_tensors='pt')
        if len(pt['input_ids'][0]) != 512:
            print(len(pt['input_ids'][0]))
        if pt['input_ids'][0][465] != 102:
            print(pt['input_ids'][0][465])
    print('done checking')

    

    
