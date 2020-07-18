from dataset import QADataset
from model import QAModel
from prepro import prepro
import IPython as nb
import json 
import sys
import pickle
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from transformers import BertTokenizer, BertModel 
from utils import DataSample
from tqdm.auto import tqdm




def load_data_list(PATH):
    data_list = None
    with open(PATH, 'rb') as f:
        data_list = pickle.load(f)
    return data_list
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)
def main():
    torch.manual_seed(94)
    batch_size=4
    train_data_list = load_data_list('./train_data_list')
    train_set = QADataset(train_data_list)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    dev_data_list = load_data_list('./dev_data_list')
    dev_set = QADataset(dev_data_list)
    dev_loader = DataLoader(dev_set, batch_size=batch_size, shuffle=True)
    
    model = QAModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    EPOCHS = 6
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=5e-5)
    loss = 0
    for epoch in range(EPOCHS):
        cum_loss = 0
        print(f'epoch: {epoch}')
        ite = 0
        model.train()
        for i, (question_id, context, question, answerable, ans_start, ans_end) in enumerate(tqdm(train_loader)):
            ite = i+1
            pt = tokenizer(context, question, return_tensors='pt')
            
            bs = len(context)
            mask = torch.zeros(bs, 512).bool()
            mask[:,466:] = True
            
            if torch.cuda.is_available():
                answerable = answerable.cuda()
                ans_start = ans_start.cuda()
                ans_end =  ans_end.cuda()
                pt['input_ids'] = pt['input_ids'].cuda()
                pt['token_type_ids'] = pt['token_type_ids'].cuda()
                pt['attention_mask'] = pt['attention_mask'].cuda()
                mask = mask.cuda()
            target = torch.cat((ans_start.unsqueeze(1), ans_end.unsqueeze(1)), dim=1).to(device)
            optimizer.zero_grad()
            output = model(pt)
            #print('output: {}'.format(output.shape))
            pred_start = output[:, :, 0]
            pred_end = output[:, :, 1]
            output[:, :, 0].masked_fill_(mask, float('-inf')) 
            output[:, :, 1].masked_fill_(mask, float('-inf')) 
            #print('start: {}'.format(pred_start.shape))
            #print('end: {}'.format(pred_end.shape))
            #start_loss = loss_fn(pred_start, ans_start)
            #end_loss = loss_fn(pred_end, ans_end)
            #nb.embed()
            loss = loss_fn(output, target)
            #print(loss)
            #print(loss)
            #print('loss1 : {}, loss2: {}, loss3: {}'.format(loss1,loss2,loss3))
            cum_loss+=float(loss)
            #print(loss)
            loss.backward()
            optimizer.step()
        print(cum_loss)

        model.eval()
        dev_loss = 0
        dev_ite = 0
        for i, (question_id, context, question, answerable, ans_start, ans_end) in enumerate(tqdm(dev_loader)):
            bs = len(context)
            mask = torch.zeros(bs, 512).bool()
            mask[:,466:] = 1

            with torch.no_grad():
                dev_ite = i+1
                pt = tokenizer(context, question, return_tensors='pt')
                if torch.cuda.is_available():
                    answerable = answerable.cuda()
                    ans_start = ans_start.cuda()
                    ans_end =  ans_end.cuda()
                    pt['input_ids'] = pt['input_ids'].cuda()
                    pt['token_type_ids'] = pt['token_type_ids'].cuda()
                    pt['attention_mask'] = pt['attention_mask'].cuda()
                    mask = mask.cuda()
                target = torch.cat((ans_start.unsqueeze(1), ans_end.unsqueeze(1)), dim=1).to(device)
                output = model(pt)
                output[:, :, 0].masked_fill_(mask, float('-inf')) 
                output[:, :, 1].masked_fill_(mask, float('-inf')) 
                loss = loss_fn(output, target)
                dev_loss+=float(loss)
        print('avg_train_loss: {}, avg_dev_loss: {}'.format(cum_loss/ite, dev_loss/dev_ite))
        SAVED_MDL_PATH = './model/'+str(epoch+1)+'.pt'
        #torch.save(model.state_dict(), SAVED_MDL_PATH)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
            }, SAVED_MDL_PATH)
        print('model {} saved'.format(SAVED_MDL_PATH))

def predict(MDL_PATH, DATA_PATH):
    batch_size=4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #start_candidates, end_candidates = [], []
    model = QAModel()
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=3e-5)

    checkpoint = torch.load(MDL_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    model.eval()
    
    dev_data_list = load_data_list(DATA_PATH)
    dev_set = QADataset(dev_data_list)
    dev_loader = DataLoader(dev_set, batch_size=batch_size, shuffle=False)
    print('run prediction')
    dic = {}
    for i, (question_id, context, question, answerable, ans_start, ans_end) in enumerate(tqdm(dev_loader)):
        bs = len(context)
        mask = torch.zeros(bs, 512).bool()
        mask[:,466:] = 1

        with torch.no_grad():
            dev_ite = i+1
            pt = tokenizer(context, question, return_tensors='pt')
            if torch.cuda.is_available():
                answerable = answerable.cuda()
                ans_start = ans_start.cuda()
                ans_end =  ans_end.cuda()
                pt['input_ids'] = pt['input_ids'].cuda()
                pt['token_type_ids'] = pt['token_type_ids'].cuda()
                pt['attention_mask'] = pt['attention_mask'].cuda()
                mask = mask.cuda()
            target = torch.cat((ans_start.unsqueeze(1), ans_end.unsqueeze(1)), dim=1).to(device)
            output = model(pt) # shape (batch_size, 512, 2)
            output[:, :, 0].masked_fill_(mask, float('-inf')) 
            output[:, :, 1].masked_fill_(mask, float('-inf')) 
            
            for batch_idx, sample in enumerate(output): # sample: shape (512, 2)
                start = sample[:,0] # start: shape (512)
                end = sample[:,1]
                start_candidates = torch.topk(start, k=30)
                end_candidates = torch.topk(end, k=30)
                ans_candidates = []
                scores = []
                for i, s in enumerate(start_candidates[1]):
                    for j, e in enumerate(end_candidates[1]):
                        if e == s and e == 0:
                            ans_candidates.append((s, e))
                            scores.append(start_candidates[0][i]+end_candidates[0][j])
                        if s<e and e-s <= 30:
                            ans_candidates.append((s, e))
                            scores.append(start_candidates[0][i]+end_candidates[0][j])
                results = list(zip(scores, ans_candidates))          
                results.sort()
                results.reverse()

                if results[0][1][0] == 0:
                    dic[question_id[batch_idx]] = ""
                else:
                    s, e = results[0][1][0], results[0][1][1]
                    ids = pt['input_ids'][batch_idx][s:e]
                    #print(tokenizer.decode(ids).replace(" ", ""))
                    dic[question_id[batch_idx]] = tokenizer.decode(ids).replace(" ", "")

    with open('prediction.json', 'w') as fp:
        json.dump(dic, fp)


if __name__ == '__main__':
    #main()
    predict('./model/6.pt', 'dev_data_list')



        
