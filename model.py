import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
class QAModel(nn.Module):
    def __init__(self, hidden_size=768):
        super(QAModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.linear = nn.Linear(hidden_size, 2)
    def forward(self, inputs):
        last_hidden_states, _ = self.bert(**inputs) ## output[0] = last_hidden_states, shape (batch_size, seq_len, hidden_size)
        output = self.linear(last_hidden_states) # output: shape (batch_size, seq_len, 2)
        if torch.cuda.is_available():
            output = output.cuda()
        return output
    def infer(self, inputs):
        pass
