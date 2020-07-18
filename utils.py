from transformers import BertTokenizer, BertModel
import argparse
import IPython

model_name = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_name,do_lower_case=True)
class DataSample():
    def __init__(self, question_id, context, question, ans_text, ans_start, answerable, max_query_len=45):
        self.question_id = question_id
        self.context = context
        self.question = question
        self.ans_text = ans_text
        self.answerable = 1 if answerable else 0
        self.ans_start = ans_start
        self.max_query_len = max_query_len
        self.ans_end = None
        # If the ans is no longer in the context after truncating, the set self.discard to True.
        # To inform the user that this is a sample to be discarded.
        self.discard = False 

        self.process_query()
        self.process_context()

    def process_query(self):
        ### Pad or truncate self.question
        tokenized_query_len = len(tokenizer.tokenize(self.question))
        if tokenized_query_len < self.max_query_len:
            self.question += '[PAD]' * (self.max_query_len - tokenized_query_len)
        elif tokenized_query_len > self.max_query_len: 
            enc_sequence = tokenizer(self.question)['input_ids'][1:-1] ## slice the string to avoid [CLS] and [SEP]
            enc_sequence = enc_sequence[:self.max_query_len]
            # I first tried with self.question = tokenizer.decode(enc_sequence).replace(" ", "") 
            # However, this didn't work since the spaces don't affect the
            # tokenizer, but replacing the spaces with "" is going to make the
            # encoded_sequence not equal to max_context_len in some samples.
            self.question = tokenizer.decode(enc_sequence)
        assert(len(tokenizer.tokenize(self.question)) == self.max_query_len)

    def process_context(self):
        ### 1. Pad or truncate the self.context and self.question
        ### 2. Give DataSample a post-tokenized ans_start and ans_end
        max_context_len = 512 - 3 - self.max_query_len # 3 denotes: [CLS], [SEP], [SEP]
        print('max_context_len: {}'.format(max_context_len))
        tokenized_context_len = len(tokenizer.tokenize(self.context))
        if self.answerable == True:
            self.ans_start = len(tokenizer.tokenize(self.context[:self.ans_start]))+1
            self.ans_end = self.ans_start + len(tokenizer.tokenize(self.ans_text))
        else:
            self.ans_start = 0# -1
            self.ans_end = 0 #-1
        original_context = self.context
        if tokenized_context_len < max_context_len:
            """ Todo:
                1. Pad the context
                2. Update ans_start and ans_end 
            """
            self.context += '[PAD]' * (max_context_len - tokenized_context_len)
        elif tokenized_context_len > max_context_len:
            enc_sequence = tokenizer(self.context)['input_ids'][1:-1] ## slice the string to avoid [CLS] and [SEP]
            enc_sequence = enc_sequence[:max_context_len]
            self.context = tokenizer.decode(enc_sequence)
            '''if len(tokenizer.tokenize(self.context)) != max_context_len:
                IPython.embed()
            if len(tokenizer.tokenize(self.context)) != max_context_len:
                print('shit')
                print(len(tokenizer.tokenize(self.context)))
            if len(tokenizer.encode(self.context,self.question)) != 512:
                print('fuck')
                #print(len(tokenizer.tokenize(self.context))+len(tokenizer.tokenize(self.question)))
                print(len(tokenizer.encode(self.context,self.question)))
                print(len(tokenizer.tokenize(self.question)))
                print(len(tokenizer.tokenize(self.context))) '''
            enc_context = tokenizer(self.context)['input_ids'] # [CLS]...context...[SEP]
            if self.ans_end > len(enc_context)-1:
                self.discard = True

        assert(len(tokenizer.tokenize(self.context)) == max_context_len)
    
    def _print(self):
        print(f'context: {self.context}, question: {self.question}, ans_text: {self.ans_text}, answerable: {self.answerable}, ans_start: {self.ans_start}, ans_end: {self.ans_end}')
if __name__ == "__main__":
    # assert all context_token_len are the identical.
    # assert all query_token_len are the identical.
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--bitch', action='store_true', default=False)
    args = parser.parse_args()
    if args.bitch:
        print('goooood')
    model_name = "bert-base-chinese"
    tokenizer = BertTokenizer.from_pretrained(model_name,do_lower_case=True)
    print(tokenizer.tokenize('幹 三 小 哈 哈 哈'))
    enc_sequence = tokenizer('幹三小哈哈哈')['input_ids'][:] ## slice the string to avoid [CLS] and [SEP]
    print(enc_sequence)
    print(tokenizer.decode(enc_sequence).replace(' ',''))
