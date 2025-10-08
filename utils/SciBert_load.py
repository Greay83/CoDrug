import numpy as np
import torch

import numpy as np
import torch


# This is for BERT
def padarray(A, size, value=0):
    t = size - len(A)
    return np.pad(A, pad_width=(0, t), mode='constant', constant_values=value)


# This is for BERT
def preprocess_each_sentence(sentence, tokenizer, max_seq_len):
    text_input = tokenizer(
        sentence, truncation=True, max_length=max_seq_len,
        padding='max_length', return_tensors='pt')

    input_ids = text_input['input_ids'].squeeze()
    attention_mask = text_input['attention_mask'].squeeze()

    sentence_tokens_ids = padarray(input_ids, max_seq_len)
    sentence_masks = padarray(attention_mask, max_seq_len)
    return [sentence_tokens_ids, sentence_masks]


# This is for BERT
def prepare_text_tokens(description, tokenizer, max_seq_len):
    B = len(description)
    tokens_outputs = [preprocess_each_sentence(description[idx], tokenizer, max_seq_len) for idx in range(B)]
    tokens_ids = [o[0] for o in tokens_outputs]
    masks = [o[1] for o in tokens_outputs]
    tokens_ids = torch.Tensor(np.array(tokens_ids)).long()
    masks = torch.Tensor(np.array(masks)).bool()
    return tokens_ids, masks

if __name__=="__main__":
    text="I am a big big man"
    from transformers import BertTokenizer, BertModel

    description_tokenizer = BertTokenizer.from_pretrained('../scibert/allenai/scibert_scivocab_uncased/vocab.txt')
    description_model = BertModel.from_pretrained('../scibert/allenai/scibert_scivocab_uncased')

    description_tokens_ids, description_masks = prepare_text_tokens(
        device='cuda:0', description=text, tokenizer=description_tokenizer, max_seq_len=512)
    description_output = description_model(input_ids=description_tokens_ids, attention_mask=description_masks)
    description_repr = description_output["pooler_output"]
    print(description_repr)
    print(description_repr.shape)
