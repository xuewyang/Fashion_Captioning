import pdb
import json
import os
from transformers import BertTokenizer

data_folder = '/home/xuewyang/Xuewen/Research/data/FACAD/jsons'
split = 'VAL'


with open(os.path.join(data_folder, split + '_CAPTIONS_5' + '.json'), 'r') as j:
    captions = json.load(j)

# Load caption lengths (completely into memory)
with open(os.path.join(data_folder, split + '_CAPLENS_5' + '.json'), 'r') as j:
    caplens = json.load(j)

with open(os.path.join(data_folder, 'WORDMAP_5.json'), 'r') as f:
    word_map = json.load(f)

dictionary = {x: y for y, x in word_map.items()}
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')


# tokens = self.tokenizer.tokenize(headline)
#
# # Add [CLS] at the beginning and [SEP] at the end of the tokens list for classification problems
# tokens = [CLS_TOKEN] + tokens + [SEP_TOKEN]
# # Convert tokens to respective IDs from the vocabulary
# input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
#
# # Segment ID for a single sequence in case of classification is 0.
# segment_ids = [0] * len(input_ids)
#
# # Input mask where each valid token has mask = 1 and padding has mask = 0
# input_mask = [1] * len(input_ids)
#
# # padding_length is calculated to reach max_seq_length
# padding_length = MAX_SEQ_LENGTH - len(input_ids)
# input_ids = input_ids + [0] * padding_length
# input_mask = input_mask + [0] * padding_length
# segment_ids = segment_ids + [0] * padding_length
#
# assert len(input_ids) == MAX_SEQ_LENGTH
# assert len(input_mask) == MAX_SEQ_LENGTH
# assert len(segment_ids) == MAX_SEQ_LENGTH
#
# return torch.tensor(input_ids, dtype=torch.long, device=DEVICE), \
#        torch.tensor(segment_ids, dtype=torch.long, device=DEVICE), \
#        torch.tensor(input_mask, device=DEVICE), \
#        torch.tensor(is_sarcastic, dtype=torch.long, device=DEVICE)

# encoded_dict = tokenizer.encode_plus(
#                         sent,                      # Sentence to encode.
#                         add_special_tokens = True, # Add '[CLS]' and '[SEP]'
#                         max_length = 64,           # Pad & truncate all sentences.
#                         pad_to_max_length = True,
#                         return_attention_mask = True,   # Construct attn. masks.
#                         return_tensors = 'pt',     # Return pytorch tensors.
#                    )

def tokens_to_tokens(caption, dictionary, tokenizer):
    """
    Transform tokens to a sentence and then the sentence can be used on bert tokenizer
    :param tokens: tokens from dataset
    :return: tokens from bert tokenizer
    """
    img_caption = [dictionary[w] for w in caption if w not in [0, 15804, 15805, 15806]]
    sent = ' '.join(img_caption)
    tokens = tokenizer.encode(sent, add_special_tokens=True)
    attention_mask = [1] * len(tokens)
    padding_length = 50 - len(tokens)
    input_ids = tokens + [0] * padding_length
    attention_mask = attention_mask + [0] * padding_length

    encoded_dict = tokenizer.encode_plus(sent, add_special_tokens=True, max_length=50, pad_to_max_length=True,
                                         return_attention_mask=True)

    pdb.set_trace()
    return encoded_dict['input_ids'], attention_mask


tokens_to_tokens(captions[0], dictionary, tokenizer)
