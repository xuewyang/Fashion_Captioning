import torch.nn as nn
from transformers import BertForSequenceClassification


class Classifier(nn.Module):
    def __init__(self, n_classes):
        super(Classifier, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', return_dict=True,
                                                                  num_labels=n_classes)
        # for param in self.model.parameters():
        #     param.requires_grad = False

    def forward(self, input_ids, attention_mask, token_type_ids, cate):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                            labels=cate)
        return outputs
