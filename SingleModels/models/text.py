import torch
from torch import nn
from transformers import AutoModel
import torch.nn as nn


class BertClassifier(nn.Module):

	def __init__(self, args):
		super(BertClassifier, self).__init__()
		self.dropout = args['dropout']
		self.output_dim = args['output_dim']
		
		self.bert = AutoModel.from_pretrained('jkhan447/sarcasm-detection-RoBerta-base-CR')

		self.bert_norm = nn.LayerNorm(768)
		self.dropout = nn.Dropout(self.dropout)

		self.linear = nn.Linear(768, self.output_dim)


	def forward(self, input_ids, mask , check):


		_, text_outputs = self.bert(input_ids= input_ids, attention_mask=mask,return_dict=False)
  
		del _
		del mask
		del input_ids
  
		text_outputs = self.bert_norm(text_outputs)
  
		if check == "train":
			text_outputs = self.dropout(text_outputs)
		
		text_outputs = self.linear(text_outputs)
		
		return text_outputs

	# make sure all the params are stored in a massive matrix which will end up being 
	# a complicated hell to make sure we get the params on every model type 

	# multi modal images 