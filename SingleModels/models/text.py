import torch
from torch import nn
from transformers import BertModel
import torch.nn as nn



# courtesy of https://github.com/FernandoLpz/Text-Classification-LSTMs-PyTorch/blob/master/src/model.py
class LSTMClassifier(nn.ModuleList):

	def __init__(self , glove_vec , args):
		super(LSTMClassifier, self).__init__()
		self.output_dim = args['output_dim']
		try:
			self.hidden_dim = args['hidden_layers'][0] # args.hidden_dim
		except:
			self.hidden_dim = args['hidden_layers'] # args.hidden_dim	
		self.LSTM_layers = args['lstm_layers'] # args.LSTM_layers
		# self.input_size = 128 # embedding dimention args.maxwords
		self.dropout = nn.Dropout(0.5)
		# self.embedding = nn.Embedding( self.input_size, self.hidden_dim)
		self.embedding = nn.Embedding.from_pretrained(glove_vec.vectors)
		# pdb.set_trace()
		self.lstm = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=self.LSTM_layers, batch_first=True)
		# self.fc1 = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
		self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)
		self.sigmoid = torch.nn.LogSigmoid()

	def forward(self, x):
		# pdb.set_trace()
		out = self.embedding(x)  # embedding will give us (batch_size, n_features , hidden_dim_size)
		out, (hidden, cell) = self.lstm(out)
		# out = self.dropout(self.fc1(out))
		out = self.fc2(out) # now (batch_size, n_features , output_dim size)
		out = self.dropout(out)
		out = torch.mean(out, dim = 1)
		out = self.sigmoid(out) 

		return out

class BertClassifier(nn.Module):

	def __init__(self, args ,dropout=0.5):
		super(BertClassifier, self).__init__()

		self.output_dim = args['output_dim']
		
		self.bert = BertModel.from_pretrained('bert-base-cased')

		self.dropout = nn.Dropout(dropout)

		self.linear = nn.Linear(768, 300)

		self.linear1 = nn.Linear(300, self.output_dim)

		self.sigmoid = nn.Sigmoid()

	def forward(self, input_id, mask):


		# print(f"input_id = {input_id}\n input_id.shape = {input_id.shape}")
		
		_, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
		# print(f"pooled_output = {pooled_output}\n pooled_output.shape = {pooled_output.shape}")
		
		dropout_output = self.dropout(pooled_output)
		# print(f"dropout_output = {dropout_output}\n dropout_output.shape = {dropout_output.shape}")
		
		linear_output = self.linear(dropout_output)
		# print(f"linear_output = {linear_output}\n linear_output.shape = {linear_output.shape}")
		
		final_layer = self.sigmoid(linear_output)

		x = self.dropout(final_layer)

		x = self.linear1(x)

		return x

	# make sure all the params are stored in a massive matrix which will end up being 
	# a complicated hell to make sure we get the params on every model type 

	# multi modal images 