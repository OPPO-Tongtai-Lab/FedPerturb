# add the implementation by OPPO
import torch.nn as nn
from torch.autograd import Variable

# from models.simple import SimpleNet
import torch

extracted_grads = []
def extract_grad_hook(module, grad_in, grad_out):
    # print(grad_out)
    extracted_grads.append(grad_out[0])

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, name=None, rnn_type="LSTM", ntoken=50000, ninp=200, nhid=200, nlayers=2, dropout=0.2, tie_weights=True, binary=False):
        super(RNNModel, self).__init__()
        if binary:
            self.encoder = nn.Embedding(ntoken, ninp)
            # self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=0.5)
            self.lstm = nn.LSTM(ninp, nhid, nlayers, dropout=0.5, batch_first=True)
            self.drop = nn.Dropout(dropout)
            self.decoder = nn.Linear(nhid, 1)
            self.sig = nn.Sigmoid()
        else:
            self.drop = nn.Dropout(dropout)
            self.encoder = nn.Embedding(ntoken, ninp)

            # self.encoder.register_backward_hook(extract_grad_hook)

            if rnn_type in ['LSTM', 'GRU']:
                self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
            else:
                try:
                    nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
                except KeyError:
                    raise ValueError( """An invalid option for `--model` was supplied,
                                    options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
                self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
            
            self.decoder = nn.Linear(nhid, ntoken)

            # Optionally tie weights as in:
            # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
            # https://arxiv.org/abs/1608.05859
            # and
            # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
            # https://arxiv.org/abs/1611.01462

            if tie_weights:
                if nhid != ninp:
                    raise ValueError('When using the tied flag, nhid must be equal to emsize')
                self.decoder.weight = self.encoder.weight

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.binary = binary

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def return_embedding_matrix(self):
        return self.encoder.weight.data

    def embedding_t(self,input):
        input = input.type(torch.LongTensor)
        input = input.cuda()
        # emb = self.drop(self.encoder(input))
        emb = self.encoder(input)
        return emb

    def forward(self, input, hidden, latern=False, emb=None):
        # input = input.type(torch.LongTensor)
        # input = input.cuda()
        # emb = self.embedding_t(input)
        if self.binary:
            batch_size = input.size(0)
            emb = self.encoder(input)
            output, hidden = self.lstm(emb, hidden)
            output = output.contiguous().view(-1, self.nhid)
            out = self.drop(output)
            out = self.decoder(out)
            sig_out = self.sig(out)
            sig_out = sig_out.view(batch_size, -1)
            sig_out = sig_out[:, -1]
            return sig_out, hidden

        else: 
            if emb is None:
                emb = self.drop(self.encoder(input))
        
            output, hidden = self.rnn(emb, hidden)
            output = self.drop(output)

            #### use output = self.drop(output) as output features
            decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
            if latern:
                return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden, emb
            else:
                return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden


    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())

def LSTM(name=None, created_time=None):
    return RNNModel(name = name)