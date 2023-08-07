import torch
torch.rand((2,3))
torch.arange(6).view(2,3)
# pip install --user -U nltk
from nltk.tokenize import TweetTokenizer
tweet=u"Snow White and the Seven Degrees #MakeAMovieCold@midnight:-)"
tokenizer = TweetTokenizer()
print(tokenizer.tokenize(tweet))

# pip install -U spacy
# import spacy
# nlp = spacy.load("en")
# doc = nlp(tweet)
# for token in doc:
#     print('{} --> {}'.format(token, token.lemma_))
#
# x = torch.randn(1, 3, 1, 2)
# y = x.squeeze(dim=0).squeeze(dim=1)
# print(x.shape)  # torch.Size([1, 3, 1, 2])
# print(y.shape)  # torch.Size([3, 2])

import matplotlib.pyplot as plt

x = torch.arange(1.0, 3.0, 0.1)
y = torch.sigmoid(x)
import torch
import torch.nn as nn

x = torch.arange(-5.0, 5.0, 0.1)
y = torch.relu(x)
import matplotlib.pyplot as plt
plt.plot(x.numpy(), y.numpy())
plt.show()
prelu = torch.nn.PReLU(num_parameters=1)
y = prelu(x)
plt.plot(x.numpy(), y.numpy())
plt.show()

