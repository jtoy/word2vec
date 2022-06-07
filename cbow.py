import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import itertools,re
#import nltk
#nltk.download('brown')
#from nltk.corpus import brown
corpus = [
    'he is a nice king',
    'she is a cool queen',
    'he is a big man',
    'she is a big woman',
    'warsaw is an awesome poland capital',
    'berlin is tight germany capital',
    'paris is france capital for real',
]

CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right

def CBOW(raw_text, window_size=2):
    data = []
    for i in range(window_size, len(raw_text)-window_size):
        context = [raw_text[i - window_size], raw_text[i - (window_size - 1)], raw_text[i + (window_size - 1)], raw_text[i + window_size]]
        #this only works for window size == 2?
        target = raw_text[i]
        data.append((context, target))
    return data

"""
brownt = ""
for cat in ['news']:
    for text_id in brown.fileids(cat):
        raw_text = list(itertools.chain.from_iterable(brown.sents(text_id)))
        #print(raw_text)
        #print("______")
        text = ' '.join(raw_text)
        text = text.lower()
        text.replace('\n', ' ')
        #text = re.sub('[^a-z ]+', '', text) + " . "
        brownt += text
        #brownt.append([w for w in text.split() if w != ''])
brownt = brownt.split()
"""

def tokenize_corpus(corpus):
    tokens = [x.split() for x in corpus]
    return tokens


test_text = """Please be a vegan.  Even if all you do is smile and say 'hello'. I've never forgotten that lesson. I also learned her name was Dorothy.""".split()
#print(brownt)

tokenized_corpus = tokenize_corpus(test_text)
cbow = CBOW(test_text)

print(tokenized_corpus)
vocabulary = []
for sentence in tokenized_corpus:
    for token in sentence:
        if token not in vocabulary:
            #print("token: ",token)
            vocabulary.append(token)

word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}
idx2word = {idx: w for (idx, w) in enumerate(vocabulary)}

vocabulary_size = len(vocabulary)
print("vocab size: ", vocabulary_size)


window_size = 2
idx_pairs = []
# for each sentence
for sentence in tokenized_corpus:
    indices = [word2idx[word] for word in sentence]
    # for each word, threated as center word
    for center_word_pos in range(len(indices)):
        # for each window position
        for w in range(-window_size, window_size + 1):
            context_word_pos = center_word_pos + w
            # make soure not jump out sentence
            if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
                continue
            context_word_idx = indices[context_word_pos]
            idx_pairs.append((indices[center_word_pos], context_word_idx))

idx_pairs = np.array(idx_pairs) # it will be useful to have this as numpy array
print(idx_pairs)

  
class CBOW_Model(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW_Model, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
        self.activation_function = nn.ReLU()

    def forward(self, inputs):
        out = self.embeddings(inputs).view((1,-1))
        out = self.linear(out) # nonlinear + projection
        #out = F.log_softmax(out, dim=1) # softmax compute log probability
        #embeds = sum(self.embeddings(inputs)).view(1,-1)
        #print("input shape should be 1,4", inputs)
        #out = self.embeddings(inputs) 
        #print("after embedding:", out.shape)
        #print("view should change to 1,1200 ", out.view(1,-1).shape)
        #print("shape wtf", embeds.shape)

        #out = self.embeddings(inputs) #.view(1,-1)
        #print("other shape", out)
        #print("other shape", out.shape)
        #out = out.mean(axis=1)
        #print("other other shape", out.shape)
        #out = self.embeddings(inputs).view((1, -1))
        #out = F.relu(self.linear1(embeds))

        #out = self.activation_function(out)
        out = F.log_softmax(out, dim=1)
        #out = self.linear2(out)
        return out
    def word2vec(self, input ):
        embeds = sum(self.embeddings(inputs)).view(1,-1)
        out = self.linear(embeds)


class SkipGram_Model(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram_Model, self).__init__()
        self.embeddings = nn.Embedding( num_embeddings = vocab_size, embedding_dim = embedding_dim)
        self.linear = nn.Linear( in_features=embedding_dim, out_features=vocab_size,)

    def forward(self, inputs):
        x = self.embeddings(inputs)
        x = self.linear(x)
        x = F.log_softmax(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

criterion = nn.CrossEntropyLoss() 
EMBEDDING_DIM = 300
model = CBOW_Model(len(vocabulary), EMBEDDING_DIM)
#optimizer = optim.SGD(model.parameters(), lr=0.001)
optimizer = optim.Adam(model.parameters())
#print(tokenized_corpus)
print(cbow)
losses = []
epochs = 550
for epoch in range(epochs):
    total_loss = 0
    print("epoch:", epoch)
    i = 0
    for context, target in cbow:
        i += 1
        #print(context)
        #print(target)
        if i % 1000 == 0:
            print("epoch: ",epoch, " pair: ", i)
        #print(context)
        context_idxs = torch.tensor([word2idx[w] for w in context], dtype=torch.long)
        model.zero_grad()
        log_probs = model(context_idxs)
        loss = criterion(log_probs, torch.tensor([word2idx[target]], dtype=torch.long))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(total_loss)

    losses.append(total_loss)
torch.save(model.state_dict(), "cbow.model")
print(losses)
print("done training")
#print(model.predict(['i','love'])

embeddings = model.embeddings
def get_closest(target_word,  n=5):
    print(word2idx)
    idx = word2idx[target_word]
    word_embedding = model.embeddings[idx]
    distances = []
    for word, index in word_to_idx.items():
        if word == target_word:
            continue
        distances.append((word, torch.dist(word_embedding, embeddings[index])))
    
    results = sorted(distances, key=lambda x: x[1])[1:n+2]
    return results
print(get_closest("name"))