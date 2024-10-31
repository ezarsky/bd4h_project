import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

####################
##### Word2Vec #####
####################

# import summaries into list
path = "../data/sf_stories.txt"
summaries = []
with open(path, encoding="utf-8") as f:
    summaries = f.readlines()

# preprocessing

# strip whitespace before and after summary and convert to lowercase
summaries = [summary.strip().lower() for summary in summaries if len(summary) > 1]
summaries = ["".join([char for char in summary if (char.isalnum() or char.isspace())]) for summary in summaries]
    

# get full text file from which to compile vocab
full_text = " ".join(summaries)

"""
# remove punctuation and special characters except spaces
all_chars = set(full_text)
endings = {".", "?", "!"}
others = {"'",  " "}
exceptions = endings.union(others)
specials = [char for char in all_chars if (char not in exceptions and not char.isalnum())]
for special in specials:
    full_text = full_text.replace(special, "")
    summaries = [summary.replace(special, "") for summary in summaries]


# get list of all words in all summaries
sentences = []
per_split = full_text.split(".")
for per_phrase in per_split:
    qst_split = per_phrase.split("?")
    for qst_phrase in qst_split:
        exc_split = qst_phrase.split("!")
        sentences.extend(exc_split)

sentences = [sentence.strip() for sentence in sentences if len(sentence)>0]

all_text = " ".join(sentences)
all_words = set(all_text.split())
"""

# get word frequencies for each word
word_counts = dict()
for word in full_text.split():
    word_counts[word] = word_counts.get(word, 0) + 1

 
vocab = set(word_counts.keys())
"""
#filter for only words with freq >= 5
vocab = set()
for word in word_counts.keys():
    if word_counts[word] >= 5:
        vocab.add(word)
"""
vocab = list(sorted(vocab))

word_to_idx = {word: i for i, word in enumerate(vocab)}

### Continuous Bag of Words (CBOW) ###
window_size = 5
embed_dim = 10
epochs = 15

# TODO: tokenize the input
# TODO: rolling version for more efficient capture
#TODO: frequency subsampling  
#TODO: negative sampling

# get context-target pairs from summaries
contar_pairs = []
for summary in summaries:
    summary_list = summary.split()
    N = len(summary_list)
    for i in range(N):
        target = summary_list[i]
        pre_context = [summary_list[i - j] for j in range(window_size, 0, -1) if (i-j >= 0)]
        post_context = [summary_list[i + j] for j in range(1, window_size+1) if (i+j < N)]
        context = pre_context + post_context
        contar_pairs.append((context, target))

dataset_size = len(contar_pairs)

# define CBOW model
class CBOWModule(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOWModule, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, inputs):
        embeddings = self.embed(inputs)
        embed_sum = embeddings.sum(dim=0).reshape((1, -1))
        vocab_proj = self.fc(embed_sum)
        log_probs = F.log_softmax(vocab_proj, dim=1)
        return log_probs


# model training setup
losses = []
loss_func = nn.NLLLoss()
model = CBOWModule(len(vocab), embed_dim)
optimizer = optim.SGD(model.parameters(), lr=0.001)


# training loop
for epoch in range(epochs):
    print(f"Epoch: {epoch+1}")
    total_loss = 0
    for i, (context, target) in enumerate(contar_pairs):
        if (i % 1000 == 0):
            print(f"{i}/{dataset_size} data points processed")
        c_id_list = [word_to_idx[word] for word in context]
        t_id_list = [word_to_idx[target]]
        c_id_tensor = torch.tensor(c_id_list, dtype=torch.long)
        t_id_tensor = torch.tensor(t_id_list, dtype=torch.long)

        model.zero_grad()
        
        log_probs = model(c_id_tensor)
        loss = loss_func(log_probs, t_id_tensor)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    print(total_loss)
    losses.append(total_loss)

fig, ax = plt.subplots()
ax.plot(np.arange(epochs), np.array(losses))
plt.show()

        
####################
####################
####################        