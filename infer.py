from multiprocessing.sharedctypes import RawArray
from preprocessing import process_data
from model import BiLSTM_Attention_model

model = None


# embedding path
Word2vec_path = "embeddings"

char_lowercase = True
# dataset for train, validate and test
vocab = "dataset/Encoded_data/vocab.json"
word_embedding = "dataset/Encoded_data/word_emb.npz"
# network parameters
num_units = 300
emb_dim = 300
char_emb_dim = 52
filter_sizes = [25, 25]
channel_sizes = [5, 5]
# training parameters
lr = 0.001
lr_decay = 0.05
minimal_lr = 1e-5
keep_prob = 0.5
batch_size = 32
epochs = 30
max_to_keep = 1
no_imprv_tolerance = 20
checkpoint_path = "checkpoint_BiLSTM_Att/"
summary_path = "checkpoint_BiLSTM_Att/summary/"
model_name = "punctuation_model"

config = {
          "Word2vec_path":Word2vec_path,\
          "char_lowercase": char_lowercase,\
          "vocab": vocab,\
          
          "word_embedding": word_embedding,\
          "num_units": num_units,\
          "emb_dim": emb_dim,\
          "char_emb_dim": char_emb_dim,\
          "filter_sizes": filter_sizes,\
          "channel_sizes": channel_sizes,\
          "lr": lr,\
          "lr_decay": lr_decay,\
          "minimal_lr": minimal_lr,\
          "keep_prob": keep_prob,\
          "batch_size": batch_size,\
          "epochs": epochs,\
          "max_to_keep": max_to_keep,\
          "no_imprv_tolerance": no_imprv_tolerance,\
          "checkpoint_path": checkpoint_path,\
          "summary_path": summary_path,\
          "model_name": model_name}

# alpha & gamma for focal loss (tune hyperparameter)
alpha = 0.1
gamma = 0.5

model = BiLSTM_Attention_model(config, alpha, gamma)
model.restore_last_session(checkpoint_path)

def inference(model,text):
    return model.infer(text)

text = 'dân số toàn đô thị phú phong là gần 1000 người, mật độ dân số khu vực tập trung là 1000 người / km². phú phong là một trong bốn đô thị động lực phát triển kinh tế xã hội của tỉnh, bao gồm thành phố quy nhơn, thị xã an nhơn, thị trấn bồng sơn, thị trấn phú phong, đồng thời là cửa ngõ giao lưu quan trọng của tỉnh bình định với khu vực tây nguyên'

import re
def de_punc(text:str):
    raw = re.sub(r"[,:\-–.!;?]", '', text)
    return raw
text = de_punc(text)
print(text)
print(inference(model,text))