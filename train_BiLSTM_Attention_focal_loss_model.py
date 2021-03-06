import pickle
import tensorflow as tf
from preprocessing import process_data
from model_tf2 import BiLSTM_Attention_model
# from model_tf2 import train
from model import batchnize_dataset
import traceback
# dataset path
prefix_dataset_path = 'dataset/icomm_news_dataset/'
raw_path = prefix_dataset_path + 'Cleansed_data'
save_path = prefix_dataset_path + "Encoded_data"
# embedding path
Word2vec_path = "embeddings"

char_lowercase = True
# dataset for train, validate and test
vocab = prefix_dataset_path + "Encoded_data/vocab.json"
train_set = prefix_dataset_path + "Encoded_data/train.json"
dev_set = prefix_dataset_path +  "Encoded_data/dev.json"
test_set = prefix_dataset_path + "Encoded_data/test.json"
word_embedding = prefix_dataset_path + "Encoded_data/word_emb.npz"
# network parameters
num_units = 300
emb_dim = 300
char_emb_dim = 52

# for convolution on char embedding
filter_sizes = [25, 25]
channel_sizes = [5, 5]

# training parameters
lr = 0.001
lr_decay = 0.05
minimal_lr = 1e-5
keep_prob = 0.5
batch_size = 2
epochs = 30
max_to_keep = 1
no_imprv_tolerance = 20
checkpoint_path = "checkpoint_BiLSTM_Att_icomm_news/"
summary_path = "checkpoint_BiLSTM_Att_icomm_news/summary/"
model_name = "punctuation_model"
import pickle
config = {"raw_path": raw_path,\
          "save_path": save_path,\
          "Word2vec_path":Word2vec_path,\
          "char_lowercase": char_lowercase,\
          "vocab": vocab,\
          "train_set": train_set,\
          "dev_set": dev_set,\
          "test_set": test_set,\
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
import os

if not os.path.exists(config["save_path"]):
    os.mkdir(config["save_path"])
process_data(config)

print("Load datasets...")

# used for training
train_set = batchnize_dataset(config["dev_set"], config["batch_size"], shuffle=True)
with open('dataset/icomm_news_dataset/cached_dataset/train.pkl','wb') as f:
    pickle.dump(train_set,f)

valid_set = batchnize_dataset(config["dev_set"], batch_size=config["batch_size"], shuffle=False)
with open('dataset/icomm_news_dataset/cached_dataset/valid.pkl','wb') as f:
    pickle.dump(valid_set,f)

# valid_set = batchnize_dataset(config["test_set"], batch_size=config["batch_size"], shuffle=False)
# with open('dataset/icomm_news_dataset/cached_dataset/test.pkl','wb') as f:
#     pickle.dump(valid_set,f)


# with open('dataset/icomm_news_dataset/cached_dataset/train.pkl','rb') as f:
#     train_set = pickle.load(f)

# with open('dataset/icomm_news_dataset/cached_dataset/valid.pkl','rb') as f:
#     valid_set = pickle.load(f)
print('Number of batch (batch size = 2):')
print(len(train_set))
print(len(valid_set))
# with open('dataset/icomm_news_dataset/cached_dataset/test.pkl','rb') as f:
#     test_set = pickle.load(f)
print('load done')
tf.config.threading.set_inter_op_parallelism_threads(
    30
)
# tf.compat.v1.disable_eager_execution()
# tf.compat.v1.reset_default_graph()

physical_devices = tf.config.list_physical_devices('GPU') 
print("Num GPUs Available: ", len(physical_devices))

for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
print("Build models...")

# model.restore_last_session(checkpoint_path)

# print(model.infer('tt n???n khai th??c c??t l???u ??? s??ng ?????ng nai ??o???n qua p long ph?????c q <num> tp hcm ng??y c??ng l???ng h??nh xem th?????ng ph??p lu???t m???t ng??y c??c ghe b??n c??? tr??m m??t kh???i c??t ch??a k??? l???p b??n ph??a tr??n m???t ph???i b??m b??? ??i th?? ?????t n??o c??n m???y anh th???y ???? tr?????c ????y ?????t c???a ng?????i d??n nh??ng b??y gi??? th??nh s??ng h???t r???i ??ng q ch??? tay v??? chi???c ghe ??ang h??t c??t ?????u c??ch b??? h??n <num> n??i v?? v???y tr?????c m???i l???i l???n n??y ng?????i ta ??ang lao v??o khai th??c v???i'))
model = BiLSTM_Attention_model(config, alpha, gamma)
model.train(train_set, valid_set)


# run the session
# model.restore_last_session(checkpoint_path)
# model.test(test_set)
