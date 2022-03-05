import tensorflow as tf
print("TensorFlow version:", tf.__version__)
from model import *
import keras
from layers_modify import multi_conv1d
from tensorflow.keras.layers import Embedding

from tensorflow.keras import Model
import sys
# from tensorflow_addons.text import crf_log_likelihood,viterbi_decode
from keras_self_attention import SeqSelfAttention
from layers_modify import AttentionCell
import os
from loss_function import CustomLoss
from logging import getLogger
class BiLSTM_Attention_model(Model):
    def __init__(self, config, alpha, gamma):
        super().__init__()
        print('init model')
        self.cfg = config
        self.alpha = alpha
        self.gamma = gamma
        # Create folders
        if not os.path.exists(self.cfg["checkpoint_path"]):
            os.makedirs(self.cfg["checkpoint_path"])
        if not os.path.exists(self.cfg["summary_path"]):
            os.makedirs(self.cfg["summary_path"])

        #Create logger
        self.logger = get_logger(os.path.join(self.cfg["checkpoint_path"], str(self.gamma) + str(self.alpha) + "log.txt"))

        # Load dictionary
        dict_data = load_data(self.cfg["vocab"])
        self.word_dict, self.char_dict = dict_data["word_dict"], dict_data["char_dict"]
        self.label_dict = dict_data["label_dict"]
        
        del dict_data
        self.word_vocab_size = len(self.word_dict)
        self.char_vocab_size = len(self.char_dict)
        self.label_vocab_size = len(self.label_dict)

        self.max_to_keep = self.cfg["max_to_keep"]
        self.checkpoint_path = self.cfg["checkpoint_path"]
        self.summary_path = self.cfg["summary_path"]
        self.word_embedding = self.cfg["word_embedding"]

        self.drop_rate = 1 - self.cfg['keep_prob']
        self.lr = self.cfg['lr']
        # Build embedding layer
        
        embedding_matrix = np.load(self.cfg["word_embedding"])["embeddings"]
        self.word_embeddings   = Embedding(
            embedding_matrix.shape[0],
            output_dim= self.cfg['emb_dim'],
            embeddings_initializer=keras.initializers.Constant(embedding_matrix),
            trainable=False,
        )
        
        self.char_embeddings   = Embedding(
            self.char_vocab_size,
            output_dim= self.cfg["char_emb_dim"],
            trainable=True,
        )

        self.char_conv = tf.keras.layers.Conv2D(25,(1,5))
        self.dropout1 = tf.keras.layers.Dropout(rate=self.drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(rate=self.drop_rate)
      

            
        self.lstm_fw_cell = tf.keras.layers.LSTM(self.cfg["num_units"],return_sequences=True)
    
        self.lstm_bw_cell = tf.keras.layers.LSTM(self.cfg["num_units"],go_backwards=True,return_sequences=True)

        self.bidirectional_lstm = tf.keras.layers.Bidirectional(self.lstm_fw_cell,backward_layer= self.lstm_bw_cell)
        self.dense_1 = keras.layers.Dense(units=2 * self.cfg["num_units"],use_bias=False)
        
        self.attn_cell = SeqSelfAttention(attention_activation='sigmoid')
        
    
        self.dense_2 = tf.keras.layers.Dense(units=self.label_vocab_size,use_bias=True)
            

    def restore_last_session(self, ckpt_path=None):
        pass
        if ckpt_path is not None:
            ckpt = tf.train.get_checkpoint_state(ckpt_path)
        else:
            ckpt = tf.train.get_checkpoint_state(self.checkpoint_path)  # get checkpoint state
        if ckpt and ckpt.model_checkpoint_path:  # restore session
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def save_session(self, epoch):
        pass
        self.saver.save(self.sess, self.checkpoint_path + self.cfg["model_name"], global_step=epoch)

    def close_session(self):
        pass
        self.sess.close()

    def _add_summary(self):
        pass
        self.summary = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.summary_path + "train", self.sess.graph)
        self.test_writer = tf.summary.FileWriter(self.summary_path + "test")

    def _get_feed_dict(self, batch, keep_prob=1.0, is_train=False, lr=None):
        feed_dict = {'words': batch["words"], "seq_len": batch["seq_len"], "batch_size": batch["batch_size"]}
        if "labels" in batch:
            feed_dict["labels"] = batch["labels"]
        feed_dict["chars"] = batch["chars"]
        feed_dict["char_seq_len"] = batch["char_seq_len"]
        feed_dict["keep_prob"] = keep_prob 
        feed_dict["drop_rate"] = 1.0 - keep_prob 
        feed_dict["is_train"] = is_train 
        if lr is not None:
            feed_dict["lr"] = lr
        return feed_dict
    def call(self,words_tokens,chars_tokens):
        # feed_dict = self._get_feed_dict(batch)
        # Build embedding layer
        
       
        word_emb = self.word_embeddings(words_tokens)
        char_emb = self.char_embeddings(chars_tokens)

        char_represent = multi_conv1d(char_emb, self.cfg["filter_sizes"],self.cfg["channel_sizes"],conv2d=self.char_conv,
                                        drop_rate=self.drop_rate, is_train=True)
        word_emb = tf.concat([word_emb, char_represent], axis=-1)

        word_emb = self.dropout1(word_emb)
    
        rnn_outs = self.bidirectional_lstm(word_emb)
        # As we have a Bi-LSTM, we have two outputs which are not connected, so we need to merge them.

        rnn_outs = tf.concat(rnn_outs, axis=-1)

        rnn_outs =self.dropout2(rnn_outs)

        p_context = self.dense_1(rnn_outs)
        # print('p_context',p_context.get_shape())
        attn_outs = self.attn_cell( p_context)  # time major based

        logits = self.dense_2(attn_outs)
        # print('logits')
        return logits
#         

    def train_epoch(self, train_set,valid_set, epoch,loss_obj,optimizer):
        
        num_batches = len(train_set)
        print('num batches',num_batches)
        prog = Progbar(target=num_batches)
        for i, batch_data in tqdm(enumerate(train_set)):
            # import time
            words_tokens, chars_tokens,labels = batch_data
       

            # st_time  = time.time()
            with tf.GradientTape() as tape:

                logits = self(words_tokens,chars_tokens)
                train_loss = loss_obj(logits,labels)
            # print('fw and loss: ',time.time() - st_time)
            
            gradients = tape.gradient(train_loss, self.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            # print('bw: ',time.time() - st_time)
            
            cur_step = (epoch - 1) * num_batches + (i + 1)
            if (i % 10 == 0):           
                prog.update(i + 1, [("Global Step", int(cur_step)), ("Train Loss", train_loss)])

        '''
        Later restore
        '''
        # for j, batch_data in enumerate(valid_set):
        #     feed_dict = self._get_feed_dict(batch_data)
        # # self.test_writer.add_summary(val_summary, step)
        # micro_f_val, out_str, micro = self.evaluate_punct(valid_set, "val")

        # return micro_f_val, train_loss
        return -1, train_loss
    '''
    Convert to tf.dataset (Utilize gpu)
    '''
    def flatten_dataset(self,custom_set):
        '''
        Reshape char token. Each list present a word need  to be modified to fix length 
        '''
        def modify(my_set):
            fix = 6
            res = []
            for record in my_set:
                new_record = []
                for word in record:
                    if len(word) > fix:
                        word = word[:fix] # Dump code. Modify later. 
                    elif len(word) < fix:
                        pad = fix - len(word)
                        word.extend([0]*pad)
                    new_record.append(word)
                res.append(new_record)
            return res
        words_tokens = []
        chars_tokens = []
        labels_tokens = []

        for batch in tqdm(custom_set):
            words_tokens.extend(batch['words'])
            chars_tokens.extend(batch['chars'])
            labels_tokens.extend(batch['labels'])
        chars_tokens = modify(chars_tokens)

        words_tokens = tf.convert_to_tensor(words_tokens)
        chars_tokens =  tf.convert_to_tensor(chars_tokens)
        labels_tokens = tf.convert_to_tensor(labels_tokens)
        print('return dataset')
        return tf.data.Dataset.from_tensor_slices((words_tokens,chars_tokens,labels_tokens)).batch(32)
    def train(self, train_set, valid_set):
        self.logger.info("Start training...")
        best_f1, no_imprv_epoch = -np.inf, 0
        # self._add_summary()
        loss_obj = CustomLoss()
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        train_set = self.flatten_dataset(train_set)
        valid_set = self.flatten_dataset(valid_set)
        print('load done')
        for epoch in range(1, self.cfg["epochs"] + 1):
            self.logger.info('Epoch {}/{}: '.format(epoch, self.cfg["epochs"],))
            micro_f_val, train_loss = self.train_epoch(train_set,valid_set, epoch,loss_obj,optimizer)  # train epochs
            
            self.logger.info('Train loss: {} - Valid micro average fscore: {}'.format(train_loss, micro_f_val))
            cur_f1 = micro_f_val
            if cur_f1 > best_f1:
                no_imprv_epoch = 0
                best_f1 = cur_f1
                '''
                Need to restore this block of code to evaluate model after 1 epoch
                '''
    #               f1_test, out_str = self.evaluate_punct(test_set, "test")
    #               self.logger.info("\nEvaluate on {} dataset:\n{}\n".format("test", out_str))
                
            else:
                no_imprv_epoch += 1
                if no_imprv_epoch >= self.cfg["no_imprv_tolerance"]:
                    self.logger.info("Early Stopping at epoch - Valid micro average fscore: {:04.2f} - {:04.2f}".format(epoch, best_f1))
                    break
            self.save_weights(self.checkpoint_path + self.cfg["model_name"])
    
    
    def _predict_op(self, data):
        pass
        feed_dict = self._get_feed_dict(data)
        pred_logits = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32)
        logits = self.sess.run(pred_logits, feed_dict=feed_dict)
        return logits
    
    # def infer(self,text:str)->str:
        
        # tmp_ls = [get_infer_dict(text)]
        
        # print('tmp_ls',tmp_ls)
        # words = tmp_ls[0]['words']
        # # data, word_dict, char_dict, label_dict
        # tmp_ls = build_dataset(tmp_ls,self.word_dict,self.char_dict,self.label_dict)
        # print('after build dataset',tmp_ls)
        # tmp_ls = batchnize_dataset(tmp_ls)
        # print('after batch nize',tmp_ls)
        # predicts = self._predict_op(tmp_ls[0])
        # print(predicts)
        # text = restore_text(words,predicts[0])
        # print(text)
        # return text
    def evaluate_punct(self, dataset, name):
        '''
        Modify this function (5 labels only)
        '''
        pass
        PUNCTUATIONS = ['O','PERIOD', 'COMMA', 'EXCLAM', 'COLON', 'QMARK','SEMICOLON']
        preds = []
        labels = []

        TP = 0.0
        FP = 0.0
        FN = 0.0

        num_class = len(PUNCTUATIONS)

#        cfm = [ [0 for i in range(7)] for j in range(num_class)]

        for data in dataset:
            predicts = self._predict_op(data)
            # print('predicts')
            # print(tf.shape(predicts))
            for pred, tag, seq_len in zip(predicts, data["labels"], data["seq_len"]):
                preds.append(pred[:seq_len])
#                print(preds)
                labels.append(tag[:seq_len])
                for i in range(len(pred)):
                    for l in range(1,7):
                        if (pred[i] == tag[i]) and (tag[i] == l):
                            TP += 1
                        elif (pred[i] != tag[i]) and (tag[i] == l):
                            FN += 1
                        elif (pred[i] != tag[i]) and (pred[i] == l):
                            FP += 1

        labels = [y for x in labels for y in x]
        preds = [y for x in preds for y in x]

        precision = precision_score(labels, preds, average=None)
        recall = recall_score(labels, preds, average=None)
        f_score = f1_score(labels, preds, average=None)

        if (TP + FN) != 0:
            micro_r = TP / (TP + FN)
        else:
            micro_r = 0
        ###################
        if (TP + FP) != 0:
            micro_p = TP / (TP + FP)
        else:
            micro_p = 0
        ################
        if (micro_r + micro_p) > 0:
            micro_f = 2*micro_r * micro_p / (micro_r + micro_p)
        else:
            micro_f = 0.0

        micro = 'MICRO AVERAGE:\n\t Precision: ' + str(100*micro_p) + '%\n\tRecall: ' + str(100*micro_r) + ' %\n\t F_1 score: '  + str(100*micro_f) + ' %\n'

        out_str = "-" * 46 + "\n"
        out_str += "{:<16} {:<9} {:<9} {:<9}\n".format("PUNCTUATION", "PRECISION", "RECALL", "F-SCORE")

        for i in range(1,num_class):
            out_str += u"{:<16} {:<9} {:<9} {:<9}\n".format(PUNCTUATIONS[i], "{:.4f}".format(100*precision[i]),
                                                            "{:.4f}".format(100*recall[i]),
                                                            "{:.4f}".format(100*f_score[i]))
        return micro_f, out_str, micro


