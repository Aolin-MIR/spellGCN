import numpy as np
import tensorflow as tf


from bert_in_graph import BertModel, BertConfig

from data_utils import pad_sequences, pad_sequences_2
from sequence_labeling import f1_score, precision_score, recall_score

# import math
from datetime import datetime
from tensorflow.python import pywrap_tensorflow
import json
from tokenization import FullTokenizer, convert_to_unicode
import os
from tqdm import tqdm
tf.logging.set_verbosity(tf.logging.INFO)
import logging
logging.basicConfig(level=logging.INFO)
import optimization
class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask):
        self.input_ids = input_ids
        self.input_mask = input_mask


def minibatches_bert(data, minibatch_size):

    data_len = len(data)
    num_batch = int((data_len - 1) / minibatch_size) + 1
    for i in range(num_batch):
        start_id = i * minibatch_size
        end_id = min((i + 1) * minibatch_size, data_len)
        batch = data[start_id:end_id]
        
        yield batch,start_id, end_id

def construct_vocab(path):
    with open(path, encoding='utf-8') as f:
        vocab = set([x.strip() for x in f.readlines() if x.strip()])
    return vocab

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, text_a):
        """Constructs a InputExample.

        Args:
            text_a: string. The untokenized paragraph of the first sequence. For single
                sequence tasks, only this sequence must be specified.
        """
        self.text_a = text_a
def create_examples(texts):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, text) in enumerate(texts):
        text_a = convert_to_unicode(text)
        examples.append(
            InputExample(text_a=text_a))
    return examples



class BERT_GCN(object):

    def __init__(self, lr=5e-5,  beta=3.0, layers_num=2, dropout=1.0, bert_max_len = 192,dir_model='../checkpoint/train/gcn/', bert_dir='../rest-api/app/models/chinese_L-12_H-768_A-12',train=True):
        # super(BERT_GCN, self).__init__(config)
        self.lr = lr
        self.dir_model = dir_model
        # self.nepochs=nepochs
        self.dropout=dropout #1
        self.bert_dir = bert_dir
        self.bert_max_len = bert_max_len
        self.beta = beta
        self.layers_num = layers_num
        config_file = bert_dir + '/bert_config.json'
        self.init_checkpoint = bert_dir + '/bert_model.ckpt'
        vocab_file = bert_dir + '/vocab.txt'
        self.vocab = construct_vocab(vocab_file)
        self.tokenizer = FullTokenizer(
            vocab_file = vocab_file, do_lower_case = True)
        self.bert_api = BertModel(
            config=BertConfig.from_json_file(config_file))
    
        # Read data from checkpoint file
        reader = pywrap_tensorflow.NewCheckpointReader(self.init_checkpoint)
        var_to_shape_map = reader.get_variable_to_shape_map()
        # Print tensor name and values
        for key in var_to_shape_map:
            if "word_embeddings" in key:
                emb_table = reader.get_tensor(key)
                break
        
        with open("filter_dict.json", 'r') as load_f:
            dict_filter=json.load(load_f)
            # in_conf_ind=sorted(dict_filter.items(),key=lambda d:d[0])
        zero_id=len(dict_filter)
        print("zero_id:", zero_id)
        self.emb_table_filted=[]
        # y=-1
        self.w_index = [zero_id]*21128
        # self.b_index = []
        self.emb_mask=np.ones([21128,768])
        for x in dict_filter:
            # if dict_filter[x]<=y:
            #     print('!!!!!!!!!!wrong!!!!!!!!!!!!!!!!!!!')
            # y=dict_filter[x]
            self.w_index[int(x)]=dict_filter[x]
            self.emb_table_filted.append(emb_table[int(x)])
            # emb_table[int(x)]=np.zeros([768])
            self.emb_mask[int(x)] = np.zeros([768])

        # for x in range(21128):
        #     if x in dict_filter:
        #         self.b_index.append(21128)
        #     else:self.b_index.append(x)

        self.emb_table_filted=np.array(self.emb_table_filted)
        r = np.load('spellgcn_adj_norm.npz')
        self.p_A = r['A_p'].astype(np.float32)
        self.s_A = r['A_s'].astype(np.float32)
        self.p_A = tf.constant(self.p_A)
        self.s_A = tf.constant(self.s_A)
    def convert_examples_to_features(self, examples, max_seq_len, tokenizer, positions=None):
        """Convert a set of `InputExample`s to a list of `InputFeatures`."""
        features = []
        if positions:
            for (ex_index, example) in enumerate(examples):
                feature = self.convert_single_example(example,
                                                      max_seq_len, tokenizer, positions[ex_index])
                features.append(feature)
        else:
            for (ex_index, example) in enumerate(examples):
                feature = self.convert_single_example(example,
                                                      max_seq_len, tokenizer)
                features.append(feature)
        return features

    def convert_single_example(self, example, max_seq_len, tokenizer, positions=None):
        """Converts a single `InputExample` into a single `InputFeatures`."""
        # tokens_a = tokenizer.tokenize(example.text_a)  ###
        tokens_a = [
            x if x in self.vocab else '[UNK]' for x in list(example.text_a)]

        
        if len(tokens_a) > max_seq_len -2 :
            tokens_a = tokens_a[0:(max_seq_len-2)]
        tokens = ["[CLS]"]

        for token in tokens_a:
            tokens.append(token)
        tokens.append("[SEP]")
        if positions:
            for position in positions:
                tokens[int(position)] = "[MASK]"
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_len:
            input_ids.append(0)
            input_mask.append(0)

        assert len(input_ids) == max_seq_len
        assert len(input_mask) == max_seq_len

        feature = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask
        )
        return feature

    def add_train_op(self, lr_method, lr, loss, clip=-1):
        """Defines self.train_op that performs an update on a batch
        Args:
            lr_method: (string) sgd method, for example "adam"
            lr: (tf.placeholder) tf.float32, learning rate
            loss: (tensor) tf.float32 loss to minimize
            clip: (python float) clipping of gradient. If < 0, no clipping
        """
        _lr_m = lr_method.lower()  # lower to make sure

        with tf.variable_scope("train_step"):
            if _lr_m == 'adam':
                optimizer = tf.train.AdamOptimizer(lr)
            elif _lr_m == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(lr)
            elif _lr_m == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(lr)
            elif _lr_m == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(lr)
            elif _lr_m == 'adam_w':
                optimizer = optimization.AdamWeightDecayOptimizer(lr)
            else:
                raise NotImplementedError("Unknown method {}".format(_lr_m))
            # self.trainable_variables = tf.trainable_variables()
            self.grads,self.vs = zip(*optimizer.compute_gradients(loss))
            if clip > 0:  # gradient clipping if clip is positive
                grads, vs = zip(*optimizer.compute_gradients(loss))
                grads, gnorm = tf.clip_by_global_norm(grads, clip)
                self.train_op = optimizer.apply_gradients(zip(grads, vs))
            else:
                self.train_op = optimizer.minimize(loss)

    def add_placeholders(self):
        """Define placeholders = entries to computational graph"""

        


        self.labels = tf.placeholder(tf.int32, shape=[None, None],
                                     name="labels")

        # shape = (batch size)
        self.sequence_lengths = tf.placeholder(tf.int32, [None],
                                               name="sequence_lengths")

        # Hyper-parameters
        # self.bert_api.dropout = tf.placeholder(dtype=tf.float32, shape=[],
        #                               name="dropout")
        # self.lr = tf.placeholder(dtype=tf.float32, shape=[],
        #                          name="lr")

    def get_feed_dict(self, batch,  type = 'Train', dropout = None):
        # input is ids
        # batch_size = len(batch)  # dynamically calculate batch size


        if type=='Train':
            input_ids, label_ids, sequence_lengths,input_mask = [], [],[],[]
            for data in batch:
                input_len=len(data['token_ids'])
                input_ids.append(data['token_ids'])
                input_mask.append([1]*input_len)
                label_ids.append(data['labels'])
                sequence_lengths.append(input_len)
            max_length=min(max(sequence_lengths), self.bert_max_len)
            # print(217,label_ids[0])
            feed = {self.bert_api.input_ids: pad_sequences(input_ids ,max_length),
            self.bert_api.input_mask:pad_sequences(input_mask,max_length),
            self.labels: pad_sequences(label_ids,max_length),
            self.sequence_lengths: sequence_lengths,
            self.bert_api.dropout:dropout}
            return feed, sequence_lengths
        elif type=='Eval':
            input_ids, label_bio, sequence_lengths,input_mask = [], [], [],[]
            for data in batch:
                input_len = len(data['token_ids'])
                # print(254, data['token_ids'], len(data['token_ids']))
                input_ids.append(data['token_ids'])
                input_mask.append([1]*input_len)
                label_bio.append(data['bio'][1:-1])
                sequence_lengths.append(input_len)
            max_length = min(max(sequence_lengths), self.bert_max_len)
            
           
            feed = {self.bert_api.input_ids: pad_sequences(input_ids, max_length),
                    self.bert_api.input_mask: pad_sequences(input_mask, max_length),
                    self.sequence_lengths: sequence_lengths}
            return feed, input_ids,label_bio,sequence_lengths
        elif type =='Pred':
            #to be finished
            sequence_lengths = [len(x) for x in batch]
            max_length = min(max(sequence_lengths), self.bert_max_len)
            feed = {self.bert_api.input_ids: pad_sequences(batch, max_length),
                    self.sequence_lengths: sequence_lengths
                    }

     





            return feed, sequence_lengths


    def add_logits_op(self):
        self.embeddings = self.bert_api.get_sequence_output()
        # self.pt=self.embeddings
        def glorot(shape, name=None):
            """Glorot & Bengio (AISTATS 2010) init."""
            init_range = np.sqrt(6.0/(shape[0]+shape[1]))
            initial = tf.random_uniform(
                shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
            return tf.Variable(initial, name=name)
        p_gcn_w,s_gcn_w = {},{}
        with tf.variable_scope("gcn"):
            self.H = tf.expand_dims(self.emb_table_filted,0) # [1,n,768]
            w_a = tf.get_variable('weight_a', shape=[768,1], initializer=tf.truncated_normal_initializer(mean=0, stddev=5, seed=0))

            for i in range(self.layers_num):#gcn layers
                p_gcn_w['p_weight'+str(i)] = glorot([768, 768],'p_weight'+str(i))
                p_gcn_out = tf.matmul(
                    self.H[-1], p_gcn_w['p_weight'+str(i)]) # [n, 768]
                p_gcn_out = tf.matmul(
                    self.p_A, p_gcn_out)  # [n, 768]

                s_gcn_w['s_weight'+str(i)] = glorot([768, 768],'s_weight'+str(i))
                s_gcn_out = tf.matmul(
                    self.H[-1], s_gcn_w['s_weight'+str(i)])
                s_gcn_out = tf.matmul(
                    self.s_A, s_gcn_out)

                temp1=tf.exp(tf.matmul(p_gcn_out, w_a)/self.beta)
                temp2=tf.exp(tf.matmul(s_gcn_out, w_a)/self.beta)
                alpha_p=tf.squeeze(temp1/(temp1+temp2),-1) #[n]
                alpha_s=tf.squeeze(temp2/(temp1+temp2),-1)
                

                C = tf.transpose(
                    p_gcn_out, [1, 0])*alpha_p + tf.transpose(s_gcn_out, [1, 0])*alpha_s  # [768,n]
                C = tf.transpose(C,[1,0])
                h = C + tf.reduce_sum(self.H , 0) 
                # self.H= tf.stack(tf.unstack(self.H)+[h])
                self.H = tf.concat([self.H, tf.expand_dims(h,0)],0)

        with tf.variable_scope("proj"):
            # self.W = tf.constant(emb_table)
            # # self.W = tf.Variable(initial_value=emb_table,trainable=False)

            # # print(292,W.shape)
            # for i in dict_filter: # {bert_vocab_ind : confusionset_vocab_ind}
            #     ii=int(i)
            #     print(ii)
            #     # print(294,type(W[ii]))
            #     # print(295, type(self.H[-1][dict_filter[i]]))
            #     # W[ii].assign(self.H[-1][dict_filter[i]])
            #     self.W = tf.concat([self.W[:ii], tf.expand_dims(
            #         self.H[-1][dict_filter[i]], 0), self.W[ii+1:]], axis=0)

            _H = tf.concat([self.H[-1],tf.zeros([1,768])],0)#[7k+1,768]
            # print(_H.shape)
            _H = tf.gather(_H, indices=self.w_index)#[21128,768]
            emb_table=self.bert_api.get_embedding_table()
            emb_table=tf.multiply(emb_table,self.emb_mask)
            _H += emb_table # [21128,768]
            
            nsteps = tf.shape(self.embeddings)[1]
            # [batch_size * max_seq_len, 2 * num_units]
            outputs_flat = tf.reshape(self.embeddings, [-1, 768])
            
            pred = tf.matmul(outputs_flat, _H, transpose_b = True ) # + b
            self.logits = tf.reshape(pred, [-1, nsteps, 21128])


        # self.log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
        #     self.logits, self.labels, self.sequence_lengths)
        # self.trans_params = trans_params  # need to evaluate it for decoding
        # self.loss = tf.reduce_mean(-self.log_likelihood)
        
        self.loss = tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=tf.one_hot(self.labels,21128))
        self.loss = tf.reduce_mean(self.loss)
        # self.g1 = tf.gradients(ys=self.loss, xs=self.embeddings)
        # for tensorboard
        tf.summary.scalar("loss", self.loss)

    def initialize_session(self):
        """Defines self.sess and initialize the variables"""
        #self.logger.info("Initializing tf session")
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        # sess_config.gpu_options.per_process_gpu_memory_fraction = 0.4
        sess_config.log_device_placement = False
        self.sess = tf.Session(config=sess_config)
        wirter = tf.summary.FileWriter('gcn/logs/', self.sess.graph)
        # self.sess.run(tf.global_variables_initializer(),
        #               options=tf.RunOptions(report_tensor_allocations_upon_oom=True))
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        return self.sess
    def build_model(self):
        self.add_placeholders()

        self.add_logits_op()


        # Generic functions that add training op and initialize session
        self.add_train_op('adam_w', self.lr, self.loss, 1)
        # self.train_op = optimization.create_optimizer(
        #     self.loss, self.lr, 100000, 10000, use_tpu=False)
        self.sess = self.initialize_session()  # now self.sess is defined and vars are init
        self.bert_api.restore(self.sess,self.init_checkpoint)
        
        for variable_name in tf.global_variables():
            if not 'bert' in str(variable_name):
                print(variable_name)
    def evaluate_batch(self, sentences):

        fd, input_ids,label_true, sequence_lengths = self.get_feed_dict(sentences, type='Eval')
        #sequence_lengths contains [cls][sep]

        logits = self.sess.run(
            self.logits, feed_dict=fd)
        # print(379, logits.shape)
        logits = np.argmax(logits,axis=-1)
        
        label_pred = []
        for i,logit in enumerate(logits):
            # print(382,logit.shape)
            temp = []
            # print(406, i,sequence_lengths[i], list(range(sequence_lengths[i])[1:-1]))
            for j in range(sequence_lengths[i])[1:-1]:
                # print(383, j, logit[j], input_ids[i][j])
                if logit[j] == input_ids[i][j]:
                    temp.append('O')
                else:
                    temp.append('B-Err')
            label_pred.append(temp)
        return label_pred,label_true

    def train(self, train, dev,epoch_num):
        """Performs training with early stopping and lr exponential decay
        Args:
            train: dataset that yields tuple of (sentences, tags)
            dev: dataset
        """
        best_score = 0
        nepoch_no_imprv = 0  # for early stopping
        

        for epoch in range(epoch_num):
            tf.logging.info('epoch:'+str(epoch))

            score = self.run_epoch(train, dev, epoch)
            self.lr *= 0.9  # decay learning rate

            # early stopping and saving best parameters
            if score >= best_score:
                nepoch_no_imprv = 0
                self.save_session()
                best_score = score
                tf.logging.info("- new best score!")
            else:
                nepoch_no_imprv += 1
                if nepoch_no_imprv >= 3:
                    tf.logging.info("- early stopping {} epochs without "
                                     "improvement".format(nepoch_no_imprv))
                    break

    def save_session(self):
        """Saves session = weights"""
        if not os.path.exists(self.dir_model):
            os.makedirs(self.dir_model)
        self.saver.save(self.sess, self.dir_model)

    def run_epoch(self, train, dev, epoch):

        batch_size = 32
        nbatches = (len(train) + batch_size - 1) // batch_size

        # Iterate over dataset
        for i, (batch, _, _) in tqdm(enumerate(minibatches_bert(train, batch_size)) ,total=nbatches):
            fd, _ = self.get_feed_dict(batch, type='Train',
                                       dropout=self.dropout) #dropout=1
            # print(i)
            try:
                # _, train_loss,g,v = self.sess.run(
                #     [self.train_op, self.loss,self.grads,self.vs], feed_dict=fd)
                _, train_loss= self.sess.run(
                    [self.train_op, self.loss], feed_dict=fd)
                # tf.logging.info('loss:'+str(train_loss))
                # tf.logging.info('g'+str(g))
                # tf.logging.info('pt'+str(logits))
            except ValueError as eer:
                print(eer,i)
            # Write to Tensorboard


        metrics = self.run_evaluate(dev)
        msg = " - ".join(["{}: {:06.4f}".format(k, v)
                          for k, v in metrics.items()])
        tf.logging.info(msg)

        return metrics["f1"]

    def run_evaluate(self, test):
        

        batch_size = 32
        f1_total = 0.0
        precision_total = 0.0
        recall_total = 0.0
        iteration = 0

        for batch, _,_ in minibatches_bert(test, batch_size):
            # y_true = []
            # y_pred = []

            labels_pred, labels_true = self.evaluate_batch(batch)
            # for i in range(len(labels_pred)):
            #     _y_true = [self.idx_to_tag[x] for x in labels[i]]
            #     _y_pred = [self.idx_to_tag[x] for x in labels_pred[i]]
            #     y_true.append(_y_true)
            #     y_pred.append(_y_pred)
            # print(499,labels_pred,labels_true)
            f1 = f1_score(labels_true, labels_pred)
            precision = precision_score(labels_true, labels_pred)
            recall = recall_score(labels_true, labels_pred)

            metrics = {"f1": f1, "precision": precision, "recall": recall}



            iteration += 1
            f1_total += f1
            precision_total += precision
            recall_total += recall

        f1_avg = f1_total / iteration
        p_avg = precision_total / iteration
        r_avg = recall_total / iteration

        metrics = {"f1": f1_avg, "precision": p_avg, "recall": r_avg}

        msg = "Final Scores: " + " - ".join(["{}: {:06.4f}".format(k, v)
                                             for k, v in metrics.items()])
        #self.logger.info(msg)
        # tf.logging.info(msg)
        return metrics



    def make_batches(self, tuples):
        length_table = {}
        for tup in tuples:
            if tup[1] in length_table:
                length_table[tup[1]].append(tup)
            else:
                length_table[tup[1]] = [tup]
        return [length_table[x] for x in length_table]





  

if __name__ == "__main__":
    model = BERT_GCN(dropout=0.9)
    model.build_model()
    data_path = 'your_data_path.npy'
    data_set = np.load(data_path, allow_pickle=True)
    train, dev = data_set
    model.train(train,dev,30)
        
