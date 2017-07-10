# BATTLE SHIP

import tensorflow as tf
import numpy as np
from data import *

def weight_variable(shape, name=None):
  initial = tf.truncated_normal(shape, stddev=0.1)
  if name:
    return tf.Variable(initial, name)
  else:
    return tf.Variable(initial)

def bias_variable(shape, name=None):
  initial = tf.constant(0.1, shape=shape)
  if name:
    return tf.Variable(initial, name)
  else:
    return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def embed_0_layer(input_var):
  partial_flat = tf.reshape(input_var, [N_BATCH, L * L * 2])
  return partial_flat

def embed_1_layer(input_var):
  partial_flat = tf.reshape(input_var, [N_BATCH, L * L * 2])
  embed = tf.layers.dense(partial_flat, 200, activation= tf.nn.relu)
  return embed


def embed_cnn_layer(input_var):

  W_conv = weight_variable([3, 3, 2, 2])
  b_conv = bias_variable([2])

  xxx_R17 = conv2d(input_var, W_conv)
  cur = tf.nn.relu(xxx_R17 + b_conv)
  embed = tf.reshape(cur, [N_BATCH, L * L * 2])

  embed = tf.layers.dense(embed, 200, activation= tf.nn.relu)

  return embed

class Implynet:

  def gen_feed_dict(self, partial_obs, full_obs):
    ret = dict()
    ret[self.partial_obs] = partial_obs
    ret[self.full_obs] = full_obs
    return ret

  # load the model and give back a session
  def load_model(self, saved_loc):
    sess = self.sess
    self.saver.restore(sess, saved_loc)
    print("Model restored.")

  # make the model
  def __init__(self, embed_func, embed_name, sess):
    self.name = 'imply_' + embed_name
    with tf.variable_scope(self.name) as scope:
      # set up placeholders
      self.partial_obs = tf.placeholder(tf.float32, [N_BATCH, L , L , 2], name="partial_obs")
      self.full_obs = tf.placeholder(tf.float32, [N_BATCH, L , L , 2], name="full_obs")

      # embed the input
      embed = embed_func(self.partial_obs) 
      embed_dim = int(embed.get_shape()[1])
    
      # do the prediction on top of the embedding
      W_preds = [weight_variable([embed_dim, 2]) for _ in range(L*L)]
      b_preds = [bias_variable([2]) for _ in range(L*L)]
      e2 = tf.constant(1e-10, shape=[N_BATCH, 2])

      self.query_preds = [tf.nn.softmax(tf.matmul(embed, W_preds[i]) + b_preds[i])+e2 for i in range(L*L)]
      print "query_preds shape ", show_dim(self.query_preds)

      # doing some reshape of the input tensor
      full_obs_trans = tf.transpose(self.full_obs, perm=[1,2,0,3])
      print full_obs_trans.get_shape()
      full_obs_split = tf.reshape(full_obs_trans, [L*L, N_BATCH, 2])
      full_obs_split = tf.unstack(full_obs_split)
      print show_dim(full_obs_split) 

      self.query_pred_costs = []
      for idx in range(L * L):
        blah = -tf.reduce_sum(full_obs_split[idx] * tf.log(self.query_preds[idx]))
        self.query_pred_costs.append(blah)
        
      print "costs shapes ", show_dim(self.query_pred_costs)
      self.cost_query_pred = sum(self.query_pred_costs)

      # ------------------------------------------------------------------------ training steps
      optimizer = tf.train.AdagradOptimizer(0.01)


      pred_gvs = optimizer.compute_gradients(self.cost_query_pred)
      capped_pred_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in pred_gvs]
      #train_pred = optimizer.minimize(cost_pred, var_list = VAR_pred)
      self.train_query_pred = optimizer.apply_gradients(capped_pred_gvs)

      # train_query_pred = optimizer.minimize(cost_query_pred, var_list = VAR_pred)
      # Before starting, initialize the variables.  We will 'run' this first.
      self.init = tf.global_variables_initializer()
      self.saver = tf.train.Saver()
      self.sess = sess

  def initialize(self):
    self.sess.run(self.init)

  # save the model
  def save(self):
    model_loc = "./models/" + self.name+".ckpt"
    sess = self.sess
    save_path = self.saver.save(sess, model_loc)
    print("Model saved in file: %s" % save_path)

  # train on a particular data batch
  def train(self, data_batch):
    sess = self.sess

    partial_obs, full_obs = data_batch
    feed_dic = self.gen_feed_dict(partial_obs, full_obs)

    cost_query_pred_pre = sess.run([self.cost_query_pred], feed_dict=feed_dic)[0]
    sess.run([self.train_query_pred], feed_dict=feed_dic)
    cost_query_pred_post = sess.run([self.cost_query_pred], feed_dict=feed_dic)[0]
    print "train query pred ", cost_query_pred_pre, " ",\
      cost_query_pred_post, " ", True if cost_query_pred_post < cost_query_pred_pre else False

  # =========== HELPERS =============

  # a placeholder to feed in a single observation
  def get_feed_dic_obs(self, obs):
    # needing to create all the nessisary feeds
    obss = []
    
    num_obs = len(obs)
    _obs = np.zeros([L,L,2])
    for ob_idx in range(num_obs):
      cord, lab = obs[ob_idx]
      xx, yy = cord
      _obs[xx][yy] = lab
    
    obss = np.array([_obs for i in range(N_BATCH)])

    feed_dic = dict()
    feed_dic[self.partial_obs] = obss
    return feed_dic

  def get_all_preds(self, obs):
    sess = self.sess
    dick = self.get_feed_dic_obs(obs)
    predzz = sess.run(self.query_preds, dick)
    predzz0 = np.array([x[0] for x in predzz])
    predzz0 = np.reshape(predzz0, [L,L,2])
    return predzz0

  def get_most_confuse(self, sess, obs):
    obs_qry = [_[0] for _ in obs]
    all_preds = self.get_all_preds(obs)
    
    all_pred_at_key1 = []
    for i in range(L):
      for j in range(L):
        qry = i, j
        value = all_preds[i][j]
        if qry not in obs_qry:
          all_pred_at_key1.append((qry, value))
    most_confs = [(abs(x[1][0] - x[1][1]), x[0]) for x in all_pred_at_key1]
    most_conf = min(most_confs)
    return most_conf[1]

  def get_most_likely(self, sess, obs):
    obs_qry = [_[0] for _ in obs]
    all_preds = self.get_all_preds(sess, obs)
    
    all_pred_at_key1 = []
    for i in range(L):
      for j in range(L):
        qry = i, j
        value = all_preds[i][j]
        if qry not in obs_qry:
          all_pred_at_key1.append((qry, value))
    most_likely = [(x[1][0], x[0]) for x in all_pred_at_key1]
    most_conf = max(most_likely)
    return most_conf[1]
    

  def get_active_trace(self, query, epi=0.0, play=False):
    sess = self.sess

    obs = []

    for i in range(OBS_SIZE):
      if np.random.random() < epi:
        rand_coord = sample_coord_new(obs)
        obs.append((rand_coord, query(rand_coord)))
      else:
        most_conf = self.get_most_confuse(sess, obs) if not play else self.get_most_likely(sess, obs)
        obs.append((most_conf, query(most_conf)))

    return obs

# class Invnet:
#   
#   def gen_feed_dict(self, true_lab, obs):
#     ret = dict()
#     ret[self.true_label] = true_lab
#     ret[self.observations] = obs
#     return ret
# 
#   # load the model and give back a session
#   def load_model(self, sess, saved_loc):
#     self.saver.restore(sess, saved_loc)
#     print("Inversion Model restored.")
# 
#   # save the model
#   def save(self, sess, model_loc="model_invert.ckpt"):
#     save_path = self.saver.save(sess, model_loc)
#     print("Model saved in file: %s" % save_path)
# 
#   def __init__(self, name):
#     with tf.variable_scope('inv') as scope:
#       self.true_label = tf.placeholder(tf.float32, [N_BATCH, X_L], name="true_label_"+name)
#       self.observations = tf.placeholder(tf.float32, [N_BATCH, L, L, 2], name="obs_"+name)
#       
#       self.n_hidden = 1200
# 
#       W_inv1 = weight_variable([L*L*2, self.n_hidden], name="W_inv1_"+name)
#       b_inv1 = bias_variable([self.n_hidden], name="b_inv1_"+name)
# 
#       W_inv2 = weight_variable([self.n_hidden,X_L], name="W_inv2_"+name)
#       b_inv2 = bias_variable([X_L], name="b_inv2_"+name)
# 
#       self.VARS = [W_inv1, b_inv1, W_inv2, b_inv2]
#       
#       reshape_ob = tf.reshape(self.observations, [N_BATCH, L*L*2])
#       blah = tf.nn.relu(tf.matmul(reshape_ob, W_inv1) + b_inv1)
#       epsilon1 = tf.constant(1e-10, shape=[N_BATCH, X_L])
#       self.pred = tf.nn.softmax(tf.matmul(blah, W_inv2) + b_inv2) + epsilon1
#       self.cost = -tf.reduce_sum(self.true_label * tf.log(self.pred))
# 
#       optimizer = tf.train.RMSPropOptimizer(0.001)
# 
#       inv_gvs = optimizer.compute_gradients(self.cost)
#       self.train_inv = optimizer.apply_gradients(inv_gvs)
# 
#       all_var_var = tf.get_collection(tf.GraphKeys.VARIABLES, scope='inv')
#       self.init = tf.initialize_variables(all_var_var)
#       self.saver = tf.train.Saver(self.VARS)
# 
#   # train on a particular data batch
#   def train(self, sess, data_batch):
#     true_lab, obss = data_batch
#     feed_dic = self.gen_feed_dict(true_lab, obss)
#     cost_pre = sess.run([self.cost], feed_dict=feed_dic)[0]
#     sess.run([self.train_inv], feed_dict=feed_dic)
#     cost_post = sess.run([self.cost], feed_dict=feed_dic)[0]
#     print "train inv ", cost_pre, " ", cost_post, " ", True if cost_post < cost_pre else False
# 
#   # get inversion from observations
#   def invert(self, sess, obs):
#     obss = [obs for _ in range(N_BATCH)]
#     fake_lab = [np.zeros(shape=[X_L]) for _ in range(N_BATCH)]
#     data_in = inv_batch_obs(fake_lab, obss)
#     feed_dic = self.gen_feed_dict(*data_in)
#     return sess.run([self.pred], feed_dict=feed_dic)[0][0]



