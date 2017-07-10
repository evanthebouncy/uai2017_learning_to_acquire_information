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

class Implynet:

  def gen_feed_dict(self, partial_obs, full_obs):
    ret = dict()
    ret[self.partial_obs] = partial_obs
    ret[self.full_obs] = full_obs
    return ret

  # load the model and give back a session
  def load_model(self, sess, saved_loc):
    self.saver.restore(sess, saved_loc)
    print("Model restored.")

  # make the model
  def __init__(self, name):
    with tf.variable_scope('imply') as scope:
      # set up placeholders
      self.partial_obs = tf.placeholder(tf.float32, [N_BATCH, L , 2], name="partial_obs")
      self.full_obs = tf.placeholder(tf.float32, [N_BATCH, L , 2], name="full_obs")

      # some constants
      self.n_hidden = 200

      # make hidden represnatation
      W1 = weight_variable([L * 2, self.n_hidden])
      b1 = bias_variable([self.n_hidden])

      W2 = weight_variable([self.n_hidden, self.n_hidden])
      b2 = bias_variable([self.n_hidden])

      partial_flat = tf.reshape(self.partial_obs, [N_BATCH, L * 2])
      hidden = tf.nn.relu(tf.matmul(partial_flat, W1) + b1)
      hidden = tf.nn.relu(tf.matmul(hidden, W2) + b2)

      W_preds = [weight_variable([self.n_hidden, 2]) for _ in range(L)]
      b_preds = [bias_variable([2]) for _ in range(L)]
      e2 = tf.constant(1e-10, shape=[N_BATCH, 2])

      self.query_preds = [tf.nn.softmax(tf.matmul(hidden, W_preds[i]) + b_preds[i])+e2 for i in range(L)]
      print "query_preds shape ", show_dim(self.query_preds)

      # doing some reshape of the input tensor
      full_obs_trans = tf.transpose(self.full_obs, perm=[1,0,2])
      print full_obs_trans.get_shape()
      full_obs_split = tf.reshape(full_obs_trans, [L, N_BATCH, 2])
      full_obs_split = tf.unpack(full_obs_split)
      print show_dim(full_obs_split) 

      self.query_pred_costs = []
      for idx in range(L ):
        blah = -tf.reduce_sum(full_obs_split[idx] * tf.log(self.query_preds[idx]))
        self.query_pred_costs.append(blah)
        
      print "costs shapes ", show_dim(self.query_pred_costs)
      self.cost_query_pred = sum(self.query_pred_costs)

      # ------------------------------------------------------------------------ training steps
      # gvs = optimizer.compute_gradients(cost)
      # capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
      # train_op = optimizer.apply_gradients(capped_gvs)


      # optimizer = tf.train.RMSPropOptimizer(0.0001)
      # optimizer = tf.train.RMSPropOptimizer(0.0001)
      optimizer = tf.train.AdagradOptimizer(0.005)


      pred_gvs = optimizer.compute_gradients(self.cost_query_pred)
      capped_pred_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in pred_gvs]
      #train_pred = optimizer.minimize(cost_pred, var_list = VAR_pred)
      self.train_query_pred = optimizer.apply_gradients(capped_pred_gvs)

      # train_query_pred = optimizer.minimize(cost_query_pred, var_list = VAR_pred)
      # Before starting, initialize the variables.  We will 'run' this first.
      self.init = tf.initialize_all_variables()
      self.saver = tf.train.Saver()

  # save the model
  def save(self, sess, model_loc="model_imply.ckpt"):
    save_path = self.saver.save(sess, model_loc)
    print("Model saved in file: %s" % save_path)

  # train on a particular data batch
  def train(self, sess, data_batch):
    partial_obs, full_obs = data_batch
    feed_dic = self.gen_feed_dict(partial_obs, full_obs)

#     qry_prd = sess.run(self.query_preds, feed_dict=feed_dic)
#     print qry_prd[0][0]
#     qry_costs = sess.run(self.query_pred_costs, feed_dict=feed_dic)
#     print qry_costs[0]

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
    _obs = np.zeros([L,2])
    for ob_idx in range(num_obs):
      xx, lab = obs[ob_idx]
      _obs[xx] = lab
    
    obss = np.array([_obs for i in range(N_BATCH)])

    feed_dic = dict()
    feed_dic[self.partial_obs] = obss
    return feed_dic

  def get_all_preds(self, sess, obs):
    dick = self.get_feed_dic_obs(obs)
    predzz = sess.run(self.query_preds, dick)
    predzz0 = np.array([x[0] for x in predzz])
    predzz0 = np.reshape(predzz0, [L,2])
    return predzz0

  def get_most_confuse(self, sess, obs):
    obs_qry = [_[0] for _ in obs]
    all_preds = self.get_all_preds(sess, obs)
    
    all_pred_at_key1 = []
    for i in range(L):
      qry = i
      value = all_preds[i]
      if qry not in obs_qry:
        all_pred_at_key1.append((qry, value))
    most_confs = [(abs(x[1][0] - x[1][1]), x[0]) for x in all_pred_at_key1]
    most_conf = min(most_confs)
    return most_conf[1]

  def get_active_trace(self, sess, query, epi=0.0):
    obs = []

    for i in range(OBS_SIZE):
      if np.random.random() < epi:
        rand_coord = sample_coord_new(obs)
        obs.append((rand_coord, query(rand_coord)))
      else:
        most_conf = self.get_most_confuse(sess, obs)
        obs.append((most_conf, query(most_conf)))

    return obs

class Invnet:
  
  def gen_feed_dict(self, true_lab, obs):
    ret = dict()
    ret[self.true_label] = true_lab
    ret[self.observations] = obs
    return ret

  # load the model and give back a session
  def load_model(self, sess, saved_loc):
    self.saver.restore(sess, saved_loc)
    print("Inversion Model restored.")

  # save the model
  def save(self, sess, model_loc="model_invert.ckpt"):
    save_path = self.saver.save(sess, model_loc)
    print("Model saved in file: %s" % save_path)

  def __init__(self, name):
    with tf.variable_scope('inv') as scope:
      self.true_label = tf.placeholder(tf.float32, [N_BATCH, X_L], name="true_label_"+name)
      self.observations = tf.placeholder(tf.float32, [N_BATCH, L, L, 2], name="obs_"+name)
      
      self.n_hidden = 1200

      W_inv1 = weight_variable([L*L*2, self.n_hidden], name="W_inv1_"+name)
      b_inv1 = bias_variable([self.n_hidden], name="b_inv1_"+name)

      W_inv2 = weight_variable([self.n_hidden,X_L], name="W_inv2_"+name)
      b_inv2 = bias_variable([X_L], name="b_inv2_"+name)

      self.VARS = [W_inv1, b_inv1, W_inv2, b_inv2]
      
      reshape_ob = tf.reshape(self.observations, [N_BATCH, L*L*2])
      blah = tf.nn.relu(tf.matmul(reshape_ob, W_inv1) + b_inv1)
      epsilon1 = tf.constant(1e-10, shape=[N_BATCH, X_L])
      self.pred = tf.nn.softmax(tf.matmul(blah, W_inv2) + b_inv2) + epsilon1
      self.cost = -tf.reduce_sum(self.true_label * tf.log(self.pred))

      optimizer = tf.train.RMSPropOptimizer(0.001)

      inv_gvs = optimizer.compute_gradients(self.cost)
      self.train_inv = optimizer.apply_gradients(inv_gvs)

      all_var_var = tf.get_collection(tf.GraphKeys.VARIABLES, scope='inv')
      self.init = tf.initialize_variables(all_var_var)
      self.saver = tf.train.Saver(self.VARS)

  # train on a particular data batch
  def train(self, sess, data_batch):
    true_lab, obss = data_batch
    feed_dic = self.gen_feed_dict(true_lab, obss)
    cost_pre = sess.run([self.cost], feed_dict=feed_dic)[0]
    sess.run([self.train_inv], feed_dict=feed_dic)
    cost_post = sess.run([self.cost], feed_dict=feed_dic)[0]
    print "train inv ", cost_pre, " ", cost_post, " ", True if cost_post < cost_pre else False

  # get inversion from observations
  def invert(self, sess, obs):
    obss = [obs for _ in range(N_BATCH)]
    fake_lab = [np.zeros(shape=[X_L]) for _ in range(N_BATCH)]
    data_in = inv_batch_obs(fake_lab, obss)
    feed_dic = self.gen_feed_dict(*data_in)
    return sess.run([self.pred], feed_dict=feed_dic)[0][0]



