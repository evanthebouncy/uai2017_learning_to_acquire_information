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

class Hypothesisnet:

  def gen_feed_dict(self, x_x, obs_x, obs_y, obs_tf):
    ret = {}
    for a, b in zip(self.ph_obs_x, obs_x):
      ret[a] = b
    for a, b in zip(self.ph_obs_y, obs_y):
      ret[a] = b
    for a, b in zip(self.ph_obs_tf, obs_tf):
      ret[a] = b

    ret[self.ph_x_x] = x_x
    return ret

  # load the model and give back a session
  def load_model(self, sess, saved_loc):
    self.saver.restore(sess, saved_loc)
    print("Model restored.")

  # make the model
  def __init__(self, name):
    with tf.variable_scope('hypothesis') as scope:
      # set up placeholders
      self.ph_x_x = tf.placeholder(tf.float32, [N_BATCH, X_L], name="ph_x_x")
      self.ph_obs_x = [tf.placeholder(tf.float32, [N_BATCH, L], 
                  name="ph_ob_x"+str(i)) for i in range(OBS_SIZE)]
      self.ph_obs_y = [tf.placeholder(tf.float32, [N_BATCH, L],
                  name="ph_ob_y"+str(j)) for j in range(OBS_SIZE)]
      self.ph_obs_tf = [tf.placeholder(tf.float32, [N_BATCH, 2],
                  name="ph_ob_tf"+str(k)) for k in range(OBS_SIZE)]

      # some constants
      self.n_hidden = 1200
      self.n_pred_hidden = 1000

      # a list of variables for different tasks
      self.VAR_inv = []

      # ------------------------------------------------------------------ convolve in the observations
      # initial lstm state
      state = tf.zeros([N_BATCH, self.n_hidden])
      # initialize some weights
      # initialize some weights
      # stacked lstm
      lstm = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(300), tf.nn.rnn_cell.LSTMCell(300)])

      hiddens = [state]

    with tf.variable_scope("hypothesis/LSTM") as scope:
      for i in range(OBS_SIZE):
        if i > 0:
          scope.reuse_variables()
        cell_input = tf.concat(1, [self.ph_obs_x[i], self.ph_obs_y[i], self.ph_obs_tf[i]])
        output, state = lstm(cell_input, state)
        hiddens.append(state)

    lstm_variables = [v for v in tf.all_variables()
                        if v.name.startswith("hypothesis/LSTM")]

    print lstm_variables

    self.VAR_inv += lstm_variables


    # -------------------------------------------------------------------- invert to predict hidden X
    with tf.variable_scope('hypothesis') as scope:
      W_inv_x = weight_variable([self.n_hidden,X_L])
      b_inv_x = bias_variable([X_L])

      self.VAR_inv += [W_inv_x, b_inv_x]

      epsilon1 = tf.constant(1e-10, shape=[N_BATCH, X_L])
      self.x_invs = [tf.nn.softmax(tf.matmul(volvoo, W_inv_x) + b_inv_x)+epsilon1 for volvoo in hiddens]
      print "invs shapes ", show_dim(self.x_invs)
      # entropies for the inversion
      entropys = [-tf.reduce_sum(tf.log(x) * x, 1) for x in self.x_invs]

      # compute costs
      inv_costs_x = [-tf.reduce_sum(self.ph_x_x * tf.log(x_pred)) for x_pred in self.x_invs]
      print "costs shapes ", show_dim(inv_costs_x)
      self.cost_inv = sum(inv_costs_x)

      # ------------------------------------------------------------------------ training steps
      optimizer = tf.train.RMSPropOptimizer(0.0001)

      inv_gvs = optimizer.compute_gradients(self.cost_inv, var_list = self.VAR_inv)
      capped_inv_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in inv_gvs]
      #train_inv = optimizer.minimize(cost_inv, var_list = VAR_inv)
      self.train_inv = optimizer.apply_gradients(capped_inv_gvs)
      # Before starting, initialize the variables.  We will 'run' this first.
      self.init = tf.initialize_all_variables()
      self.saver = tf.train.Saver()

  # save the model
  def save(self, sess, model_loc="model_hypothesis.ckpt"):
    save_path = self.saver.save(sess, model_loc)
    print("Model saved in file: %s" % save_path)

  # train on a particular data batch
  def train(self, sess, data_batch, train_inv=False):
    x_x, obs_x, obs_y, obs_tfs, new_ob_x, new_ob_y, new_ob_tf, _ = data_batch
    feed_dic = self.gen_feed_dict(x_x, obs_x, obs_y, obs_tfs)

    # train inversion 
    cost_inv_pre = sess.run([self.cost_inv], feed_dict=feed_dic)[0]
    sess.run([self.train_inv], feed_dict=feed_dic)
    cost_inv_post = sess.run([self.cost_inv], feed_dict=feed_dic)[0]
    print "--------------------------- train inv ",\
      cost_inv_pre, " ", cost_inv_post, " ", True if cost_inv_post < cost_inv_pre else False

  # =========== HELPERS =============

  # a placeholder to feed in a single observation
  def get_feed_dic_obs(self, obs):
    # needing to create all the nessisary feeds
    obs_x = []
    obs_y = []
    obs_tf = []
    
    for _ in range(OBS_SIZE):
      obs_x.append(np.zeros([N_BATCH,L]))
      obs_y.append(np.zeros([N_BATCH,L]))
      obs_tf.append(np.zeros([N_BATCH,2]))

    num_obs = len(obs)
    for ob_idx in range(num_obs):
      ob_coord, ob_lab = obs[ob_idx]
      ob_x, ob_y = vectorize(ob_coord)
      obs_x[ob_idx] = np.tile(ob_x, [50,1])
      obs_y[ob_idx] = np.tile(ob_y, [50,1])
      obs_tf[ob_idx] = np.tile(ob_lab, [50,1])

    feed_dic = dict(zip(self.ph_obs_x + self.ph_obs_y + self.ph_obs_tf, 
                        obs_x + obs_y + obs_tf))
    return feed_dic

  def get_preds_batch(self, sess, obs, batch_querys):
    ret = [[] for _ in range(OBS_SIZE+1)]

    feed_dic = self.get_feed_dic_obs(obs)
    assert len(batch_querys) == N_BATCH

    new_ob_x = []
    new_ob_y = []

    for q in batch_querys:
      q_x, q_y = vectorize(q)
      new_ob_x.append(q_x)
      new_ob_y.append(q_y)

      
    feed_dic[self.ph_new_ob_x] = np.array(new_ob_x)
    feed_dic[self.ph_new_ob_y] = np.array(new_ob_y)

    pred_tfs = sess.run(self.query_preds, feed_dict=feed_dic)
    for key_ob in range(OBS_SIZE+1):
      for q_idx, q in enumerate(batch_querys):
        ret[key_ob].append((q, pred_tfs[key_ob][q_idx]))

    return ret

  def get_all_preds_fast(self, sess, obs):
    all_querys = []
    for i in range(L):
      for j in range(L):
        all_querys.append((i,j))

    def batch_qrys(all_qs):
      ret = []
      while len(all_qs) != 0:
        to_add = [(0,0) for _ in range(N_BATCH)]
        for idk in range(N_BATCH):
          if len(all_qs) == 0:
            break
          to_add[idk] = all_qs.pop()
        ret.append(to_add)
      return ret

    ret = [[] for _ in range(OBS_SIZE+1)]
    batched_qrysss = batch_qrys(all_querys)
    for batched_q in batched_qrysss:
      ppp = self.get_preds_batch(sess, obs, batched_q)
      for ijk in range(OBS_SIZE+1):
        ret[ijk] += ppp[ijk]

    return ret

  def get_most_confuse(self, sess, obs):
    key_ob = len(obs)
    all_preds = self.get_all_ents_fast(sess, obs)
    
    all_pred_at_key = all_preds[key_ob]

    # get rid of already seen things
    # print all_pred_at_key[0]
    # print obs
    # assert 0
    observed_coords = [x[0] for x in obs]
    all_pred_at_key1 = filter(lambda x: x[0] not in observed_coords, all_pred_at_key)
    # print len(all_pred_at_key), " ", len(all_pred_at_key1)

    most_confs = [(abs(x[1][0] - x[1][1]), x[0]) for x in all_pred_at_key1]
    most_conf = min(most_confs)

    return most_conf[1]

  def get_active_inv(self, sess, query, epi=0.0):
    obs = []

    for i in range(OBS_SIZE):
      if np.random.random() < epi:
        rand_coord = sample_coord()
        obs.append((rand_coord, query(rand_coord)))
      else:
        most_conf = self.get_most_confuse(sess, obs)
        obs.append((most_conf, query(most_conf)))

    feed_dic = self.get_feed_dic_obs(obs)
    invs = [(x[0], np.argmax(x[0])) for x in sess.run(self.x_invs, feed_dict=feed_dic)]
    return zip([None] + obs, self.get_all_preds_fast(sess, obs), invs)

  def get_random_inv(self, sess, query):
    ob_pts = [sample_coord_bias(query) for _ in range(OBS_SIZE)]
    obs = [(op, query(op)) for op in ob_pts]
    
    feed_dic = self.get_feed_dic_obs(obs)
    invs = [(x[0], np.argmax(x[0])) for x in sess.run(self.x_invs, feed_dict=feed_dic)]
    return zip([None] + obs, self.get_all_preds_fast(sess, obs), invs)

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



