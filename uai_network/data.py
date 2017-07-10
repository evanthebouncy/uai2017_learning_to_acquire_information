import numpy as np
import random
from scipy.misc import imresize
from scipy.ndimage.filters import gaussian_filter
from tensorflow.examples.tutorials.mnist import input_data
from saved_graph import *

from numpy.linalg import norm

L = len(G_OBS)

N_BATCH = 30
OBS_SIZE = 100


# ---------------------------- helpers

def dist(v1, v2):
  diff = np.array(v1) - np.array(v2)
  return np.dot(diff, diff)

def vectorize(coords):
  retX, retY = np.zeros([L]), np.zeros([L])
  retX[coords[0]] = 1.0
  retY[coords[1]] = 1.0
  return retX, retY

def vectorize_flat(coords):
  ret = np.zeros([L*L])
  ret[coords[0]*L + coords[1]] = 1.0
  return ret


# show dimension of a data object (list of list or a tensor)
def show_dim(lst1):
  if hasattr(lst1, '__len__') and len(lst1) > 0:
    return [len(lst1), show_dim(lst1[0])]
  else:
    try:
      return lst1.get_shape()
    except:
      try:
        return lst1.shape
      except:
        return type(lst1)

# -------------------------------------- making the datas

# assume X is already a 2D matrix
def mk_query(X):
  def query(O):
    return X[O]
  return query

def sample_coord():
  return np.random.randint(0, L), np.random.randint(0, L) 

def sample_coord_new(obs):
  ret = [i for i in range(L)]
  ret = filter(lambda x: x not in [ob[0] for ob in obs], ret)
  return random.choice(ret)

def gen_O(X):
  query = mk_query(X)
  O = random.randint(0, len(X))
  return O, query(O) 
     

def get_instance(test=False):
  ge_fail, link_fail = random_fail(G_E)
  path_chg = path_changed(G_E, ge_fail, G_OBS)
  

  return link_fail, path_chg, ge_fail 
  # img, _x = gen_crack()
  # img = gaussian_filter(img, 1.0)
  # return img, _x

# a trace is named tuple
# (Img, S, Os) 
# where Img is the black/white image
# where S is the hidden hypothesis (i.e. label of the img)
# Os is a set of Observations which is (qry_pt, label)
import collections
Trace = collections.namedtuple('Trace', 'Img S Os')

def gen_rand_trace(test=False):
  img, _x = get_instance(test)
  obs = []
  for ob_idx in range(OBS_SIZE):
    obs.append(gen_O(_x))
  return Trace(img, _x, obs)

# a class to hold the experiences
class Experience:
  
  def __init__(self, buf_len):
    self.buf = []
    self.buf_len = buf_len

  def trim(self):
    while len(self.buf) > self.buf_len:
      self.buf.pop()

  def add(self, trace):
    self.buf.append(trace)
    self.trim()
  
  def sample(self):
    idxxs = np.random.choice(len(self.buf), size=1, replace=False)
    return self.buf[idxxs[0]]

def data_from_exp(exp, epi):
  traces = [exp.sample() for _ in range(N_BATCH)]
  x = []
  
  obs_x = [[] for i in range(OBS_SIZE)]
  obs_tfs = [[] for i in range(OBS_SIZE)]
  new_ob_x = []
  new_ob_tf = []

  imgs = []

  for bb in range(N_BATCH):
    trr = traces[bb]
    # generate a hidden variable X
    # get a single thing out
    img = trr.Img
    _x = trr.S 
    imgs.append(img)

    x.append(_x)
    # generate a FRESH new observation for demanding an answer
    _new_ob_coord, _new_ob_lab = gen_O(_x)
    new_ob_x.append(vectorize_flat(_new_ob_coord))
    new_ob_tf.append(_new_ob_lab)

    # generate observations for this hidden variable x
    for ob_idx in range(OBS_SIZE):
      _ob_coord, _ob_lab = trr.Os[ob_idx]
      obs_x[ob_idx].append(vectorize_flat(_ob_coord))
      obs_tfs[ob_idx].append(_ob_lab)

  return  None,\
          np.array(obs_x, np.float32),\
          np.array(obs_tfs, np.float32),\
          np.array(new_ob_x, np.float32),\
          np.array(new_ob_tf, np.float32), imgs

# the thing is we do NOT use the trace observations, we need to generate random observations
# to be sure we can handle all kinds of randomizations
def inv_data_from_label_data(labelz, inputz):
  labs = []
  obss = []

  for bb in range(N_BATCH):
    img = inputz[bb]
    lab = labelz[bb]
    labs.append(lab)

    obs = np.zeros([L,L,2])
    # generate observations for this hidden variable x
    for ob_idx in range(OBS_SIZE):
      ob_coord, ob_lab = gen_O(img)
      ox, oy = ob_coord
      if ob_lab[0] == 1.0:
        obs[ox][oy][0] = 1.0
      if ob_lab[1] == 1.0:
        obs[ox][oy][1] = 1.0
    obss.append(obs)
  return  np.array(labs, np.float32),\
          np.array(obss, np.float32)

# uses trace info
def inv_batch_obs(labz, batch_Os):
  obss = []

  for bb in range(N_BATCH):
    Os = batch_Os[bb]
    obs = np.zeros([L,L,2])
    # generate observations for this hidden variable x
    for ob_idx in range(OBS_SIZE):
      ob_coord, ob_lab = Os[ob_idx]
      ox, oy = ob_coord
      if ob_lab[0] == 1.0:
        obs[ox][oy][0] = 1.0
      if ob_lab[1] == 1.0:
        obs[ox][oy][1] = 1.0
    obss.append(obs)
  return  np.array(labz, np.float32),\
          np.array(obss, np.float32)

def rand_data(epi):
  partial_obss = []
  full_obss = []

  for bb in range(N_BATCH):
    # generate a hidden variable X
    # get a single thing out
    img, _x, _ = get_instance()
    qry = mk_query(_x)
    partial_obs = np.zeros([L,2])
    full_obs = np.zeros([L,2])
    for i in range(L):
      full_obs[i] = qry(i)
      if np.random.random() < epi:
        partial_obs[i] = qry(i)
    partial_obss.append(partial_obs)
    full_obss.append(full_obs)

  return  np.array(partial_obss),\
          np.array(full_obss)
