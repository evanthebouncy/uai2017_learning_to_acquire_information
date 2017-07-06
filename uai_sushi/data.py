import numpy as np
import random
from scipy.misc import imresize
from scipy.ndimage.filters import gaussian_filter
from tensorflow.examples.tutorials.mnist import input_data
from sort_data import *
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

from numpy.linalg import norm

NOISE = False
NOISE_PR = 0.0

L = 10

X_L = 10
N_BATCH = 30
OBS_SIZE = 100

# ---------------------------- helpers

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
def mk_query(X, noise=False):
  def query(O):
    ii, jj = O

    # generate a condition that is corrupted with noise with some prob
    cond = X.index(ii) < X.index(jj)
    if noise == True and np.random.random() < NOISE_PR:
      cond = not cond

    if cond:
      return [1.0, 0.0]
    else:
      return [0.0, 1.0]
  return query

def sample_coord():
  return np.random.randint(0, L), np.random.randint(0, L) 

def sample_coord_new(obs):
  ret = [(i,j) for i in range(L) for j in range(L)]
  ret = filter(lambda x: x not in [ob[0] for ob in obs], ret)
  return random.choice(ret)

def sample_coord_center():
  Ox, Oy = np.random.multivariate_normal([L/2,L/2], [[L*0.7, 0.0], [0.0, L*0.7]])
  Ox, Oy = round(Ox), round(Oy)
  if 0 <= Ox < L:
    if 0 <= Oy < L:
      return Ox, Oy
  return sample_coord()

def sample_coord_bias(qq):
  def find_positive(qq):
    C = sample_coord()
    if qq(C) == [1.0, 0.0]:
      return C
    else:
      return find_positive(qq)
  def find_negative(qq):
    C = sample_coord()
    if qq(C) == [0.0, 1.0]:
      return C
    else:
      return find_negative(qq)

  toss = np.random.random() < 0.5
  if toss:
    return find_positive(qq)
  else:
    return find_negative(qq)

def gen_O(X):
  query = mk_query(X)
  Ox, Oy = sample_coord()
  O = (Ox, Oy)
  return O, query(O) 

def gen_O_bias(X, hit_bias):
  someO = gen_O(X)
  if np.random.random() < hit_bias:
    if someO[1][0] > 0.5:
      return someO
    else:
      return gen_O_bias(X, hit_bias)
  return someO
     
def perm_2_img(perm):
  qry = mk_query(perm)
  ret = np.zeros([L,L,1])
  for i in range(L):
    for j in range(L):
      ret[i][j] = qry((i,j))[0]
  return ret

def get_img_class(test=False, idx=None):
  if test==False:
    toss = np.random.random() < 0.5
    if toss:
      perm = random.choice(sort_train)
    else:
      perm = list(np.random.permutation(10))
    return perm_2_img(perm), perm
  if test == True:
    if idx != None:
      perm = sort_test[idx]
      print "permy ", perm
      return perm_2_img(perm), perm

# a trace is named tuple
# (Img, S, Os) 
# where Img is the black/white image
# where S is the hidden hypothesis (i.e. label of the img)
# Os is a set of Observations which is (qry_pt, label)
import collections
Trace = collections.namedtuple('Trace', 'Img S Os')

def gen_rand_trace(test=False):
  img, _x = get_img_class(test)
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

def rand_data(epi):
  partial_obss = []
  full_obss = []

  for bb in range(N_BATCH):
    # generate a hidden variable X
    # get a single thing out
    img, _x = get_img_class()
    noisy_qry = mk_query(_x, noise=NOISE)
    pure_qry = mk_query(_x, noise=False)
    partial_obs = np.zeros([L,L,2])
    full_obs = np.zeros([L,L,2])
    for i in range(L):
      for j in range(L):
        full_obs[i][j] = pure_qry((i,j))
        if np.random.random() < epi:
          partial_obs[i][j] = noisy_qry((i,j))
    partial_obss.append(partial_obs)
    full_obss.append(full_obs)

  return  np.array(partial_obss),\
          np.array(full_obss)

def rand_coord():
  xx = np.random.randint(0,L)
  yy = np.random.randint(0,L)
  return np.array([xx, yy])


