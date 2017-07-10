import numpy as np
import random
from scipy.misc import imresize
from scipy.ndimage.filters import gaussian_filter
from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

from numpy.linalg import norm

L = 10
NOISE = 0.1

X_L = 10
N_BATCH = 30
OBS_SIZE = L * L


boat_shapes = [(2,4), (1,5), (1,3), (1,3), (1,3)]

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

def corrupt(o):
  if o == [1.0, 0.0]: return [0.0, 1.0]
  if o == [0.0, 1.0]: return [1.0, 0.0]

# -------------------------------------- making the datas

# assume X is already a 2D matrix
def mk_query(X):
  # give a probability of noise corrupting the observation
  def query(O):
    def _query(O):
      for xx in X:
        # print O, xx, dist(xx, O)
        if dist(xx, O) < 1:
          return [1.0, 0.0]
      return [0.0, 1.0]
    o = _query(O)
    if np.random.random() < NOISE:
      return corrupt(o)
    else:
      return o
  return query

def sample_coord():
  return np.random.randint(0, L), np.random.randint(0, L) 

def gen_O(X):
  query = mk_query(X)
  Ox, Oy = sample_coord()
  O = (Ox, Oy)
  return O, query(O) 

def get_img_class(test=False):
  total_mass = sum([x[0]*x[1] for x in boat_shapes])
  def _gen_boats():
    ret = np.zeros([L, L])
    done = []
    poses = []

    joint_cstr = []
    for b_sh in boat_shapes:
      crd = rand_coord()
      wh,d = rand_orient(*b_sh)
      joint_cstr.append(rect_constr(crd, wh))
      poses.append((crd[0],crd[1],d))

    joint_constr = or_constr(joint_cstr)
    for i in range(L):
      for j in range(L):
        if joint_constr((i,j)):
          done.append((i,j))
          ret[i][j] = 1.0

    return ret, done, poses

  ret, done, poses = _gen_boats()
  if len(done) == total_mass:
    return ret, done, poses
  else:
    return get_img_class(test)

def rand_orient(w,h):
  if np.random.random() < 0.5:
    return (w,h),True
  else:
    return (h,w),False



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

# def data_from_exp(exp, epi):
#   traces = [exp.sample() for _ in range(N_BATCH)]
#   x = []
#   
#   obs_x = [[] for i in range(OBS_SIZE)]
#   obs_tfs = [[] for i in range(OBS_SIZE)]
#   new_ob_x = []
#   new_ob_tf = []
# 
#   imgs = []
# 
#   for bb in range(N_BATCH):
#     trr = traces[bb]
#     # generate a hidden variable X
#     # get a single thing out
#     img = trr.Img
#     _x = trr.S 
#     imgs.append(img)
# 
#     x.append(_x)
#     # generate a FRESH new observation for demanding an answer
#     _new_ob_coord, _new_ob_lab = gen_O(_x)
#     new_ob_x.append(vectorize_flat(_new_ob_coord))
#     new_ob_tf.append(_new_ob_lab)
# 
#     # generate observations for this hidden variable x
#     for ob_idx in range(OBS_SIZE):
#       _ob_coord, _ob_lab = trr.Os[ob_idx]
#       obs_x[ob_idx].append(vectorize_flat(_ob_coord))
#       obs_tfs[ob_idx].append(_ob_lab)
# 
#   return  None,\
#           np.array(obs_x, np.float32),\
#           np.array(obs_tfs, np.float32),\
#           np.array(new_ob_x, np.float32),\
#           np.array(new_ob_tf, np.float32), imgs
# 

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
    img, _x, _pose = get_img_class()
    qry = mk_query(_x)
    partial_obs = np.zeros([L,L,2])
    full_obs = np.zeros([L,L,2])
    for i in range(L):
      for j in range(L):
        full_obs[i][j] = qry((i,j))
        if np.random.random() < epi:
          partial_obs[i][j] = qry((i,j))
    partial_obss.append(partial_obs)
    full_obss.append(full_obs)

  return  np.array(partial_obss),\
          np.array(full_obss)

def rand_coord():
  xx = np.random.randint(0,L)
  yy = np.random.randint(0,L)
  return np.array([xx, yy])


def rect_constr(left_top, wid_hei):
  left, top = left_top
  wid, hei = wid_hei
  right, down = left + wid, top+hei
  def constr(crd):
    xx, yy = crd
    in_e1 = xx >= left
    in_e2 = xx < right
    in_e3 = yy >= top
    in_e4 = yy < down
    return in_e1 and in_e2 and in_e3 and in_e4
  return constr

def not_constr(cr):
  def constr(crd):
    return not cr(crd)
  return constr

def and_constr(crs):
  def constr(crd):
    for cr in crs:
      if not cr(crd):
        return False
    return True
  return constr

def or_constr(crs):
  def constr(crd):
    for cr in crs:
      if cr(crd):
        return True
    return False
  return constr


def game_end(obs):
  total_mass = sum([x[0]*x[1] for x in boat_shapes])
  seen_pos = [o for o in obs if o[1][0] > o[1][1]]  
  ret = len(seen_pos) == total_mass
  return ret

def det_crack(ship_poses):
  ret = np.zeros([L, L])
  done = []

  joint_cstr = []
  for i,b_sh in enumerate(boat_shapes):
    x,y,d = ship_poses[i]
    crd = x,y
    wh = det_orient(d,*b_sh)
    joint_cstr.append(rect_constr(crd, wh))

  joint_constr = or_constr(joint_cstr)
  for i in range(L):
    for j in range(L):
      if joint_constr((i,j)):
        done.append((i,j))
        ret[i][j] = 1.0

  return ret, done


