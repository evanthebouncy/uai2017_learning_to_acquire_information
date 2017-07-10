from data import *
import random

def baseline_get_all_preds(trace_prefix):
  ret = np.zeros([L,2])
  for i in range(L):
    ret[i] = [0.0, 1.0]
  for ob in trace_prefix:
    crd, ans = ob
    ret[crd] = ans
  return ret 

def b_random_next_move(obs):
  all_crds = [i for i in range(N)]
#  print all_crds
#  print obs
  for crd in [o[0] for o in obs]:
    all_crds.remove(crd)
  return random.choice(all_crds)

def rrandom_trace(qry):
  obs = []
  for _ in range(N):
    qry_pt = b_random_next_move(obs)
    answer = qry(qry_pt)
    obs.append((qry_pt, answer))
  return obs

