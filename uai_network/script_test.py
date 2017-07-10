from model import *
from draw import *
from naive_baseline import *

# ------------- helpers --------------
# some on-policy data generation
def full_output_to_trace(full_out, Img, S):
  obs = [x[0] for x in full_out[1:]]
  return Trace(Img, S, obs)

def is_correct(path_chg, all_preds):
  n_cor, n_inc = 0, 0
  for iii in range(N):
    if np.argmax(path_chg[iii]) == np.argmax(all_preds[iii]):
      n_cor += 1
    else:
      n_inc += 1
  return n_cor, n_inc

def link_correct(path_chg, all_preds, particular_link):
  link_idx = G_OBS.index(particular_link)
  return np.argmax(path_chg[link_idx]) == np.argmax(all_preds[link_idx]) 

def accuracy(path_chg, all_preds):
  n_cor, n_inc = is_correct(path_chg, all_preds)
  return float(n_cor) / (n_cor + n_inc)

def get_belief(link_fail, all_preds):
  ret = []
  for iii in range(N):
    for jjj in range(iii):
      if link_fail[iii][jjj] != 0:
        ret.append((all_preds[iii][jjj][0], (iii,jjj)))
  ret.sort()
  return ret

links_fail, _x, ge_fail = get_instance()
print "links failure: "
print links_fail
qry = mk_query(_x) 

draw_graph(G_V, ge_fail, "drawings/graph_fail.png")
# draw_orig(img, "drawings/link_fail.png")

impnet = Implynet("imp")
sess = tf.Session()
impnet.load_model(sess, "model_imply.ckpt")


# trace = impnet.get_active_trace(sess, qry, epi=0.0)
# rand_trace = rrandom_trace(qry)
# 
# for i in range(N):
#   trace_prefix = trace[:i]
#   rand_trace_prefix = rand_trace[:i]
#   all_preds = impnet.get_all_preds(sess, trace_prefix)
#   all_rand_preds = baseline_get_all_preds(rand_trace_prefix)
#   n_cor, n_inc = is_correct(_x, all_preds)
#   print i, n_cor, n_inc, float(n_cor) / (n_cor + n_inc)
# 
#   n_cor, n_inc = is_correct(_x, all_rand_preds)
#   print i, n_cor, n_inc, float(n_cor) / (n_cor + n_inc)
#          
#   print "------------------------------------", G_OBS[trace[i][0]], trace[i], "made ob"
# 

acc_rand = np.array([0.0 for _ in range(OBS_SIZE)])
acc_oc = np.array([0.0 for _ in range(OBS_SIZE)])

link_stats = dict()

for test_id in range(1,10000+1):
  print "iteration ", test_id
  links_fail, path_chg, ge_fail = get_instance()
  print "links failure: "
  print links_fail
  qry = mk_query(path_chg) 

  rand_trace = rrandom_trace(qry)
  oc_trace = impnet.get_active_trace(sess, qry, epi=0.0)

  for i in range(OBS_SIZE):
    oc_prefix = oc_trace[:i]
    rand_prefix = rand_trace[:i]

    rand_all_preds = baseline_get_all_preds(rand_prefix)
    oc_all_preds = impnet.get_all_preds(sess, oc_prefix)

    a_rand = accuracy(path_chg, rand_all_preds)
    a_oc = accuracy(path_chg, oc_all_preds)

    acc_rand[i] += a_rand
    acc_oc[i] += a_oc

    if i == 50:
      for fail_l in links_fail:
        is_cor = link_correct(path_chg, oc_all_preds, fail_l)
        if fail_l not in link_stats:
          link_stats[fail_l] = [0,0]
        if is_cor:
          link_stats[fail_l][0] += 1
        else:
          link_stats[fail_l][1] += 1

  print "rand accuracy "
  print acc_rand / test_id
  print "oc accuracy "
  print acc_oc / test_id
  print "link fails "
  print link_stats
