from model import *
from draw import *
from naive_baseline import *
from solver_bs import *

# ------------- helpers --------------
def pred_acc(preds, qry):                                                              
  num_cor = 0
  for i in range(L):
    for j in range(L):
      if np.argmax(preds[i][j]) == np.argmax(qry((i,j))):
        num_cor += 1 
  return float(num_cor) / L*L     

def pose_pred_acc(p_poses, true_poses):
  num_cor = 0
  for poo in true_poses:
    if poo in p_poses:
      num_cor += 1
  return num_cor

def trace_to_obs(trace):
  ret = []
  for tr in trace:
    cord, ans = tr
    aa = True if ans[0]>ans[1] else False
    ret.append((cord, aa))
  return ret

impnet = Implynet("imp")
sess = tf.Session()
impnet.load_model(sess, "model_imply.ckpt")

# img, _x, poses = get_img_class()
# qry = mk_query(_x) 
# 
# trace = impnet.get_active_trace(sess, qry, epi=0.0)
# rand_trace = rrandom_trace(qry)
# 
# draw_orig(img, "drawings_solver/orig.png")
# for i in range(len(trace)):
#   print "prefix of length ", i
#   trace_prefix = trace[:i]
#   rand_trace_prefix = rand_trace[:i]
# 
#   all_preds = impnet.get_all_preds(sess, trace_prefix)
#   draw_allob(all_preds, "drawings_solver/pred_ob{0}.png".format(i), trace_prefix)
# 
#   print "using constraint to solve "
#   ships_poses = findConfig(boat_shapes, trace_to_obs(trace_prefix))
#   ships_img, _ = det_crack(ships_poses)
#   draw_orig(ships_img, "drawings_solver/constr{0}.png".format(i))
# 
#   print "using constraint to solve random"
#   ships_poses = findConfig(boat_shapes, trace_to_obs(rand_trace_prefix))
#   ships_img, _ = det_crack(ships_poses)
#   draw_orig(ships_img, "drawings_solver/rand_constr{0}.png".format(i))


print "we gonna run on 1000 trials and compare the 4 agents"
num_random = np.array([0.0 for _ in range(L*L)])
num_sink = np.array([0.0 for _ in range(L*L)])
num_oc = np.array([0.0 for _ in range(L*L)])

for _ in range(1, 1000):
  img, _x, true_poses = get_img_class()
  qry = mk_query(_x) 
  
  rand_trace = rrandom_trace(qry)
  sink_trace = baseline_get_trace(qry)
  oc_trace = impnet.get_active_trace(sess, qry, epi=0.0, play=False)

  for i in range(OBS_SIZE):
    rand_prefix = rand_trace[:i]
    sink_prefix = sink_trace[:i]
    oc_prefix = oc_trace[:i]
  
    rand_poses = findConfig(boat_shapes, trace_to_obs(rand_prefix))
    sink_poses = findConfig(boat_shapes, trace_to_obs(sink_prefix))
    oc_poses = findConfig(boat_shapes, trace_to_obs(oc_prefix))

    rand_acc = pose_pred_acc(rand_poses, true_poses)
    num_random[i] += rand_acc
    sink_acc = pose_pred_acc(sink_poses, true_poses)
    num_sink[i] += sink_acc
    oc_acc = pose_pred_acc(oc_poses, true_poses)
    num_oc[i] += oc_acc

  print "iteration ", _
  print "random ", num_random / _
  print "sink ", num_sink / _
  print "oc ", num_oc / _


