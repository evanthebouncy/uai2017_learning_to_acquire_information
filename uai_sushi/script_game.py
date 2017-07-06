from model import *
from draw import *
from naive_baseline import *

# ------------- helpers --------------
# some on-policy data generation
def full_output_to_trace(full_out, Img, S):
  obs = [x[0] for x in full_out[1:]]
  return Trace(Img, S, obs)

def game_winning_round(trace):
  for i in range(len(trace)):
    trace_prefix = trace[:i]
    if game_end(trace_prefix):
      return i
  assert 0, "run game longer until terminate plz"

impnet = Implynet("imp")
sess = tf.Session()
impnet.load_model(sess, "model_imply.ckpt")

img, _x = get_img_class()
qry = mk_query(_x) 


draw_orig(img, "game_drawings/orig.png")

baseline_trace = baseline_get_trace(qry)
for i in range(len(baseline_trace)):
  trace_prefix = baseline_trace[:i]
  all_preds = baseline_get_all_preds(trace_prefix)
  draw_allob(all_preds, "game_drawings/baseline{0}.png".format(i), trace_prefix)

print "baseline ", game_winning_round(baseline_trace)

active_trace = impnet.get_active_trace(sess, qry, epi=0.0, play=True)
for i in range(len(active_trace)):
  trace_prefix = active_trace[:i]
  all_preds = impnet.get_all_preds(sess, trace_prefix)
  draw_allob(all_preds, "game_drawings/active{0}.png".format(i), trace_prefix)

print "active ", game_winning_round(active_trace)

print "we gonna run on 1000 trials and compare the 2 agents"
num_baseline = 0
num_active = 0

for _ in range(1, 1000):
  img, _x = get_img_class()
  qry = mk_query(_x) 
  baseline_trace = baseline_get_trace(qry)
  active_trace = impnet.get_active_trace(sess, qry, epi=0.0, play=True)
  num_baseline += game_winning_round(baseline_trace)
  num_active += game_winning_round(active_trace)
  print "baseline ", float(num_baseline) / _, "active ", float(num_active) / _


