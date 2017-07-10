from model import *
from draw import *
from naive_baseline import *
from test_data import *
import sys

model_type = sys.argv[1]
if model_type == "0layer":
  impnet = Implynet(embed_0_layer, "0layer", tf.Session())
  impnet.load_model("./models/imply_0layer.ckpt")
if model_type == "1layer":
  impnet = Implynet(embed_1_layer, "1layer", tf.Session())
  impnet.load_model("./models/imply_1layer.ckpt")
if model_type == "cnnlayer":
  impnet = Implynet(embed_cnn_layer, "cnnlayer", tf.Session())
  impnet.load_model("./models/imply_cnnlayer.ckpt")

# ------------- helpers --------------
def pred_acc(preds, qry):                                                              
  num_cor = 0
  for i in range(L):
    for j in range(L):
      if np.argmax(preds[i][j]) == np.argmax(qry((i,j))):
        num_cor += 1 
  return float(num_cor) / L*L     


img, _x, _poss = get_img_class()
qry = mk_query(_x) 

trace = impnet.get_active_trace(qry, epi=0.0)

draw_orig(img, "drawings/orig.png")
for i in range(len(trace)):
  trace_prefix = trace[:i]
  all_preds = impnet.get_all_preds(trace_prefix)
  draw_allob(all_preds, "drawings/pred_ob{0}.png".format(i), trace_prefix)

print "we gonna run on 1000 trials and compare the 4 agents"
num_random = np.array([0.0 for _ in range(L*L)])
num_sink = np.array([0.0 for _ in range(L*L)])
num_oc = np.array([0.0 for _ in range(L*L)])

iter_num = 0

for test_d in test_data:
  iter_num += 1
  img, _x, _pos = test_d
  qry = mk_query(_x) 
  
  rand_trace = rrandom_trace(qry)
  #print rand_trace
  sink_trace = baseline_get_trace(qry)
  oc_trace = impnet.get_active_trace(qry, epi=0.0, play=False)

  for i in range(OBS_SIZE):
    rand_prefix = rand_trace[:i]
    sink_prefix = sink_trace[:i]
    oc_prefix = oc_trace[:i]

    rand_all_p = baseline_get_all_preds(rand_prefix)
    sink_all_p = baseline_get_all_preds(sink_prefix)
    oc_all_p = impnet.get_all_preds(oc_prefix)

    rand_acc = pred_acc(rand_all_p, qry)
    num_random[i] += rand_acc / 100
    sink_acc = pred_acc(sink_all_p, qry)
    num_sink[i] += sink_acc / 100
    oc_acc = pred_acc(oc_all_p, qry)
    num_oc[i] += oc_acc / 100

  print "iteration ", iter_num
  print "random ", num_random / iter_num
  print "sink ", num_sink / iter_num
  print "oc "+impnet.name+" " , num_oc / iter_num



