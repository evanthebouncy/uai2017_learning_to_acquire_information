from model import *
from draw import *
from naive_baseline import *
import sys

model_type = sys.argv[1]
if model_type == "0layer":
  impnet = Implynet(embed_0_layer, "0layer", tf.Session())
  impnet.load_model("./models/imply_0layer.ckpt")
if model_type == "1layer":
  impnet = Implynet(embed_1_layer, "1layer", tf.Session())
  impnet.load_model("./models/imply_1layer.ckpt")

# ------------- helpers --------------
def pred_acc(preds, qry):                                                              
  num_cor = 0
  for i in range(L):
    for j in range(L):
      if np.argmax(preds[i][j]) == np.argmax(qry((i,j))):
        num_cor += 1 
  return float(num_cor) / L*L     

def get_best_item(preds):
  scores = [0 for i in range(L)]
  for ii in range(L):
    for jj in range(L):
      if len(preds[ii][jj]) == 2:
        if preds[ii][jj][0] > preds[ii][jj][1]:
          scores[ii] += 1
      if len(preds[ii][jj]) == 1:
        scores[ii] += preds[ii][jj]
  return np.argmax(scores)
  

img, _x = get_img_class(test=True, idx=0)
draw_orig(img, "drawings/orig.png")
qry = mk_query(_x) 

trace = impnet.get_active_trace(qry, epi=0.0)

for i in range(len(trace)):
  trace_prefix = trace[:i]
  all_preds = impnet.get_all_preds(trace_prefix)
  draw_allob(all_preds, "drawings/pred_ob{0}.png".format(i), trace_prefix)

print "we gonna run on 2500 test data"
num_active = np.array([0.0 for _ in range(L*L)])

for _ in range(2500):
  img, _x = get_img_class(test=True, idx=_)
  noisy_qry = mk_query(_x, noise=NOISE)
  pure_qry = mk_query(_x, noise=False)
  active_trace = impnet.get_active_trace(noisy_qry, epi=0.0, play=False)

  for i in range(len(active_trace)):
    trace_prefix = active_trace[:i]
    all_preds = impnet.get_all_preds(trace_prefix)
    acti_acc = pred_acc(all_preds, pure_qry)
    num_active[i] += acti_acc

  print "iteration ", _
  print impnet.name + " ", num_active / (_ +1)



