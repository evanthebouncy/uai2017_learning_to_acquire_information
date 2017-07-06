# from model import *
from draw import *
from naive_baseline import *
from quicksort import *

# ------------- helpers --------------
def get_id_map(start_sort, truth):
  ret = dict(zip(start_sort, truth))
  return ret

def pred_acc(preds, qry):                                                              
  num_cor = 0
  for i in range(L):
    for j in range(L):
      if np.argmax(preds[i][j]) == np.argmax(qry((i,j))):
        num_cor += 1 
  return float(num_cor) / L*L     

def ord_2_pred(ordr):
  ret = np.zeros([L,L,2])
  for i in range(L):
    for j in range(L):
      if ordr.index(i) < ordr.index(j):
        ret[i][j] = [1.0, 0.0]
      else:
        ret[i][j] = [0.0, 1.0]
  return ret
         

num_sort = np.array([0.0 for _ in range(L*L)])

for _ in range(2500):
  img, _x = get_img_class(test=True, idx=_)
  qry = mk_query(_x) 

  start_sort = np.random.permutation(L)
  id_mapping = get_id_map(range(L), _x)

  # merge sort
  # trace = sortb(start_sort)
  # bubble sort
  trace = sortc(start_sort)
  # quicksort
  # trace = sorta(start_sort)

  print "truth"
  print _x
  for idx, blah in enumerate(trace):
    trace[idx] = map(lambda x: id_mapping[x], blah)

  for i in range(L*L):
    tr = trace[i] if i < len(trace) else trace[-1]
    preds = ord_2_pred(tr)
    num_sort[i] += pred_acc(preds, qry)

  print num_sort / (_ + 1)


