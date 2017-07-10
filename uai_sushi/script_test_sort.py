# from model import *
from draw import *
from naive_baseline import *
from quicksort import *
from copy import deepcopy

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
         

num_sort_bubble = np.array([0.0 for _ in range(L*L)])
num_sort_quick = np.array([0.0 for _ in range(L*L)])
num_sort_merge = np.array([0.0 for _ in range(L*L)])

for _ in range(2500):
  img, _x = get_img_class(test=True, idx=_)
  qry = mk_query(_x) 

  start_sort = np.random.permutation(L)
  start_sort1 = deepcopy(start_sort)
  start_sort2 = deepcopy(start_sort)
  id_mapping = get_id_map(range(L), _x)

  bubble_trace = sortc(start_sort)
  quick_trace = sorta(start_sort1)
  merge_trace = sortb(start_sort2)


  print "truth"
  print _x
  for idx, blah in enumerate(bubble_trace):
    bubble_trace[idx] = map(lambda x: id_mapping[x], blah)
  for idx, blah in enumerate(quick_trace):
    quick_trace[idx] = map(lambda x: id_mapping[x], blah)
  for idx, blah in enumerate(merge_trace):
    merge_trace[idx] = map(lambda x: id_mapping[x], blah)

  for i in range(L*L):
    bubble_tr = bubble_trace[i] if i < len(bubble_trace) else bubble_trace[-1]
    bubble_preds = ord_2_pred(bubble_tr)
    num_sort_bubble[i] += pred_acc(bubble_preds, qry)

    quick_tr = quick_trace[i] if i < len(quick_trace) else quick_trace[-1]
    quick_preds = ord_2_pred(quick_tr)
    num_sort_quick[i] += pred_acc(quick_preds, qry)

    merge_tr = merge_trace[i] if i < len(merge_trace) else merge_trace[-1]
    merge_preds = ord_2_pred(merge_tr)
    num_sort_merge[i] += pred_acc(merge_preds, qry)

  print " bubble sort ", num_sort_bubble / (_ + 1)
  print " quick sort ", num_sort_quick / (_ + 1)
  print " merge sort ", num_sort_merge / (_ + 1)


