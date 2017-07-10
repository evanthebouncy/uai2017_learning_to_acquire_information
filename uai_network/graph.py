import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import copy

N = 100

def dist(c1, c2):
  return np.linalg.norm(np.array(c1) - np.array(c2))

def too_close(pt, pts):
  for pt_other in pts:
    if dist(pt, pt_other) < 0.05:
      return True
  return False
  
def gen_pts(n):
 
  ret = []
  while len(ret) < N:
    new_pt = (np.random.random(), np.random.random())
    if not too_close(new_pt, ret):
      ret.append(new_pt)
  return ret

def gen_spanning_tree(n_vertex):
  nnn = len(n_vertex)
  ret = [[0 for i in range(nnn)] for j in range(nnn)]

  def get_closest(n_set, nodey):
    dists = [(dist(nss[1], nodey[1]), nss) for nss in n_set]
    return min(dists)[1]

  connected_set = []
  n_v = zip(range(nnn), n_vertex)
  while len(connected_set) < nnn:
    # pick random
    rand_node = random.choice(n_v)
    n_v.remove(rand_node)
    if connected_set == []:
      connected_set.append(rand_node)
    else:
      closest = get_closest(connected_set, rand_node)
      ret[closest[0]][rand_node[0]] = 1
      ret[rand_node[0]][closest[0]] = 1
      connected_set.append(rand_node)
 
  return ret

def gen_graph(n_vertex, c):
  n = len(n_vertex)
  ret = [[0 for i in range(n)] for j in range(n)]
  for i in range(n):
    for j in range(i):
      crd_i, crd_j = n_vertex[i], n_vertex[j]
      if np.random.random() < np.power(2, -c * dist(crd_i, crd_j)) or\
         dist(crd_i, crd_j) < 0.3:
        ret[i][j] = 1
        ret[j][i] = 1
  return ret

def get_blob(blobs, node):
  for blob in blobs:
    if node in blob:
      return blob
  return None

# return a map from node to components
def get_ccomp(ge):
  blobs = []
  for iii in range(N):
    for jjj in range(N):
      if ge[iii][jjj] == 1 or iii == jjj:
        blob_i = get_blob(blobs, iii)
        blob_j = get_blob(blobs, jjj)
        if blob_i == None and blob_j == None:
          blobs.append(set([iii,jjj]))
        if blob_i != None and blob_j == None:
          blob_i.add(jjj)
        if blob_i == None and blob_j != None:
          blob_j.add(iii)
        if blob_i != None and blob_j != None and blob_i != blob_j:
          blobs.remove(blob_i)
          blobs.remove(blob_j)
          blobs.append(blob_i.union(blob_j))
  ret = dict()
  for i, blob in enumerate(blobs):
    for bb in blob:
      ret[bb] = i
  return ret

def get_shortest_path(ge, node_i):
  fringe = [node_i]
  seen = set()
  dists = {}
  for _ in range(N):
    dists[_] = 999

  cur_hop = 0
  while fringe != []:
    cur_nodes = fringe
    seen.update(cur_nodes)
    fringe = []
    for c_n in cur_nodes:
      dists[c_n] = cur_hop
      for other_j in range(N):
        if ge[c_n][other_j] and other_j not in seen:
          fringe.append(other_j) 
    fringe = list(set(fringe))
    # print fringe
    cur_hop += 1
  return dists

def gen_obs_links(ge):
  ret = []
  for iii in range(N):
    for jjj in range(iii):
      if ge[iii][jjj] == 1:
        ret.append((iii,jjj))
  for i in range(300):
    pick_x = random.randint(0, N-1)
    pick_y = random.randint(0, N-1)
    ret.append((pick_x, pick_y))
  return ret

def get_shortest_paths(ge):
  return [get_shortest_path(ge, i) for i in range(N)]

def random_fail(ge, prob=0.02):
  ret = copy.deepcopy(ge)
  link_fail = []
  for i in range(N):
    for j in range(i):
      if np.random.random() < prob and ge[i][j] == 1:
        ret[i][j] = 0
        ret[j][i] = 0
        link_fail.append((i,j))
  return ret, link_fail 

def path_changed(ge, ge_fail, G_OBS):
  ret = []
#   sp_ge = get_shortest_paths(ge)
#   sp_ge_fail = get_shortest_paths(ge_fail)

#  ccp_ge = get_ccomp(ge)
  ccp_ge_fail = get_ccomp(ge_fail)

  for test_pair in G_OBS:
    i, j = test_pair
    if ccp_ge_fail[i] != ccp_ge_fail[j]:
      ret.append([1.0, 0.0])
    else:
      ret.append([0.0, 1.0])
#     if sp_ge[i][j] != sp_ge_fail[i][j]:
#       ret.append([1.0, 0.0])
#     else:
#       ret.append([0.0, 1.0])
  return ret  

def draw_graph(gv, ge, name):
  Gr = nx.Graph()
  for i in range(N):
    Gr.add_node(i, pos=gv[i])

  for i in range(N):
    for j in range(N):
      if ge[i][j]:
        Gr.add_edge(i,j)

  labels = dict()
  for i in range(N):
    labels[i] = str(i)

  pos=nx.get_node_attributes(Gr,'pos')

  nx.draw(Gr, pos=pos, 
      node_size=400, with_labels=False)
  nx.draw_networkx_labels(Gr, pos, labels)

  plt.savefig(name)

def get_dependency_coef(ge, link):
  broken_graph = copy.deepcopy(ge)
  iii, jjj = link
  broken_graph[iii][jjj] = 0
  broken_graph[jjj][iii] = 0
  
  comps = get_ccomp(broken_graph)
  halfs = dict()
  for blah in comps:
    if comps[blah] not in halfs:
      halfs[comps[blah]] = 1
    else:
      halfs[comps[blah]] += 1

  M = float(halfs[0])
  N = float(halfs[1])

  return (M * N) / ( (M + N) * (M + N - 1) )

# V = gen_pts(N)
# # G = gen_graph(V, 6)
# G = gen_spanning_tree(V)
# 
# G_V = V
# G_E = G
# 
# print G_V
# print G_E
# 
# Gr = nx.Graph()
# for i in range(N):
#   Gr.add_node(i, pos=G_V[i])
# 
# for i in range(N):
#   for j in range(N):
#     if G_E[i][j]:
#       Gr.add_edge(i,j)
# 
# labels = dict()
# for i in range(N):
#   labels[i] = str(i)
# 
# pos=nx.get_node_attributes(Gr,'pos')
# 
# nx.draw(Gr, pos=pos, 
#     node_size=250, with_labels=False)
# nx.draw_networkx_labels(Gr, pos, labels)
# 
# plt.savefig("graph.png")
