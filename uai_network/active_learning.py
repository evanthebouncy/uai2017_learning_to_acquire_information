from model import *

# ------------------------------------------------------------------ helpers and mgmt
def get_feed_dic_obs(obs):
  # needing to create all the nessisary feeds
  obs_x = []
  obs_y = []
  obs_tf = []
  
  for _ in range(OBS_SIZE):
    obs_x.append(np.zeros([N_BATCH,L]))
    obs_y.append(np.zeros([N_BATCH,L]))
    obs_tf.append(np.zeros([N_BATCH,2]))

  num_obs = len(obs)
  for ob_idx in range(num_obs):
    ob_coord, ob_lab = obs[ob_idx]
    ob_x, ob_y = vectorize(ob_coord)
    obs_x[ob_idx] = np.tile(ob_x, [50,1])
    obs_y[ob_idx] = np.tile(ob_y, [50,1])
    obs_tf[ob_idx] = np.tile(ob_lab, [50,1])

  feed_dic = dict(zip(ph_obs_x + ph_obs_y + ph_obs_tf, 
                      obs_x + obs_y + obs_tf))
  return feed_dic

def get_inv(sess, obs):
  num_obs = len(obs)
  return get_inv_tr(sess, obs)[num_obs]

def get_all_preds(sess, obs):
  feed_dic = get_feed_dic_obs(obs)
  all_querys = []
  for i in range(L):
    for j in range(L):
      all_querys.append((i,j))

  ret = [[] for _ in range(OBS_SIZE+1)]

  for q in all_querys:
    q_x, q_y = vectorize(q)
    feed_dic[ph_new_ob_x] = np.tile(q_x, [N_BATCH,1])
    feed_dic[ph_new_ob_y] = np.tile(q_y, [N_BATCH,1])
    pred_tfs = sess.run(query_preds, feed_dict=feed_dic)
    for key_ob in range(OBS_SIZE+1):
      ret[key_ob].append((q, pred_tfs[key_ob][0]))

  return ret

def get_preds_batch(sess, obs, batch_querys):
  ret = [[] for _ in range(OBS_SIZE+1)]

  feed_dic = get_feed_dic_obs(obs)
  assert len(batch_querys) == N_BATCH

  new_ob_x = []
  new_ob_y = []

  for q in batch_querys:
    q_x, q_y = vectorize(q)
    new_ob_x.append(q_x)
    new_ob_y.append(q_y)

    
  feed_dic[ph_new_ob_x] = np.array(new_ob_x)
  feed_dic[ph_new_ob_y] = np.array(new_ob_y)

  pred_tfs = sess.run(query_preds, feed_dict=feed_dic)
  for key_ob in range(OBS_SIZE+1):
    for q_idx, q in enumerate(batch_querys):
      ret[key_ob].append((q, pred_tfs[key_ob][q_idx]))

  return ret
  

def get_all_preds_fast(sess, obs):
  all_querys = []
  for i in range(L):
    for j in range(L):
      all_querys.append((i,j))

  def batch_qrys(all_qs):
    ret = []
    while len(all_qs) != 0:
      to_add = [(0,0) for _ in range(N_BATCH)]
      for idk in range(N_BATCH):
        if len(all_qs) == 0:
          break
        to_add[idk] = all_qs.pop()
      ret.append(to_add)
    return ret

  ret = [[] for _ in range(OBS_SIZE+1)]
  batched_qrysss = batch_qrys(all_querys)
  for batched_q in batched_qrysss:
    ppp = get_preds_batch(sess, obs, batched_q)
    for ijk in range(OBS_SIZE+1):
      ret[ijk] += ppp[ijk]

  return ret

def get_most_confuse(sess, obs):
  key_ob = len(obs)
  all_preds = get_all_preds_fast(sess, obs)
  
  all_pred_at_key = all_preds[key_ob]

  most_confs = [(abs(x[1][0] - x[1][1]), x[0]) for x in all_pred_at_key]
  most_conf = min(most_confs)

  return most_conf[1]

def get_random_inv(sess, query, OBS_SIZE=OBS_SIZE):
  ob_pts = [sample_coord_bias(query) for _ in range(OBS_SIZE)]
  obs = [(op, query(op)) for op in ob_pts]
  
  feed_dic = get_feed_dic_obs(obs)
  invs = [(x[0], np.argmax(x[0])) for x in sess.run(x_invs, feed_dict=feed_dic)]
  return zip([None] + obs, get_all_preds(sess, obs), invs)

def get_random_inv_direct(sess, query):
  ob_pts = [(np.random.randint(0, L), np.random.randint(0,L)) for _ in range(OBS_SIZE)]
  obs = [(op, query(op)) for op in ob_pts]
  ret =  sess.run([inv_s1_xs[OBS_SIZE],
                   inv_s1_ys[OBS_SIZE],
                   inv_s1_os[OBS_SIZE],
                   inv_s2_xs[OBS_SIZE],
                   inv_s2_ys[OBS_SIZE],
                   inv_s2_os[OBS_SIZE]], feed_dict=get_feed_dic_obs(obs))
  return [x[0] for x in ret]

def get_active_inv_direct(sess, query):
  obs = []
  for i in range(OBS_SIZE):
    most_conf = get_most_confuse(sess, obs)
    obs.append((most_conf, query(most_conf)))
  ret =  sess.run([inv_s1_xs[OBS_SIZE],
                   inv_s1_ys[OBS_SIZE],
                   inv_s1_os[OBS_SIZE],
                   inv_s2_xs[OBS_SIZE],
                   inv_s2_ys[OBS_SIZE],
                   inv_s2_os[OBS_SIZE]], feed_dict=get_feed_dic_obs(obs))
  return [x[0] for x in ret]
  

def get_active_inv(sess, query, OBS_SIZE=OBS_SIZE):
  obs = []

  for i in range(OBS_SIZE):
    most_conf = get_most_confuse(sess, obs)
    print "most conf so far ", most_conf
    obs.append((most_conf, query(most_conf)))

  feed_dic = get_feed_dic_obs(obs)
  invs = [(x[0], np.argmax(x[0])) for x in sess.run(x_invs, feed_dict=feed_dic)]
  return zip([None] + obs, get_all_preds(sess, obs), invs)
  
