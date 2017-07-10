from model import *
from draw import *
# ------------- helpers --------------
# some on-policy data generation
def full_output_to_trace(full_out, Img, S):
  obs = [x[0] for x in full_out[1:]]
  return Trace(Img, S, obs)

# start the training process some arguments
restart = True

impnet = Implynet("imp")
invnet = Invnet("inv")
sess = tf.Session()
impnet.load_model(sess, "model_imply.ckpt")
sess.run(invnet.init)

exp_rand = Experience(500)
for i in range(500):
  exp_rand.add(gen_rand_trace())

# twas a bad idea...
# print "generating active traces "
# exp_acti = Experience(500)
# for i in range(500):
#   print "gen acti trace ", i
#   tr_ran = exp_rand.buf[i]
#   qry = mk_query(tr_ran.Img)
#   act_inv = impnet.get_active_inv(sess, qry, epi=0.0)
#   act_tr = full_output_to_trace(act_inv, tr_ran.Img, tr_ran.S)
#   exp_acti.add(act_tr)

  
  

for i in xrange(1000000):
  print i
  # train it on a sample of expeirence
  lab_batch = []
  img_batch = []
  for _ in range(N_BATCH):
    tr = exp_rand.sample()
    lab_batch.append(tr.S)
    img_batch.append(tr.Img)
  invnet.train(sess, inv_data_from_label_data(lab_batch, img_batch))

# twas a bad idea
#  # train it on a sample of active trace expeirence
#  lab_batch = []
#  obs_batch = []
#  for _ in range(N_BATCH):
#    tr = exp_acti.sample()
#    lab_batch.append(tr.S)
#    obs_batch.append(tr.Os)
#  b_obs = inv_batch_obs(lab_batch, obs_batch)
#  invnet.train(sess, b_obs)

  if i % 1000 == 0:
    print "testing it on random trace . . ."
    rand_tr = gen_rand_trace(test=True)
    print rand_tr.S, np.argmax(rand_tr.S)
    invv = invnet.invert(sess, rand_tr.Os)
    print invv, np.argmax(invv)
    draw_obs(rand_tr.Os, "drawings/inv_rand_obs.png")

    print "testing it on active trace . . ."
    qry = mk_query(rand_tr.Img)
    act_inv = impnet.get_active_trace(sess, qry, epi=0.0)
    act_tr = full_output_to_trace(act_inv, rand_tr.Img, rand_tr.S)
    print act_tr.S, np.argmax(act_tr.S)
    invv = invnet.invert(sess, act_tr.Os)
    print invv, np.argmax(invv)
    draw_obs(act_tr.Os, "drawings/inv_acti_obs.png")
    draw(np.reshape(rand_tr.Img, [L,L,1]), "drawings/inv_orig.png")


    invnet.save(sess, "model_invert.ckpt")
    

