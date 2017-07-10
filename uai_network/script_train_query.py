from model import *
from draw import *

# start the training process some arguments
restart = False
# restart = True

impnet = Implynet("imp")
sess = tf.Session()

if restart:
  sess.run(impnet.init)
else:
  impnet.load_model(sess, "model_imply.ckpt")

for i in xrange(10000):
  epi = np.random.uniform(0.0, 1.0)
  print i, " ", epi
  # train it on a sample of expeirence
  impnet.train(sess, rand_data(epi))

  if i % 20 == 0:
#     partial, full = rand_data(epi)
#     predzz = sess.run(impnet.query_preds, impnet.gen_feed_dict(partial, full))
#     predzz0 = np.array([x[0] for x in predzz])
#     print show_dim(predzz0)
#     predzz0 = np.reshape(predzz0, [L,L,2])
#     draw_allob(predzz0, "drawings/pred_ob.png", [])
#     draw_allob(full[0], "drawings/orig_ob.png", [])
#     draw_allob(partial[0], "drawings/partial_ob.png", [])
    impnet.save(sess)

