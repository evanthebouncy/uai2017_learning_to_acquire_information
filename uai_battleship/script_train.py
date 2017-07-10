from model import *
from draw import *
import sys


# start the training process some arguments
restart = True

def train_model(impnet, epoch=3000):

  if restart:
    impnet.initialize()
  else:
    impnet.load_model(sess, self.name+".ckpt")

  for i in xrange(epoch):
    epi = np.random.random()
    print i, " ", epi
    impnet.train(rand_data(epi))

    if i % 20 == 0:
      partial, full = rand_data(epi)
      predzz = impnet.sess.run(impnet.query_preds, 
                               impnet.gen_feed_dict(partial, full))
      predzz0 = np.array([x[0] for x in predzz])
      print show_dim(predzz0)
      predzz0 = np.reshape(predzz0, [L,L,2])
      draw_allob(predzz0, "drawings/pred_ob.png", [])
      draw_allob(full[0], "drawings/orig_ob.png", [])
      draw_allob(partial[0], "drawings/partial_ob.png", [])
      impnet.save()

# train for different kind of models . . .
model_type = sys.argv[1]
if model_type == "0layer":
  impnet = Implynet(embed_0_layer, "0layer", tf.Session())
if model_type == "1layer":
  impnet = Implynet(embed_1_layer, "1layer", tf.Session())
if model_type == "cnnlayer":
  impnet = Implynet(embed_cnn_layer, "cnnlayer", tf.Session())

train_model(impnet, 10000)

