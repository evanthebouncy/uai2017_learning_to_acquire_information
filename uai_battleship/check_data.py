from data import *
from draw import *

partial, full = rand_data(0.5)
partial, full = partial[0], full[0]

draw_allob(full, "full_ob.png", [])
draw_allob(partial, "partial_ob.png", [])

poses = [(3, 6, False), (2, 2, True), (9, 7, True), (1, 5, True), (2, 0, False)]
poses2 = [(3, 6, False), (2, 3, False), (2, 4, True), (7, 9, False), (3, 5, False)]

blah1, blah2 = det_crack(poses2)

draw_orig(blah1, "hmm2.png")
