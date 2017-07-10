from data import *
from draw import *

partial, full = rand_data(0.5)
partial, full = partial[0], full[0]

draw_allob(full, "full_ob.png", [])
draw_allob(partial, "partial_ob.png", [])

