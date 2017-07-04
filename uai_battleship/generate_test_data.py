from data import *

tests = []

for _ in range(1000):
  tests.append(get_img_class())

wrte_file = open("test_data.py", "w")

to_write = "from numpy import array \ntest_data = " + repr(tests)

wrte_file.write(to_write)
  

