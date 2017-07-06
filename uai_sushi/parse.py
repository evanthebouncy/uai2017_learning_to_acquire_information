fd = open("sushi3a.5000.10.order")

training = []
testing = []

for i, l in enumerate(fd.readlines()):
  if i == 0:
    continue
  print i, l
  l = l.split()
  print l
  if i < 2500:
    training.append(map(lambda x: int(x), l[2:]))
  else:
    testing.append(map(lambda x: int(x), l[2:]))

new_file = open("sort_data.py", "w")
new_file.write("sort_train = "+repr(training))
new_file.write("\n")
new_file.write("sort_test = "+repr(testing))
new_file.write("\n")
