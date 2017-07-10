from data import NOISE, NOISE_PR
import random
from copy import copy
from copy import deepcopy
import numpy as np

def corrupt(bit):
  if NOISE and np.random.random() < NOISE_PR:
    return not bit
  return bit

def sorta(arrr):
  tracker = []

  def sub_partition(array, start, end, idx_pivot):

      'returns the position where the pivot winds up'

      if not (start <= idx_pivot <= end):
          raise ValueError('idx pivot must be between start and end')

      array[start], array[idx_pivot] = array[idx_pivot], array[start]
      tracker.append(copy(array))
      pivot = array[start]
      i = start + 1
      j = start + 1

      while j <= end:
          tracker.append(copy(array))
          if corrupt(array[j] <= pivot):
              array[j], array[i] = array[i], array[j]
              i += 1
          j += 1

      array[start], array[i - 1] = array[i - 1], array[start]
      tracker.append(copy(array))
      return i - 1

  def quicksort(array, start=0, end=None):

      if end is None:
          end = len(array) - 1

      if end - start < 1:
          return

      idx_pivot = random.randint(start, end)
      i = sub_partition(array, start, end, idx_pivot)
      #print array, i, idx_pivot
      quicksort(array, start, i - 1)
      quicksort(array, i + 1, end)

  quicksort(arrr)
  return tracker

# merge sort

def sortb(arrr):
  tracker = []
  def merge_sort(xs):
      unit = 1
      while unit <= len(xs):
          h = 0
          for h in range(0, len(xs), unit * 2):
              l, r = h, min(len(xs), h + 2 * unit)
              mid = h + unit
              # merge xs[h:h + 2 * unit]
              p, q = l, mid
              while p < mid and q < r:
                  if corrupt(xs[p] < xs[q]): p += 1
                  else:
                      tmp = xs[q]
                      xs[p + 1: q + 1] = xs[p:q]
                      xs[p] = tmp
                      p, mid, q = p + 1, mid + 1, q + 1
                  tracker.append(copy(xs))

          unit *= 2
      
      return xs
  merge_sort(arrr)

  return tracker

def sortc(arrr):
  tracker = []
  
  
  sortedd = False
  while not sortedd:
      sortedd = True  # Assume the list is now sortedd
      for element in range(0, len(arrr)-1):
          if corrupt(arrr[element] > arrr[element + 1]):
              sortedd = False  # We found two elements in the wrong order
              hold = arrr[element + 1]
              arrr[element + 1] = arrr[element]
              arrr[element] = hold
          tracker.append(copy(arrr))

  return tracker

# trace = sortc([10,1,8,7,6,2,4,3,5,9])
# for tr in trace:
#   print tr
