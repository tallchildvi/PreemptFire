import numpy as np
from numba import float32, int64, types
from numba.experimental import jitclass
from numba.typed import List

node_type = types.Tuple((float32, int64, int64))

spec = [
    ("_heap", types.ListType(node_type)),
    ("_heapSize", int64),
]


@jitclass(spec)
class PriorityQueue:
    def __init__(self):
        self._heap = List.empty_list(node_type)
        self._heapSize = 0

    def __len__(self):
        return self._heapSize

    def add(self, node: tuple):
        self._heap.append(node)
        self._heapSize += 1
        i = self._heapSize - 1

        while i > 0:
            parent = (i - 1) // 2
            if self._heap[parent][0] > self._heap[i][0]:
                self._heap[parent], self._heap[i] = self._heap[i], self._heap[parent]
                i = parent
            else:
                break

    def peek(self):
        if self._heapSize == 0:
            return (np.float32(np.inf), np.int64(-1), np.int64(-1))
        return self._heap[0]

    def extract_min(self):
        if self._heapSize == 0:
            return (np.float32(np.inf), np.int64(-1), np.int64(-1))

        min_val = self._heap[0]

        if self._heapSize == 1:
            self._heap.pop()
            self._heapSize = 0
            return min_val

        self._heap[0] = self._heap.pop()
        self._heapSize -= 1
        i = 0

        while True:
            left = 2 * i + 1
            right = 2 * i + 2

            if left >= self._heapSize:
                break

            smallest = left
            if right < self._heapSize and self._heap[right][0] < self._heap[left][0]:
                smallest = right

            if self._heap[i][0] <= self._heap[smallest][0]:
                break

            self._heap[i], self._heap[smallest] = self._heap[smallest], self._heap[i]
            i = smallest

        return min_val