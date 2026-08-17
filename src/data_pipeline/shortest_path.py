import numpy as np

class BinaryHeap:
    def __init__(self):
        self._heap = []
        self._heapSize = 0

    def __len__(self):
        return self._heapSize

    def add(self, node: tuple):
        self._heap.append(node)
        self._heapSize += 1
        i = self._heapSize - 1
        
        while i > 0:
            parent = (i - 1) // 2
            if self._heap[parent] > self._heap[i]:
                self._heap[parent], self._heap[i] = self._heap[i], self._heap[parent]
                i = parent
            else:
                break
    def peek(self):
        return self._heap[0]

    def ExtractMin(self):
        if self._heapSize == 0:
            return None
        
        min_val = self._heap[0]

        if self._heapSize == 1:
            self._heap.pop()
            self._heapSize -= 1
            return min_val
        
        self._heap[0] = self._heap.pop()
        self._heapSize -= 1
        i = 0
        while (i * 2 + 1) < self._heapSize:
            left = i * 2 + 1
            right = i * 2 + 2
            smallest = i 

            if self._heap[left] < self._heap[smallest]:
                smallest = left

            if right < self._heapSize and self._heap[right] < self._heap[smallest]:
                smallest = right

            if smallest == i:
                break

            self._heap[i], self._heap[smallest] = self._heap[smallest], self._heap[i]
            i = smallest
        return min