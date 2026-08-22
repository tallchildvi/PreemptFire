import heapq
import numpy as np

try:
    from numba import njit, float32, int64, types
    from numba.experimental import jitclass
    from numba.typed import List
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

if HAS_NUMBA:
    node_type = types.Tuple((float32, int64, int64))

    spec = [
        ("_heap", types.ListType(node_type)),
        ("_heapSize", int64),
    ]

    @jitclass(spec)
    class PriorityQueue:
        def __init__(self):
            self._heap = List.empty_list(node_type)
            self._heapSize = int64(0)

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
                return (np.float32(np.inf), int64(-1), int64(-1))
            return self._heap[0]

        def extract_min(self):
            if self._heapSize == 0:
                return (np.float32(np.inf), int64(-1), int64(-1))
            min_val = self._heap[0]
            if self._heapSize == 1:
                self._heap.pop()
                self._heapSize = 0
                return min_val
            self._heap[0] = self._heap.pop()
            self._heapSize -= 1
            i = 0
            while True:
                left  = 2 * i + 1
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


if HAS_NUMBA:
    @njit(fastmath=True, cache=True)
    def _dijkstra_numba(
        elevation: np.ndarray,
        sources: np.ndarray,
        passable_mask: np.ndarray,
        resolution: float,
    ) -> np.ndarray:

        rows, cols = elevation.shape

        optimal_time = np.full((rows, cols), np.inf, dtype=np.float32)

        queue = PriorityQueue()

        for r in range(rows):
            for c in range(cols):
                if sources[r, c] == 1 and passable_mask[r, c] == 1:
                    optimal_time[r, c] = np.float32(0.0)
                    queue.add((np.float32(0.0), int64(r), int64(c)))

        if len(queue) == 0:
            return optimal_time

        DR = (-1,  1,  0,  0, -1, -1,  1,  1)
        DC = ( 0,  0, -1,  1, -1,  1, -1,  1)
        DF = (1.0, 1.0, 1.0, 1.0,
              1.41421356, 1.41421356, 1.41421356, 1.41421356)

        while len(queue) != 0:
            curr_time, r, c = queue.extract_min()

            if curr_time > optimal_time[r, c]:
                continue

            z_curr = elevation[r, c]

            for i in range(8):
                nr = r + DR[i]
                nc = c + DC[i]

                if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                    continue
                if passable_mask[nr, nc] == 0:
                    continue

                dx = resolution * DF[i]
                dh = float(elevation[nr, nc]) - float(z_curr)
                slope = dh / dx

                if slope > 0.7813 or slope < -0.7002:
                    continue

                speed_kmh = 6.0 * np.exp(-3.5 * np.abs(slope + 0.05))

                if speed_kmh <= 0.01:
                    continue

                t_step = np.float32((dx / 1000.0) / speed_kmh)

                if dh > 0.0:
                    t_step += np.float32(dh / 600.0)

                if sources[nr, nc] == 0:
                    t_step *= np.float32(2.5)

                cand_time = curr_time + t_step

                # Relaxation
                if cand_time < optimal_time[nr, nc]:
                    optimal_time[nr, nc] = cand_time
                    queue.add((cand_time, int64(nr), int64(nc)))

        return optimal_time


NEIGHBOR_OFFSETS = np.array([
    [-1,  0, 1.0],
    [ 1,  0, 1.0],
    [ 0, -1, 1.0],
    [ 0,  1, 1.0],
    [-1, -1, 1.41421356],
    [-1,  1, 1.41421356],
    [ 1, -1, 1.41421356],
    [ 1,  1, 1.41421356],
], dtype=np.float64)


def _tobler_travel_time(dh: float, dx: float) -> float:
    slope = dh / dx
    if slope > 0.7813 or slope < -0.7002:
        return np.inf
    speed_kmh = 6.0 * np.exp(-3.5 * abs(slope + 0.05))
    if speed_kmh <= 0.01:
        return np.inf
    return (dx / 1000.0) / speed_kmh


def _dijkstra_python(
    elevation: np.ndarray,
    sources: np.ndarray,
    passable_mask: np.ndarray,
    resolution: float,
) -> np.ndarray:
    rows, cols = elevation.shape
    time_grid = np.full((rows, cols), np.inf, dtype=np.float32)
    visited   = np.zeros((rows, cols), dtype=bool)
    pq        = []

    source_r, source_c = np.where((sources == 1) & (passable_mask == 1))
    for r, c in zip(source_r, source_c):
        time_grid[r, c] = 0.0
        heapq.heappush(pq, (0.0, int(r), int(c)))

    while pq:
        curr_time, r, c = heapq.heappop(pq)

        if visited[r, c]:
            continue
        visited[r, c] = True

        z_curr = elevation[r, c]

        for i in range(8):
            nr = r + int(NEIGHBOR_OFFSETS[i, 0])
            nc = c + int(NEIGHBOR_OFFSETS[i, 1])
            step_dist_m = resolution * NEIGHBOR_OFFSETS[i, 2]

            if 0 <= nr < rows and 0 <= nc < cols:
                if not visited[nr, nc] and passable_mask[nr, nc] == 1:
                    dh = float(elevation[nr, nc]) - float(z_curr)
                    t_step    = _tobler_travel_time(dh, step_dist_m)
                    cand_time = curr_time + t_step
                    if cand_time < time_grid[nr, nc]:
                        time_grid[nr, nc] = cand_time
                        heapq.heappush(pq, (cand_time, nr, nc))

    return time_grid


def dijkstra_kernel(
    elevation: np.ndarray,
    sources: np.ndarray,
    passable_mask: np.ndarray,
    resolution: float,
) -> np.ndarray:
    if HAS_NUMBA:
        return _dijkstra_numba(
            elevation.astype(np.float32),
            sources.astype(np.uint8),
            passable_mask.astype(np.uint8),
            float(resolution),
        )
    return _dijkstra_python(elevation, sources, passable_mask, resolution)