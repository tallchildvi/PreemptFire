import numpy as np
from numba import njit
from src.data_pipeline.accessibility.queue import PriorityQueue


@njit
def dijkstra_kernel(elevation: np.ndarray, sources: np.ndarray, resolution: float):
    if elevation.shape != sources.shape:
        raise ValueError("Shape mismatch: elevation shape must match sources shape.")

    rows, cols = elevation.shape
    optimal_time = np.where(sources == 1, 0.0, np.inf).astype(np.float32)
    queue = PriorityQueue()

    for row in range(rows):
        for col in range(cols):
            if optimal_time[row, col] == 0:
                queue.add((np.float32(0.0), np.int64(row), np.int64(col)))

    while len(queue) != 0:
        time, r, c = queue.extract_min()

        if time > optimal_time[r, c]:
            continue

        delta_time, arr_shape, arr_pos = get_time(elevation, r, c, resolution)
        abs_time = delta_time + time
        shape_rows, shape_cols = arr_shape
        r_idx, c_idx = arr_pos

        for row in range(shape_rows):
            for col in range(shape_cols):
                if delta_time[row, col] == 0:
                    continue

                curr_r = r_idx + row
                curr_c = c_idx + col

                if optimal_time[curr_r, curr_c] == 0:
                    continue

                if optimal_time[curr_r, curr_c] > abs_time[row, col]:
                    optimal_time[curr_r, curr_c] = abs_time[row, col]
                    node = (optimal_time[curr_r, curr_c], np.int64(curr_r), np.int64(curr_c))
                    queue.add(node)

    return optimal_time


@njit
def get_time(elevation: np.ndarray, pixel_row: int, pixel_col: int, resolution: float):
    rows, cols = elevation.shape
    central_pixel_elevation = elevation[pixel_row, pixel_col]

    min_r = -1 if pixel_row > 0 else 0
    max_r = 1 if pixel_row < rows - 1 else 0

    min_c = -1 if pixel_col > 0 else 0
    max_c = 1 if pixel_col < cols - 1 else 0

    shape_r = max_r - min_r + 1
    shape_c = max_c - min_c + 1

    pos_r = pixel_row + min_r
    pos_col = pixel_col + min_c

    result = np.zeros((shape_r, shape_c), dtype=np.float32)

    for r in range(min_r, max_r + 1):
        for c in range(min_c, max_c + 1):
            i = r - min_r
            j = c - min_c

            if r == 0 and c == 0:
                result[i, j] = 0.0
                continue

            dx = resolution if (r == 0 or c == 0) else resolution * 1.41421356
            dh = elevation[pixel_row + r, pixel_col + c] - central_pixel_elevation

            result[i, j] = toblers_hiking_func(dh, dx)

    return result, (shape_r, shape_c), (pos_r, pos_col)


@njit
def toblers_hiking_func(dh, dx):
    return dx / (6 * np.exp(-3.5 * np.abs(dh / dx + 0.05))) / 1000