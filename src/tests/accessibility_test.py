import numpy as np
from src.data_pipeline.accessibility.solver import dijkstra_kernel

def test_dijkstra_30x30():

    size = 30
    resolution = 10.0 

    elevation = np.zeros((size, size), dtype=np.float32)

    for col in range(size):
        elevation[:, col] = max(0, col - 14) * 2.0

    sources = np.zeros((size, size), dtype=np.uint8)

    sources[5, :] = 1
    sources[24, :] = 1

    optimal_time = dijkstra_kernel(
        elevation,
        sources,
        resolution
    )

    assert optimal_time.shape == (30, 30)
    assert np.all(optimal_time[5, :] == 0)
    assert np.all(optimal_time[24, :] == 0)
    assert np.all(optimal_time >= 0)
    assert np.all(np.isfinite(optimal_time))

    t_near = optimal_time[6, 10]
    t_far = optimal_time[14, 10]

    assert t_far > t_near

    print("\nAccessibility time [hours]:")
    print(optimal_time)

    print("\nStatistics:")
    print(f"min:  {optimal_time.min():.6f} h")
    print(f"max:  {optimal_time.max():.6f} h")
    print(f"mean: {optimal_time.mean():.6f} h")


if __name__ == "__main__":
    test_dijkstra_30x30()