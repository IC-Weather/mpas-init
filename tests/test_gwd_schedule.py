import torch

from mpas_init._gwd_schedule import footprint_batches


def test_footprint_schedule_preserves_cells_shapes_and_memory_budget():
    shapes = [(10, 12), (10, 12), (20, 30), (10, 12), (100, 100), (20, 30)]
    latitude = torch.tensor([.2, -.5, 0., .2, 1.57, -.1], dtype=torch.float64)
    longitude = torch.tensor([.1, 1., -1., .15, 0., -1.1], dtype=torch.float64)
    groups = footprint_batches(shapes, latitude, longitude, max_pixels=600, max_cells=2)
    restored = [None] * len(shapes)
    for shape, cells in groups:
        assert len(cells) <= 2
        assert len(cells) * shape[0] * shape[1] <= 600 or len(cells) == 1
        for cell in cells:
            assert restored[cell] is None
            assert shapes[cell] == shape
            restored[cell] = shape
    assert restored == shapes
    assert next(cells for shape, cells in groups if shape == (100, 100)) == [4]
