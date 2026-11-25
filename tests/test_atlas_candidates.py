import pytest
import torch

from mpas_init._atlas import compile_candidates


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -1.])
def test_atlas_rejects_invalid_query_geometry_before_launch(bad):
    with pytest.raises(ValueError, match='geometry|bounds'):
        compile_candidates('must-not-launch', torch.zeros((1, 3), dtype=torch.float64),
                           torch.tensor([[0., 0., 0., bad]], dtype=torch.float64))
