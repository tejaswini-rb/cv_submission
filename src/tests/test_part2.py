from vision.part2_baseline import Baseline
from tests.model_test_utils import count_params, get_layers, get_model_layer_counts
import torch
import numpy as np


def test_count_points():
    model = Baseline(20, voxel_resolution=2)
    points = torch.tensor([[
        [-0.828125, 0.0888671875, -0.6591796875],
        [-0.578125, -0.3447265625, -0.42724609375],
        [-0.578125, 0.2822265625, -0.665283203125],
        [-0.546875, -0.1552734375, -0.427734375],
        [-0.328125, -0.1220703125, 0.23291015625],
        [-0.296875, -0.3076171875, -0.0986328125],
        [-0.265625, -0.4951171875, -0.09912109375],
        [-0.203125, -0.3173828125, 0.23193359375],
        [-0.203125, 0.0869140625, -0.099609375],
        [-0.203125, 0.5263671875, -0.67431640625],
        [-0.140625, -0.5263671875, 0.23193359375],
        [0.078125, -0.1435546875, -0.341796875],
        [0.109375, 0.0986328125, 0.22998046875],
        [0.140625, -0.2763671875, 0.56494140625],
        [0.140625, -0.0654296875, 0.56591796875],
        [0.328125, 0.0517578125, -0.34619140625],
        [0.421875, -0.1513671875, -0.0068359375],
        [0.453125, 0.2490234375, -0.00732421875],
        [0.609375, 0.2724609375, -0.3515625],
        [0.671875, -0.1533203125, 0.67431640625],
        [0.765625, -0.1630859375, 0.33251953125],
        [0.765625, 0.0439453125, 0.33251953125],
        [0.828125, 0.0595703125, 0.67431640625]
    ]])

    counts = model.count_points(points)    
    expected =  torch.tensor([[0.1739, 0.1304, 0.1739, 0.0000, 0.0870, 0.1739, 0.1304, 0.1304]]) 

    assert torch.allclose(counts, expected, atol=1e-5, rtol=1e-3)


def test_baseline():

    model = Baseline(20)
    params = count_params(model)
    layers = get_layers(model)
    layer_counts = get_model_layer_counts(layers)

    assert params < 5000
    assert len(layers) == 1
    assert layer_counts['Linear'] == 1
    assert layers[-1].out_features == 20