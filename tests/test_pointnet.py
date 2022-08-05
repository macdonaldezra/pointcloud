import torch
from pointcloud.models.pointnet import (
    PointNetClassifier,
    PointNetSegmentor,
    TNet,
    TNetFeatureTransformer,
)


def test_tnet_dim() -> None:
    """
    Test that TNet model has output matching expected dimensions.
    """
    sim_data = torch.autograd.Variable(torch.rand(32, 3, 2500))
    trans = TNet()
    out = trans(sim_data)
    actual = out.size()
    expected = [32, 3, 3]
    assert len(actual) == len(expected)
    assert all([a == b for a, b in zip(actual, expected)])


def test_feature_transform_dim() -> None:
    """
    Test that feature transform model has expected output dimensions.
    """
    sim_data = torch.autograd.Variable(torch.rand(32, 3, 2500))

    point_feature = TNetFeatureTransformer()
    out, _ = point_feature(sim_data)
    assert all([a == b for a, b in zip(out.size(), [32, 1024])])

    point_feature = TNetFeatureTransformer(global_feature=False)
    out, _ = point_feature(sim_data)
    assert all([a == b for a, b in zip(out.size(), [32, 1088, 2500])])

    classifier = PointNetClassifier(k=5)
    out, _ = classifier(sim_data)
    assert all([a == b for a, b in zip(out.size(), [32, 5])])


def test_segmentor_dim() -> None:
    """
    Test that the segmentation model has expected output dimensions.
    """
    sim_data = torch.autograd.Variable(torch.rand(32, 3, 2500))

    segmentor = PointNetSegmentor(k=3)
    out, _ = segmentor(sim_data)
    assert all([a == b for a, b in zip(out.size(), [32, 2500, 3])])
