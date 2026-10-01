from scripts.make_results import mean_std, split_seed


def test_split_seed_names():
    assert split_seed("random") == ("random", 0)
    assert split_seed("lidar_depth0.2_s1") == ("lidar_depth0.2", 1)
    assert split_seed("lidar_depth0.05_15k") == ("lidar_depth0.05_15k", 0)
    assert split_seed("random_s12") == ("random", 12)


def test_mean_std_formatting():
    assert mean_std([]) == "n/a"
    assert mean_std([1.0], 2) == "1.00"
    assert mean_std([1.0, 2.0, 3.0], 2) == "2.00 +- 1.00"
