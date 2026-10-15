import pandas as pd

from retail_forecasting.simulation import generate_synthetic_data


def test_synthetic_generation_is_reproducible_and_complete():
    first = generate_synthetic_data(periods=35, stores=2, seed=7)
    second = generate_synthetic_data(periods=35, stores=2, seed=7)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 35 * 2 * 8
    assert first["data_source"].eq("synthetic_simulation").all()
    assert first["stockout_flag"].isin([0, 1]).all()


def test_invalid_simulation_size_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        generate_synthetic_data(periods=0)
