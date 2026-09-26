from pathlib import Path

import pytest

from src.data.contracts import DatasetBundle
from src.data.preprocess import preprocess_ml1m


@pytest.fixture
def ml1m_ratings_file(tmp_path: Path) -> Path:
    ratings_path = tmp_path / "ratings.dat"
    ratings_path.write_text(
        "\n".join(
            (
                "10::100::5::10",
                "10::101::4::20",
                "10::102::3::20",
                "20::100::4::11",
                "20::103::5::12",
                "20::104::2::13",
                "30::100::3::1",
                "30::101::3::2",
                "40::105::2::1",
                "40::105::5::2",
                "40::106::4::3",
                "40::107::4::4",
                "50::108::5::1",
                "50::109::5::2",
                "50::110::5::3",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return ratings_path


@pytest.fixture
def processed_bundle(ml1m_ratings_file: Path) -> DatasetBundle:
    return preprocess_ml1m(ml1m_ratings_file)
