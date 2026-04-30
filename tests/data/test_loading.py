from pathlib import Path

import pandas as pd
import pytest

from life_expectancy.data.loading import load_csv, load_source


def test_load_csv_success(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample.csv"
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(csv_path, index=False)

    df = load_csv(csv_path)

    assert df.shape == (2, 2)
    assert df["a"].tolist() == [1, 2]


def test_load_csv_missing_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError):
        load_csv(missing_path)


def test_load_source_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "who.csv"
    pd.DataFrame({"Country": ["Albania"], "Year": [2015]}).to_csv(csv_path, index=False)

    config = {
        "project": {"root": str(tmp_path)},
        "data": {
            "raw_sources": {
                "who": {
                    "path": "who.csv",
                    "loader": "csv",
                }
            }
        },
    }

    df = load_source(config, "who")

    assert df.shape == (1, 2)
    assert df.loc[0, "Country"] == "Albania"


def test_load_source_unsupported_loader(tmp_path: Path) -> None:
    config = {
        "project": {"root": str(tmp_path)},
        "data": {
            "raw_sources": {
                "bad": {
                    "path": "bad.txt",
                    "loader": "txt",
                }
            }
        },
    }

    with pytest.raises(ValueError):
        load_source(config, "bad")
