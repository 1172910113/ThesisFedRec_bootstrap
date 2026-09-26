def test_package_import() -> None:
    import src

    assert src.__version__ == "0.1.0"
