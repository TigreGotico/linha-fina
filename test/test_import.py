def test_package_imports():
    import linha_fina  # noqa: F401
    from linha_fina.version import __version__

    assert __version__


def test_engine_imports():
    from linha_fina.engine import IntentEngine

    assert IntentEngine is not None


def test_opm_imports():
    from linha_fina.opm import LinhaFinaPipeline

    assert LinhaFinaPipeline is not None
