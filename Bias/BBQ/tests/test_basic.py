"""
Basic tests for BBQ-Eval package
"""

import pytest


def test_package_import():
    """Test that the package can be imported successfully"""
    import bbq_eval

    # Test that version exists and follows semantic versioning pattern
    assert hasattr(bbq_eval, "__version__")
    assert isinstance(bbq_eval.__version__, str)
    assert len(bbq_eval.__version__.split(".")) >= 2  # At least major.minor


def test_package_metadata():
    """Test package metadata"""
    import bbq_eval

    assert hasattr(bbq_eval, "__version__")
    assert hasattr(bbq_eval, "__email__")


def test_main_module_import():
    """Test that main module can be imported"""
    from bbq_eval import main

    assert callable(main)


def test_evaluate_module_import():
    """Test that evaluate module can be imported"""
    from bbq_eval import evaluate_main, get_accuracy

    assert callable(evaluate_main)
    assert callable(get_accuracy)


def test_data_directory_exists():
    """Test that data directory exists"""
    import os

    import bbq_eval

    package_dir = os.path.dirname(bbq_eval.__file__)
    data_dir = os.path.join(package_dir, "data")
    # Data directory should exist (it's included in pyproject.toml)
    assert os.path.isdir(data_dir)


def test_cli_entry_point():
    """Test that CLI entry point is properly configured"""
    # Test that the CLI entry point path exists (bbq_eval.main:main)
    # This is what Poetry uses for the CLI command

    # Import the module as it would be imported by the CLI
    import importlib

    main_module = importlib.import_module("bbq_eval.main")

    # Test that the main function exists and is callable
    assert hasattr(main_module, "main")
    assert callable(getattr(main_module, "main"))

    # Also test that we can import the main function at package level
    from bbq_eval import main as package_main

    assert callable(package_main)


if __name__ == "__main__":
    pytest.main([__file__])
