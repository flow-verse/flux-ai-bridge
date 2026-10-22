from flux_ai.cli import main


def test_help_exits():
    try:
        main(["--help"])
    except SystemExit as e:
        assert e.code == 0
