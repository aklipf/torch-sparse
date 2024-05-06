from typing import Callable


def assert_no_out_arr(fn: Callable) -> Callable:
    def sys_check(capsys):
        fn()

        captured = capsys.readouterr()
        assert len(captured.out) == 0, f"Captured stdout:\n{captured.out}"
        assert len(captured.err) == 0, f"Captured stderr:\n{captured.err}"

    return sys_check
