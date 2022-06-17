import pytest

def pytest_addoption(parser):
   parser.addoption(
       "--enable-vscode-break-on-exception",
       action="store_true",
       default = False,
       help="If true, re-throw exceptions so vscode can break",
   )

break_on_exception = False
def pytest_configure(config):
    global break_on_exception
    break_on_exception = config.getoption("--enable-vscode-break-on-exception")

@pytest.hookimpl(tryfirst=True)
def pytest_exception_interact(call):
    if break_on_exception:
        raise call.excinfo.value

@pytest.hookimpl(tryfirst=True)
def pytest_internalerror(excinfo):
    if break_on_exception:
        raise excinfo.value