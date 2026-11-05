"""Application checks with real EDF bytes injected at the upload widget boundary."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def check(app):
    assert not app.exception, list(app.exception)


app = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=90).run()
check(app)
assert not app.error and len(app.metric) == 2
app.select_slider[0].set_value(15).run()
app.selectbox[0].set_value("F7-T7").run()
check(app)
assert not app.error and len(app.metric) == 2
app.radio[0].set_value("Upload EDF").run()
check(app)
content = (ROOT / "data/raw/chb01/chb01_18.edf").read_bytes()
for payload, size, expected_error in [
    (content, len(content), False),
    (b"invalid EDF", 11, True),
    (b"", 51 * 1024**2, True),
]:
    upload = SimpleNamespace(size=size, getvalue=lambda: payload)
    with patch("streamlit.file_uploader", return_value=upload):
        app.run()
    check(app)
    assert bool(app.error) == expected_error, list(app.error)
print("PASS: sample, window/channel navigation, real EDF upload handler, corrupt/oversized upload rejection")
print("Upload widget was mocked. Native browser file selection remains a manual check.")
