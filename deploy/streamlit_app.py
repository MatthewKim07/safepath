"""Streamlit Community Cloud entrypoint: `streamlit run deploy/streamlit_app.py`.

The `safepath` package is not installed on Streamlit Cloud (installing it would pull the
whole pipeline's dependencies), so `src/` goes on the path and the app runs from the
bundled data in app_bundle/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from safepath.app.streamlit_app import main  # noqa: E402

main()
