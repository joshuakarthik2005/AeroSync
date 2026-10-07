"""Streamlit Community Cloud entry point.

Cloud runs ``streamlit run streamlit_app.py`` from the repository root without
installing the package, so the root is on ``sys.path`` and ``aerosync`` imports
directly. The dashboard itself lives in ``aerosync/webapp/app.py``; it is executed
on every Streamlit rerun.
"""

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).parent / "aerosync" / "webapp" / "app.py"), run_name="__main__")
