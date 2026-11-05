"""
Gurobi license configuration.

Credentials are never stored in this repository. For a Web License Service
(WLS) license, provide them in one of two ways (the environment wins):

1. Environment variables, using Gurobi's own names:
       GRB_WLSACCESSID, GRB_WLSSECRET, GRB_LICENSEID
2. An untracked file ``gurobi.env`` at the repository root (or the path in
   ``MASS_GUROBI_ENV``) with one ``NAME=value`` per line, same names.

If none are set, Gurobi falls back to a local license file (``~/gurobi.lic``
or ``GRB_LICENSE_FILE``). Set ``MASS_GUROBI_LOCAL_LICENSE=1`` to force that.

A free academic license: https://www.gurobi.com/academia/academic-program-and-licenses/
"""

import os
from pathlib import Path

_KEYS = ("GRB_WLSACCESSID", "GRB_WLSSECRET", "GRB_LICENSEID")
_DEFAULT_ENV_FILE = Path(__file__).resolve().parent.parent / "gurobi.env"


def _read_env_file(path):
    values = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                name, value = line.split("=", 1)
                values[name.strip()] = value.strip().strip('"').strip("'")
    except FileNotFoundError:
        pass
    return values


def _load_wls_config():
    from_file = _read_env_file(os.environ.get("MASS_GUROBI_ENV", _DEFAULT_ENV_FILE))
    values = {k: os.environ.get(k) or from_file.get(k) for k in _KEYS}
    if not all(values.values()):
        return None
    return {
        "WLSACCESSID": values["GRB_WLSACCESSID"],
        "WLSSECRET": values["GRB_WLSSECRET"],
        "LICENSEID": int(values["GRB_LICENSEID"]),
    }


# Use a local license file instead of WLS.
USE_LOCAL_LICENSE = os.environ.get("MASS_GUROBI_LOCAL_LICENSE", "") not in ("", "0")

# WLS connection parameters, or None when no credentials are configured.
GUROBI_WLS_CONFIG = None if USE_LOCAL_LICENSE else _load_wls_config()
