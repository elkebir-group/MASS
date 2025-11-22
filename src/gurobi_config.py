"""
Gurobi license configuration file.

This file contains the Gurobi Web License Service (WLS) connection parameters.
You can obtain a free academic license from: https://www.gurobi.com/academia/academic-program-and-licenses/

For WLS licenses, you need:
- WLSACCESSID: Your WLS access ID
- WLSSECRET: Your WLS secret key
- LICENSEID: Your license ID number

Alternatively, if you have a local Gurobi license, you can set these to None and Gurobi will use
the local license file (typically located at ~/gurobi.lic or configured via GRB_LICENSE_FILE).
"""

# Gurobi Web License Service (WLS) connection parameters
# Set to None if using a local license file instead
GUROBI_WLS_CONFIG = {
    "WLSACCESSID": "3de5b0e3-4676-4546-858c-045b24afbf61",
    "WLSSECRET": "27e16772-b43e-4edc-840d-e1b55a766684",
    "LICENSEID": 2685986,
}

# Alternative: Use local license file
# Set this to True to use a local Gurobi license file instead of WLS
USE_LOCAL_LICENSE = False

