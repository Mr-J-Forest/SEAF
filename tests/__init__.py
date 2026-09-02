# Test suite package.
#
# Making ``tests`` a regular package guarantees ``from tests.xxx import ...``
# resolves to this directory even when a third-party distribution also
# installs a package named ``tests`` into site-packages (which otherwise
# shadows the namespace-package fallback).
