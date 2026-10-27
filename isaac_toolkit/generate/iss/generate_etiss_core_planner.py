"""Compatibility entry point for the now-default encoding planner generator."""
from .generate_etiss_core import *


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
