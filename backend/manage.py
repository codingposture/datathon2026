#!/usr/bin/env python
import os
from pathlib import Path
import signal
import sys

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
    # Ensure local worker cleanup also runs when the development server is terminated.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    from django.core.management import execute_from_command_line
    execute_from_command_line(sys.argv)
