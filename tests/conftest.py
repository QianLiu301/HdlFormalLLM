"""No test may reach a real database.

The project's .env points at the Neon database on a developer's machine, and
code under test reads it when DATABASE_URL is not set. Tests that need a database
use XDB_TEST_URL (a local one they may wipe) and set DATABASE_URL themselves.
"""
import os

os.environ['XDB_DISABLE_DOTENV'] = '1'
os.environ.pop('DATABASE_URL', None)
