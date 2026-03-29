import pytest
from storage.sqlite_store import SQLiteStore

@pytest.fixture
def store(tmp_path):
    return SQLiteStore(db_path=str(tmp_path / "test.db"))

def test_config_audit_written(store):
    store.upsert_config_audit("abc123def", "hash_xyz_456")
    row = store._conn.execute(
        "SELECT sha, config_hash FROM config_audit ORDER BY loaded_at DESC LIMIT 1"
    ).fetchone()
    assert row is not None
    assert row[0] == "abc123def"
    assert row[1] == "hash_xyz_456"

def test_config_audit_multiple_entries(store):
    store.upsert_config_audit("sha1", "hash1")
    store.upsert_config_audit("sha2", "hash2")
    rows = store._conn.execute("SELECT COUNT(*) FROM config_audit").fetchone()
    assert rows[0] == 2
