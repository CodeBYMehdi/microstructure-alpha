"""Manual halt clearance tool. Run when operator confirms safe to resume.

Usage: python -m infrastructure.clear_halt [db_path]
"""
import sys
from storage.sqlite_store import SQLiteStore


def main():
    db_path = sys.argv[1] if len(sys.argv) > 1 else "trades.db"
    store = SQLiteStore(db_path=db_path)

    halt = store.get_halt()
    if halt is None:
        print("No active halt.")
        return

    print(f"Active halt: {halt['reason']} (at {halt['created_at']})")
    confirm = input("Clear this halt and resume trading? [y/N] ")
    if confirm.strip().lower() == "y":
        store.set_halt(None)
        print("Halt cleared. Strategy will resume accepting signals.")
    else:
        print("Halt NOT cleared.")

    store.close()


if __name__ == "__main__":
    main()
