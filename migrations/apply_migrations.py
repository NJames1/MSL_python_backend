import os
from sqlalchemy import text
from db import engine

MIGRATIONS_DIR = os.path.join(os.path.dirname(__file__))


def apply_migration_file(path):
    with open(path, "r") as f:
        sql = f.read()
    with engine.begin() as conn:
        conn.execute(text(sql))


def main():
    files = sorted(f for f in os.listdir(MIGRATIONS_DIR) if f.endswith('.sql'))
    for fname in files:
        path = os.path.join(MIGRATIONS_DIR, fname)
        print(f"Applying {fname}...")
        apply_migration_file(path)
    print("Migrations applied.")


if __name__ == '__main__':
    main()
