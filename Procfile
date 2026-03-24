release: python3 migrations/apply_migrations.py || true && python3 seed.py || true
web: uvicorn app:app --host 0.0.0.0 --port $PORT
