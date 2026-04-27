import sqlite3
conn = sqlite3.connect('msl_dev.db')

# Check fingerprints columns
cursor = conn.execute("SELECT * FROM fingerprints LIMIT 1")
print(f"Fingerprints columns: {[d[0] for d in cursor.description]}")

# Check locations columns (This is likely where 'AW212' is stored)
cursor = conn.execute("SELECT * FROM locations LIMIT 1")
print(f"Locations columns: {[d[0] for d in cursor.description]}")

conn.close()