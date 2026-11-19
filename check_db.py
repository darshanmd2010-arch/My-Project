import sqlite3
conn = sqlite3.connect("data/higher_ed.db")
cols = [r[1] for r in conn.execute("pragma table_info(students)")]
print(len(cols), "columns:", cols)
conn.close()
