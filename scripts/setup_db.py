import psycopg2
from pathlib import Path

# Establish a connection to your PostgreSQL database
conn = psycopg2.connect(
    dbname="postgres",
    user="john",
    password="",
    host="localhost", 
    port="5432" 
)

schema_path = Path(__file__).resolve().with_name("create_nba_game_logs.sql")
with schema_path.open("r") as file:
    sql_script = file.read()

# Create a cursor object and execute the SQL
cur = conn.cursor()
cur.execute(sql_script)

# Commit the changes and close the connection
conn.commit()
cur.close()
conn.close()

print("Database setup complete!")
