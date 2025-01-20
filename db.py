import psycopg2
from psycopg2 import OperationalError

connection_string = "postgresql://postgres:Soumyaspeechtotext@db.pxsbfrvoqnereymigedz.supabase.co:5432/postgres"

try:
    conn = psycopg2.connect(connection_string)
    print("Connection successful!")
    conn.close()
except OperationalError as e:
    print("Error connecting to the database:", str(e))
