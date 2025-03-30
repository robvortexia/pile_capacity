import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

def setup_database():
    # Connect to the default postgres database
    conn = psycopg2.connect(
        dbname='postgres',
        user='postgres',
        password='6154Lampard',
        host='localhost',
        port='5432'
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()

    try:
        # Check if database exists
        cur.execute("SELECT 1 FROM pg_database WHERE datname = 'flask_app'")
        exists = cur.fetchone()
        
        if not exists:
            # Create the database
            cur.execute('CREATE DATABASE flask_app')
            print("Database 'flask_app' created successfully!")
        else:
            print("Database 'flask_app' already exists.")
            
    except Exception as e:
        print(f"Error creating database: {e}")
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    setup_database() 