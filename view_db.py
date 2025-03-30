import psycopg2
from tabulate import tabulate
import os

def get_database_url():
    """Get database URL from environment variable or return default local connection string"""
    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        # Default connection string for Render PostgreSQL database
        database_url = "postgresql://analytics_ivy3_user:c78LEqbQpjYwG59DwoIcJmo91CKZ2Crb@dpg-cviv241r0fns73e9vtdg-a.singapore-postgres.render.com/analytics_ivy3"
        print("\nNo DATABASE_URL environment variable found.")
        print("Using Render PostgreSQL database connection.")
    return database_url

def view_database():
    try:
        database_url = get_database_url()
        
        # Connect using DATABASE_URL
        # If DATABASE_URL contains 'render.com', it's a remote connection requiring SSL
        # Otherwise, it's a local connection where SSL is not needed
        conn_params = {
            'dsn': database_url,
            'sslmode': 'require' if 'render.com' in database_url else 'disable'
        }
        
        print("\nAttempting to connect to database...")
        conn = psycopg2.connect(**conn_params)
        cur = conn.cursor()
        
        # Get list of tables
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        tables = [row[0] for row in cur.fetchall()]
        
        print("\nTables in database:")
        print("------------------")
        for table in tables:
            print(f"- {table}")
        
        # View contents of each table
        for table in tables:
            print(f"\nContents of {table}:")
            print("-" * (len(table) + 13))
            
            cur.execute(f"SELECT * FROM {table}")
            rows = cur.fetchall()
            
            if rows:
                # Get column names
                cur.execute(f"""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = '{table}'
                    ORDER BY ordinal_position
                """)
                headers = [col[0] for col in cur.fetchall()]
                print(tabulate(rows, headers=headers, tablefmt='grid'))
            else:
                print("(empty table)")
            print()
                
    except psycopg2.OperationalError as e:
        print(f"\nError connecting to database: {e}")
        print("\nPlease ensure that:")
        print("1. PostgreSQL is installed and running")
        print("2. The database exists")
        print("3. The username and password are correct")
        print("4. The port number is correct (default: 5432)")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    view_database() 