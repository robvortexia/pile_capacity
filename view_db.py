import psycopg2
from tabulate import tabulate

# Database connection parameters
DB_PARAMS = {
    'dbname': 'pile_capacity_calculator_website_analytics',
    'user': 'analytics_ivy3_user',
    'password': 'c78LEqkQpjYwG59DwoI_',
    'host': 'dpg-cviv241r8fns73e9vtdg-a.singapore-postgres.render.com',
    'port': '5432',
    'sslmode': 'require'
}

def view_database():
    try:
        # Connect directly using psycopg2
        conn = psycopg2.connect(**DB_PARAMS)
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
                
    except Exception as e:
        print(f"Error connecting to database: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    view_database() 