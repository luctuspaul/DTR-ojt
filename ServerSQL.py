import pyodbc

# ---------- Configuration ----------

def get_odbc_driver() -> str:
    """Return the latest available SQL Server ODBC driver."""
    drivers = [d for d in pyodbc.drivers() if "ODBC Driver" in d and "SQL Server" in d]
    if not drivers:
        raise Exception("❌ No suitable SQL Server ODBC driver found.")
    return drivers[-1]

# SQL Server credentials
SERVER = 'localhost\\SQLEXPRESS2014'
USERNAME = 'sa'
PASSWORD = 'ojt@2025'
DEFAULT_DB = 'ojt'

# ---------- Core Connection ----------

def make_connection(db_name: str = DEFAULT_DB) -> pyodbc.Connection:
    """Establish connection to specified SQL Server database."""
    driver = get_odbc_driver()
    conn_str = (
        f'DRIVER={{{driver}}};'
        f'SERVER={SERVER};'
        f'DATABASE={db_name};'
        f'UID={USERNAME};'
        f'PWD={PASSWORD};'
        'Encrypt=yes;'
        'TrustServerCertificate=yes;'
    )
    conn = pyodbc.connect(conn_str)
    print(f"✅ Connected to: {conn.getinfo(2)} (DB) on {conn.getinfo(7)} (Server)")
    return conn

# ---------- Accessors ----------

def get_register_connection() -> pyodbc.Connection:
    return make_connection()

def get_attendance_connection() -> pyodbc.Connection:
    return make_connection()

def get_employee_info_connection() -> pyodbc.Connection:
    return make_connection()

def list_columns(conn: pyodbc.Connection, table_name: str):
    print(f"📋 Columns in {table_name}:")
    cursor = conn.cursor()
    cursor.execute(f"SELECT COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = ?", table_name)
    for col in cursor.fetchall():
        print(f" - {col.COLUMN_NAME} ({col.DATA_TYPE})")

# ---------- Testing ----------

if __name__ == "__main__":
    try:
        reg_conn = get_register_connection()
        att_conn = get_attendance_connection()
        emp_conn = get_employee_info_connection()
        print("✅ All connections succeeded.")

        # Optional: list column structure
        list_columns(reg_conn, "RegisteredFaces")
        list_columns(att_conn, "emp_attendance")
        list_columns(emp_conn, "EmployeeList")

        # Test queries
        try:
            cursor = reg_conn.cursor()
            cursor.execute("SELECT TOP 1 face_id FROM RegisteredFaces")
            row = cursor.fetchone()
            print("🧑‍🦰 RegisteredFaces ➜", f"{row.face_id}" if row else "⚠️ No data in RegisteredFaces")
        except Exception as e:
            print("❌ Error querying RegisteredFaces:", e)

        try:
            cursor = att_conn.cursor()
            cursor.execute("SELECT TOP 1 emp_id, in_out, att_date FROM emp_attendance")
            row = cursor.fetchone()
            print("🕒 emp_attendance ➜", f"{row.emp_id} - {row.in_out} at {row.att_date}" if row else "⚠️ No data in emp_attendance")
        except Exception as e:
            print("❌ Error querying emp_attendance:", e)

        try:
            cursor = emp_conn.cursor()
            cursor.execute("SELECT TOP 1 id, full_name FROM EmployeeList")
            row = cursor.fetchone()
            print("👤 EmployeeList ➜", f"{row.id} - {row.full_name}" if row else "⚠️ No data in EmployeeList")
        except Exception as e:
            print("❌ Error querying EmployeeList:", e)

    except Exception as e:
        print("❌ Connection or query failed:", e)

    finally:
        for conn in [locals().get('reg_conn'), locals().get('att_conn'), locals().get('emp_conn')]:
            if conn:
                conn.close()
