import sqlite3

def get_connection():
    conn = sqlite3.connect("medguardai.db")
    return conn
