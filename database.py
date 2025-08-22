import sqlite3
import json
from datetime import datetime

DB_PATH = 'orders.db'

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            items TEXT NOT NULL,
            total_price REAL NOT NULL,
            order_time TIMESTAMP NOT NULL,
            customer_name TEXT,
            customer_phone TEXT,
            customer_address TEXT,
            status TEXT DEFAULT 'Pending'
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS carts (
            session_id TEXT PRIMARY KEY,
            items TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def insert_order(session_id, items, total_price, customer_details):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO orders (session_id, items, total_price, order_time, customer_name, customer_phone, customer_address)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (session_id, json.dumps(items), total_price, datetime.now(),
          customer_details.get('name', ''), customer_details.get('phone', ''), customer_details.get('address', '')))
    cursor.execute('DELETE FROM carts WHERE session_id = ?', (session_id,))
    conn.commit()
    conn.close()

def get_order_history(session_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM orders WHERE session_id = ? ORDER BY order_time DESC', (session_id,))
    rows = cursor.fetchall()
    conn.close()
    return [{"id": r[0], "items": json.loads(r[2]), "total": r[3], "time": r[4], "name": r[5], "phone": r[6], "address": r[7], "status": r[8]} for r in rows]

def get_all_orders():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM orders ORDER BY order_time DESC')
    rows = cursor.fetchall()
    conn.close()
    return [{"id": r[0], "session_id": r[1], "items": json.loads(r[2]), "total": r[3], "time": r[4], "name": r[5], "phone": r[6], "address": r[7], "status": r[8]} for r in rows]

def update_order_status(order_id, status):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('UPDATE orders SET status = ? WHERE id = ?', (status, order_id))
    conn.commit()
    conn.close()

def cancel_order(order_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT status FROM orders WHERE id = ?', (order_id,))
    status = cursor.fetchone()
    if status and status[0] == 'Pending':
        cursor.execute('UPDATE orders SET status = ? WHERE id = ?', ('Cancelled', order_id))
        conn.commit()
        conn.close()
        return "Order cancelled successfully."
    conn.close()
    return "Cannot cancel: Order is not in Pending status."

def save_cart(session_id, items):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('INSERT OR REPLACE INTO carts (session_id, items) VALUES (?, ?)',
                   (session_id, json.dumps(items)))
    conn.commit()
    conn.close()

def get_cart(session_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT items FROM carts WHERE session_id = ?', (session_id,))
    row = cursor.fetchone()
    conn.close()
    return json.loads(row[0]) if row else []

def get_sales_insights():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Filtered for non-cancelled
    cursor.execute('SELECT items FROM orders WHERE status != "Cancelled"')
    rows = cursor.fetchall()
    item_counts = {}
    for row in rows:
        items = json.loads(row[0])
        for item in items:
            name = item['name']
            item_counts[name] = item_counts.get(name, 0) + item['quantity']
    best_selling = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)[:10]  # Top 10
    
    cursor.execute('SELECT SUM(total_price) FROM orders WHERE status != "Cancelled"')
    total_revenue = cursor.fetchone()[0] or 0
    
    cursor.execute('SELECT COUNT(*) FROM orders WHERE status != "Cancelled"')
    total_orders = cursor.fetchone()[0]
    
    cursor.execute('SELECT status, COUNT(*) FROM orders GROUP BY status')
    status_counts = cursor.fetchall()
    
    cursor.execute('SELECT DATE(order_time) as date, SUM(total_price) FROM orders WHERE status != "Cancelled" GROUP BY date ORDER BY date')
    daily_sales = cursor.fetchall()
    
    conn.close()
    return {
        'best_selling': best_selling,
        'total_revenue': total_revenue,
        'total_orders': total_orders,
        'status_counts': status_counts,
        'daily_sales': daily_sales
    }

init_db()