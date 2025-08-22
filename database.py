# database.py
import sqlite3
import json
from datetime import datetime

DB_PATH = 'orders.db'

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # existing tables
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

    # new: conversation log table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,             -- 'user' or 'assistant' or 'system'
            content TEXT,
            metadata TEXT,         -- json blob (additional_kwargs, response_metadata etc.)
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # new: api usage logging table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS api_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            endpoint TEXT,
            model TEXT,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            total_tokens INTEGER,
            prompt_time REAL,
            completion_time REAL,
            queue_time REAL,
            duration REAL,
            finish_reason TEXT,
            raw_metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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

    cursor.execute('SELECT items FROM orders WHERE status != "Cancelled"')
    rows = cursor.fetchall()
    item_counts = {}
    for row in rows:
        items = json.loads(row[0])
        for item in items:
            name = item['name']
            item_counts[name] = item_counts.get(name, 0) + item['quantity']
    best_selling = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)[:10]

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

# ---------------------------
# Conversation & API usage helpers
# ---------------------------

def log_conversation(session_id: str, role: str, content: str, metadata: dict | None = None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('INSERT INTO conversations (session_id, role, content, metadata, created_at) VALUES (?, ?, ?, ?, ?)',
                   (session_id, role, content, json.dumps(metadata or {}), datetime.now()))
    conn.commit()
    conn.close()

def get_conversations(session_id: str | None = None, limit: int = 200):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    if session_id:
        cursor.execute('SELECT id, session_id, role, content, metadata, created_at FROM conversations WHERE session_id = ? ORDER BY created_at DESC LIMIT ?', (session_id, limit))
    else:
        cursor.execute('SELECT id, session_id, role, content, metadata, created_at FROM conversations ORDER BY created_at DESC LIMIT ?', (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [{"id": r[0], "session_id": r[1], "role": r[2], "content": r[3], "metadata": json.loads(r[4] or "{}"), "created_at": r[5]} for r in rows]

def log_api_usage(metadata: dict, session_id: str | None = None, endpoint: str | None = None):
    """
    metadata: freely structured dict; we try to extract known fields if present.
    """
    try:
        model = metadata.get('model') or metadata.get('model_name') or None
        token_usage = metadata.get('token_usage') or metadata.get('usage') or {}
        prompt_tokens = int(token_usage.get('prompt_tokens') or token_usage.get('input_tokens') or 0)
        completion_tokens = int(token_usage.get('completion_tokens') or token_usage.get('output_tokens') or 0)
        total_tokens = int(token_usage.get('total_tokens') or token_usage.get('total') or (prompt_tokens + completion_tokens))
        prompt_time = float(metadata.get('prompt_time') or metadata.get('prompt_time_secs') or 0.0)
        completion_time = float(metadata.get('completion_time') or metadata.get('completion_time_secs') or 0.0)
        queue_time = float(metadata.get('queue_time') or 0.0)
        duration = float(metadata.get('duration') or 0.0)
        finish_reason = str(metadata.get('finish_reason') or "")
    except Exception:
        # Defensive defaults if metadata shape is different
        model = metadata.get('model') if isinstance(metadata, dict) else None
        prompt_tokens = completion_tokens = total_tokens = 0
        prompt_time = completion_time = queue_time = duration = 0.0
        finish_reason = ""

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO api_usage (session_id, endpoint, model, prompt_tokens, completion_tokens, total_tokens, prompt_time, completion_time, queue_time, duration, finish_reason, raw_metadata, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (session_id, endpoint, model, prompt_tokens, completion_tokens, total_tokens, prompt_time, completion_time, queue_time, duration, finish_reason, json.dumps(metadata or {}), datetime.now()))
    conn.commit()
    conn.close()

def get_api_usage(limit: int = 200):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT id, session_id, endpoint, model, prompt_tokens, completion_tokens, total_tokens, duration, finish_reason, raw_metadata, created_at FROM api_usage ORDER BY created_at DESC LIMIT ?', (limit,))
    rows = cursor.fetchall()
    conn.close()
    out = []
    for r in rows:
        out.append({
            "id": r[0],
            "session_id": r[1],
            "endpoint": r[2],
            "model": r[3],
            "prompt_tokens": r[4],
            "completion_tokens": r[5],
            "total_tokens": r[6],
            "duration": r[7],
            "finish_reason": r[8],
            "raw_metadata": json.loads(r[9] or "{}"),
            "created_at": r[10]
        })
    return out

def get_api_usage_summary():
    """
    Produce simple aggregates: totals and averages.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT SUM(prompt_tokens), SUM(completion_tokens), SUM(total_tokens), COUNT(*) FROM api_usage')
    sums = cursor.fetchone()
    cursor.execute('SELECT AVG(total_tokens), AVG(duration) FROM api_usage')
    avgs = cursor.fetchone()
    conn.close()
    return {
        "total_prompt_tokens": int(sums[0] or 0),
        "total_completion_tokens": int(sums[1] or 0),
        "total_tokens": int(sums[2] or 0),
        "total_calls": int(sums[3] or 0),
        "avg_tokens_per_call": float(avgs[0] or 0.0),
        "avg_duration_secs": float(avgs[1] or 0.0)
    }


# ensure DB exists / initialized on import
init_db()
