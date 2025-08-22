# admin_enhanced.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from database_enhanced import get_sales_insights, update_order_status, get_order_history
import json

def main():
    st.set_page_config(page_title="Admin Dashboard", layout="wide")
    st.title("🍔 Restaurant Admin Dashboard")
    
    # Get sales insights
    insights = get_sales_insights()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Revenue", f"${insights['total_revenue']:.2f}")
    col2.metric("Total Orders", insights['total_orders'])
    col3.metric("Best Seller", insights['best_selling'][0][0] if insights['best_selling'] else "N/A")
    
    # Status distribution for metric
    status_dict = dict(insights['status_counts'])
    delivered_count = status_dict.get('Delivered', 0)
    col4.metric("Orders Delivered", delivered_count)
    
    # Charts row
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # Best selling items chart
        if insights['best_selling']:
            st.subheader("Best Selling Items")
            best_df = pd.DataFrame(insights['best_selling'], columns=['Item', 'Quantity'])
            fig = px.bar(best_df, x='Item', y='Quantity', color='Quantity')
            st.plotly_chart(fig, use_container_width=True)
    
    with chart_col2:
        # Order status pie chart
        st.subheader("Order Status Distribution")
        if insights['status_counts']:
            status_df = pd.DataFrame(insights['status_counts'], columns=['Status', 'Count'])
            fig = px.pie(status_df, values='Count', names='Status')
            st.plotly_chart(fig, use_container_width=True)
    
    # Recent orders table
    st.subheader("Recent Orders (Last 7 Days)")
    if insights['recent_orders']:
        recent_df = pd.DataFrame(insights['recent_orders'], 
                               columns=['ID', 'Customer', 'Total', 'Status', 'Time'])
        st.dataframe(recent_df, use_container_width=True)
    else:
        st.info("No recent orders.")
    
    # Order management section
    st.subheader("Order Management")
    
    # Get all orders
    all_orders = []
    try:
        # This would need to be implemented to fetch all orders
        # For now, we'll use a placeholder
        pass
    except:
        st.error("Could not load orders. Please check database connection.")
    
    if all_orders:
        for order in all_orders:
            with st.expander(f"Order #{order['id']} - {order['status']} - ${order['total']:.2f}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Customer:** {order['name']}")
                    st.write(f"**Phone:** {order['phone']}")
                    st.write(f"**Address:** {order['address']}")
                    st.write(f"**Time:** {order['time']}")
                
                with col2:
                    st.write("**Items:**")
                    for item in order['items']:
                        st.write(f"- {item['name']} x{item['quantity']} - ${item['price'] * item['quantity']:.2f}")
                    
                    st.write(f"**Total:** ${order['total']:.2f}")
                
                # Status update
                new_status = st.selectbox(
                    "Update Status",
                    ['Pending', 'Confirmed', 'Preparing', 'Out for Delivery', 'Delivered', 'Cancelled'],
                    index=['Pending', 'Confirmed', 'Preparing', 'Out for Delivery', 'Delivered', 'Cancelled'].index(order['status']),
                    key=f"status_{order['id']}"
                )
                
                if st.button("Update Status", key=f"update_{order['id']}"):
                    if update_order_status(order['id'], new_status):
                        st.success("Status updated!")
                        st.rerun()
                    else:
                        st.error("Failed to update status.")
    else:
        st.info("No orders to display.")

if __name__ == "__main__":
    main()