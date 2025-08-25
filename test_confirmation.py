#!/usr/bin/env python3
"""
Quick test for order confirmation flow
"""
import uuid
from core.ai_agent import process_query

def test_order_confirmation():
    """Test the exact scenario from the conversation"""
    session_id = str(uuid.uuid4())
    
    # Simulate the conversation flow
    responses = []
    
    print("🧪 Testing Order Confirmation Flow")
    print("=" * 50)
    
    # Step 1: Add items
    print("\n1. Adding items to cart...")
    response1 = process_query("order 2 margherita pizza", session_id)
    print(f"Response: {response1[:100]}...")
    responses.append(response1)
    
    # Step 2: Specify preferences
    print("\n2. Adding preferences...")
    response2 = process_query("small, thin, no extras, delivery", session_id)
    print(f"Response: {response2[:100]}...")
    responses.append(response2)
    
    # Step 3: Provide details
    print("\n3. Adding customer details...")
    response3 = process_query("plain cheese margherita, no sides, palanpur canal road B-702, UPI on delivery", session_id)
    print(f"Response: {response3[:100]}...")
    responses.append(response3)
    
    # Step 4: Provide name and phone
    print("\n4. Providing customer info...")
    response4 = process_query("Aditya Jethani, 9328223890", session_id)
    print(f"Response: {response4[:100]}...")
    responses.append(response4)
    
    # Step 5: CRITICAL TEST - Final confirmation
    print("\n5. 🎯 CRITICAL TEST - Final order confirmation...")
    response5 = process_query("yes finalize", session_id)
    print(f"Final Response: {response5}")
    
    # Check if order was actually placed
    if "ORDER CONFIRMED" in response5 and "Order #" in response5:
        print("\n✅ SUCCESS: Order was confirmed and order ID generated!")
        return True
    else:
        print("\n❌ FAILED: Order was not properly confirmed")
        print(f"Final response: {response5}")
        return False

if __name__ == "__main__":
    test_order_confirmation()
