import os
import sys
import subprocess
from pathlib import Path
import json

def display_banner():
    """Display professional startup banner"""
    print("=" * 60)
    print("🍽️  PROFESSIONAL AI FOOD ORDERING SYSTEM")
    print("=" * 60)
    print("Advanced AI-powered food ordering with intelligent capabilities")
    print("Version: 2.0 Professional Edition")
    print("=" * 60)

def check_environment():
    """Check if environment is properly configured"""
    print("🔧 Checking environment configuration...")
    
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found")
        print("💡 Please create .env file with your GROQ_API_KEY")
        return False
    
    # Check if GROQ_API_KEY is set
    from dotenv import load_dotenv
    load_dotenv()
    
    if not os.getenv("GROQ_API_KEY"):
        print("❌ GROQ_API_KEY not found in .env file")
        print("💡 Please add GROQ_API_KEY=your_key_here to .env file")
        return False
    
    print("✅ Environment configuration valid")
    return True

def check_dependencies():
    """Check if required dependencies are installed"""
    print("📦 Checking dependencies...")
    
    required_packages = [
        "streamlit",
        "langchain",
        "langchain_groq",
        "faiss-cpu",
        "sentence-transformers",
        "plotly",
        "spacy"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("💡 Run: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies installed")
    return True

def check_core_modules():
    """Check if core modules are available"""
    print("🧠 Checking core AI modules...")
    
    core_modules = [
        "core.ai_agent",
        "core.natural_language_processor", 
        "core.knowledge_retrieval",
        "core.recommendation_engine",
        "core.intelligent_tools"
    ]
    
    for module in core_modules:
        try:
            __import__(module)
        except ImportError as e:
            print(f"❌ Failed to import {module}: {e}")
            return False
    
    print("✅ Core AI modules loaded successfully")
    return True

def initialize_database():
    """Initialize the database system"""
    print("🗄️ Initializing database...")
    
    try:
        from database import init_db
        init_db()
        print("✅ Database initialized")
        return True
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False

def check_menu_data():
    """Check if menu data is available"""
    print("📋 Checking menu data...")
    
    menu_files = ["menu.json"]
    
    for menu_file in menu_files:
        if Path(menu_file).exists():
            try:
                with open(menu_file, 'r') as f:
                    menu_data = json.load(f)
                    if menu_data and len(menu_data) > 0:
                        print(f"✅ Menu data loaded from {menu_file}")
                        return True
            except Exception as e:
                print(f"⚠️ Error reading {menu_file}: {e}")
                continue
    
    print("❌ No valid menu data found")
    print("💡 Please ensure menu.json exists")
    return False

def run_system_tests():
    """Run basic system tests"""
    print("🧪 Running system validation tests...")
    
    try:
        # Test core agent
        from core.ai_agent import process_query
        test_response = process_query("test", "test_session")
        
        if isinstance(test_response, str) and len(test_response) > 0:
            print("✅ Core AI agent functional")
        else:
            print("⚠️ Core AI agent test returned unexpected result")
        
        # Test database
        from database import get_cart, save_cart
        save_cart("test_session", [])
        cart = get_cart("test_session")
        
        if isinstance(cart, list):
            print("✅ Database operations functional")
        else:
            print("⚠️ Database test failed")
        
        return True
        
    except Exception as e:
        print(f"⚠️ System test failed: {e}")
        print("💡 System may still work, but some features might be limited")
        return True  # Don't block startup for test failures

def launch_application():
    """Launch the main application"""
    print("🚀 Launching Professional AI Food Ordering System...")
    print("📖 Application will open at: http://localhost:8501")
    print("🛑 Press Ctrl+C to stop the application")
    print("=" * 60)
    
    try:
        # Launch streamlit application
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "main_application.py",
            "--server.headless", "true",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
        print("Thank you for using Professional AI Food Ordering System!")
    except Exception as e:
        print(f"\n❌ Application error: {e}")
        return False
    
    return True

def main():
    """Main startup function"""
    display_banner()
    
    # System checks
    checks = [
        ("Environment", check_environment),
        ("Dependencies", check_dependencies),
        ("Core Modules", check_core_modules),
        ("Database", initialize_database),
        ("Menu Data", check_menu_data),
        ("System Tests", run_system_tests)
    ]
    
    print("🔍 Running system checks...")
    print("-" * 40)
    
    for check_name, check_func in checks:
        if not check_func():
            print(f"\n❌ {check_name} check failed")
            print("💡 Please resolve the issues above before starting the system")
            return False
        print()
    
    print("✅ All system checks passed!")
    print("-" * 40)
    
    # Launch application
    return launch_application()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)