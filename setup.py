import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def download_spacy_model():
    """Download spaCy language model"""
    print("🔤 Downloading spaCy language model...")
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        print("✅ spaCy model downloaded successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download spaCy model: {e}")
        print("💡 You can skip this and the system will work with reduced NLU capabilities")
        return False

def setup_environment():
    """Setup environment variables"""
    env_file = Path(".env")
    
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    print("🔧 Setting up environment variables...")
    
    groq_key = input("Enter your GROQ API key (required): ").strip()
    if not groq_key:
        print("❌ GROQ API key is required")
        return False
    
    openai_key = input("Enter your OpenAI API key (optional, press Enter to skip): ").strip()
    anthropic_key = input("Enter your Anthropic API key (optional, press Enter to skip): ").strip()
    
    env_content = f"GROQ_API_KEY={groq_key}\n"
    if openai_key:
        env_content += f"OPENAI_API_KEY={openai_key}\n"
    if anthropic_key:
        env_content += f"ANTHROPIC_API_KEY={anthropic_key}\n"
    
    try:
        with open(".env", "w") as f:
            f.write(env_content)
        print("✅ Environment file created")
        return True
    except Exception as e:
        print(f"❌ Failed to create environment file: {e}")
        return False

def initialize_database():
    """Initialize the database"""
    print("🗄️ Initializing database...")
    try:
        from database import init_db
        init_db()
        print("✅ Database initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize database: {e}")
        return False

def setup_menu_data():
    """Ensure menu data is available"""
    enhanced_menu = Path("menu.json")
    
    if enhanced_menu.exists():
        print("✅ Enhanced menu data found")
        return True
    else:
        print("❌ No menu data found")
        print("💡 Please ensure menu.json exists")
        return False

def run_tests():
    
    print("🧪 Running system tests...")
    try:
        from test_system import run_all_tests
        success = run_all_tests()
        if success:
            print("✅ All tests passed")
        else:
            print("⚠️ Some tests failed, but system should still work")
        return True
    except Exception as e:
        print(f"⚠️ Could not run tests: {e}")
        print("💡 System should still work, but testing is recommended")
        return True

def create_startup_script():
    """Create startup script for easy launching"""
    startup_content = """#!/bin/bash
# AI Food Ordering System Startup Script

echo "🍽️ Starting AI Food Ordering System..."

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
fi

# Start the application
echo "🚀 Launching application..."
streamlit run advanced_app.py

echo "✅ Application started! Open http://localhost:8501 in your browser"
"""
    
    try:
        with open("start.sh", "w") as f:
            f.write(startup_content)
        os.chmod("start.sh", 0o755)
        print("✅ Startup script created (start.sh)")
        return True
    except Exception as e:
        print(f"⚠️ Could not create startup script: {e}")
        return True

def main():
    """Main setup function"""
    print("🍽️ AI Food Ordering System Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Download spaCy model (optional)
    download_spacy_model()
    
    # Setup environment
    if not setup_environment():
        return False
    
    # Initialize database
    if not initialize_database():
        return False
    
    # Check menu data
    if not setup_menu_data():
        return False
    
    # Run tests
    run_tests()
    
    # Create startup script
    create_startup_script()
    
    print("\n🎉 Setup completed successfully!")
    print("\n🚀 To start the application:")
    print("   Option 1: ./start.sh")
    print("   Option 2: streamlit run advanced_app.py")
    print("\n📖 Open http://localhost:8501 in your browser")
    print("\n📚 Check README.md for detailed documentation")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)