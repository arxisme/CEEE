import sys
import platform

def check_environment():
    print("="*50)
    print("🔍 ML Lab Environment Diagnostic Tool")
    print("="*50)
    
    # 1. Check Python Version
    print(f"[INFO] Operating System: {platform.system()} {platform.release()}")
    py_version = sys.version_info
    print(f"[INFO] Python Version: {py_version.major}.{py_version.minor}.{py_version.micro}")
    
    if py_version.major < 3 or (py_version.major == 3 and py_version.minor < 8):
        print("❌ ERROR: Python 3.8 or higher is required.")
        return
    else:
        print("✅ Python version is compatible.")

    print("-" * 50)
    print("Checking required libraries...\n")

    # 2. Check Libraries
    libraries = {
        "numpy": "Data Manipulation",
        "pandas": "Data Analysis",
        "sklearn": "Machine Learning",
        "matplotlib": "Data Visualization",
        "torch": "Deep Learning (PyTorch)",
        "jupyter": "Jupyter Notebooks"
    }
    all_passed = True

    for lib, description in libraries.items():
        try:
            module = __import__(lib)
            version = getattr(module, '__version__', 'Unknown Version')
            print(f"✅ {lib.ljust(12)} : Installed (v{version}) - {description}")
        except ImportError:
            print(f"❌ {lib.ljust(12)} : NOT INSTALLED - {description}")
            all_passed = False

    print("-" * 50)
    if all_passed:
        print("🎉 SUCCESS: Your ML environment is perfectly set up!")
        print("You are ready for the Lab classes.")
    else:
        print("⚠️ WARNING: Some packages are missing.")
        print("Please run: pip install -r requirements.txt")
    print("="*50)

if __name__ == "__main__":
    check_environment()