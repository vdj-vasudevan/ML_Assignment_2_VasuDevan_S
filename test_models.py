#!/usr/bin/env python3
"""
Test script to check model loading and dependencies
"""

import pickle
import sys

def test_xgboost():
    """Test if XGBoost is installed"""
    try:
        import xgboost
        print("✅ XGBoost is installed (version: {})".format(xgboost.__version__))
        return True
    except ImportError:
        print("❌ XGBoost is NOT installed")
        print("   Install with: pip install xgboost")
        return False

def test_model_loading():
    """Test loading all models"""
    model_names = ['logistic_regression', 'decision_tree', 'knn', 'naive_bayes', 'random_forest', 'xgboost']
    loaded = []
    failed = []
    
    print("\nTesting model loading:")
    print("-" * 50)
    
    for name in model_names:
        try:
            with open(f'model/saved_models/{name}.pkl', 'rb') as f:
                model = pickle.load(f)
            print(f"✅ {name.replace('_', ' ').title()}: OK")
            loaded.append(name)
        except FileNotFoundError:
            print(f"❌ {name.replace('_', ' ').title()}: File not found")
            failed.append(name)
        except Exception as e:
            print(f"❌ {name.replace('_', ' ').title()}: {str(e)}")
            failed.append(name)
    
    print("-" * 50)
    print(f"\nSummary: {len(loaded)}/{len(model_names)} models loaded successfully")
    
    if failed:
        print(f"\nFailed models: {', '.join(failed)}")
        return False
    return True

def test_dependencies():
    """Test all required dependencies"""
    dependencies = [
        'streamlit',
        'pandas',
        'numpy',
        'sklearn',
        'matplotlib',
        'seaborn',
        'xgboost'
    ]
    
    print("\nTesting dependencies:")
    print("-" * 50)
    
    all_ok = True
    for dep in dependencies:
        try:
            if dep == 'sklearn':
                import sklearn
                module = sklearn
            else:
                module = __import__(dep)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {dep}: {version}")
        except ImportError:
            print(f"❌ {dep}: NOT INSTALLED")
            all_ok = False
    
    print("-" * 50)
    return all_ok

if __name__ == "__main__":
    print("=" * 50)
    print("Model Loading Test")
    print("=" * 50)
    
    # Test dependencies
    deps_ok = test_dependencies()
    
    # Test XGBoost specifically
    xgb_ok = test_xgboost()
    
    # Test model loading
    models_ok = test_model_loading()
    
    print("\n" + "=" * 50)
    if deps_ok and models_ok:
        print("✅ All tests passed! Ready to run the app.")
        print("\nRun: streamlit run app.py")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        if not xgb_ok:
            print("\n💡 To install XGBoost: pip install xgboost")
        sys.exit(1)
