"""
Simple startup script for Medicinal Leaf Classification Website
"""

import os
import sys

def main():
    """Start the website"""
    print("="*60)
    print("MEDICINAL LEAF CLASSIFICATION WEBSITE")
    print("="*60)
    print("Backend: http://localhost:5000")
    print("Frontend: Open index.html in your browser")
    print("Press Ctrl+C to stop the server")
    print("="*60)
    
    try:
        from app_simple import app, initialize_app
        initialize_app()
        app.run(debug=False, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
