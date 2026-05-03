import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from app import app, load_and_train

print("Training model on startup…")
load_and_train()
print("Model ready.")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
