from flask import Flask, render_template, request, jsonify
from rag_backend import rag_ask  # 直接导入你的后端逻辑

app = Flask(__name__)

@app.route('/')
def home():
    """Render homepage"""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat request API"""
    data = request.json
    user_query = data.get('query')
    
    if not user_query:
        return jsonify({"error": "Please enter a question"}), 400

    try:
        # Call the core function from rag_backend.py
        answer, sources = rag_ask(user_query)
        
        return jsonify({
            "answer": answer,
            "sources": sources
        })
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Start Flask service
    print("Starting RAG chatbot...")
    app.run(debug=True, port=5000)