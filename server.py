from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from recommender import (
    classify_input,
    recommend_tools,
    answer_business_doubt_local,
    get_daily_news
)

app = Flask(__name__)
CORS(app)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message", "")

    # News shortcut
    if user_input.strip().lower() == "news":
        news = get_daily_news()
        return jsonify({
            "type": "news",
            "news": news
        })

    input_type = classify_input(user_input)

    if input_type == "usecase":
        tools = recommend_tools(user_input)
        return jsonify({
            "type": "usecase",
            "tools": tools if tools else []
        })
    else:
        answer = answer_business_doubt_local(user_input)
        return jsonify({
            "type": "doubt",
            "answer": answer
        })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
