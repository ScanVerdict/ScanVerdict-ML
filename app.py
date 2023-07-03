from flask import Flask, request
from execute_analysis import execute_analysis
from analyze_review import single_review

app = Flask(__name__)

@app.route("/get_place_details", methods=["POST"])
def get_place_details():
    global my_place_id
    data = request.get_json()
    my_place_id = data["place_id"]
    return execute_analysis(my_place_id)

@app.route("/get_review", methods=["POST"])
def get_review():
    global my_review
    data = request.get_json()
    my_review = data["place_id"]
    return single_review(my_review)

if __name__ == "__main__":
    app.run(port=5000)