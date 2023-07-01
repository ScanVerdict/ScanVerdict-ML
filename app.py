from flask import Flask, request
from execute_analysis import execute_analysis

app = Flask(__name__)

my_place_id = ""

@app.route("/get_place_details", methods=["POST"])
def get_place_details():
    global my_place_id
    data = request.get_json()
    my_place_id = data["place_id"]
    return "Received place id: " + my_place_id

if __name__ == "__main__":
    app.run(port=5000)