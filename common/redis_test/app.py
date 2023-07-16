# gunicorn -w 4 -b 0.0.0.0 'app:app'
from flask import Flask, request
import redis
from redis.exceptions import ResponseError

app = Flask(__name__)

pool = redis.ConnectionPool(host='localhost', port=6379, db=0)
r = redis.Redis(connection_pool=pool)

LIMITS = {
    'audio': 5,  # limit for audio service
    'image': 10,  # limit for image service
}

@app.route("/audio")
def audio_service():
    return handle_request('audio')

@app.route("/image")
def image_service():
    return handle_request('image')

def handle_request(service):
    user_id = request.args.get('uuid')

    key = f"user:{user_id}:{service}"

    try:
        # Using Redis' pipeline to perform two atomic operations
        # The first one is to set the value of the key to 1, or if the key exists to increment it by 1
        # The second one is to set the key to expire in 86400 seconds (one day)
        pipe = r.pipeline()
        pipe.incr(key)
        pipe.expire(key, 86400)
        count = pipe.execute()[0]

        if count > LIMITS[service]:
            return "Usage limit exceeded", 429
    except ResponseError:
        return "Error processing request", 500

    # Process request ...
    return f"{service.capitalize()} service executed successfully."

if __name__ == "__main__":
    app.run()