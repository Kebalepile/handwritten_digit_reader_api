# gunicorn_config.py
bind = "0.0.0.0:8000"
workers = 4  # Adjust the number of workers based on your server's capacity
timeout = 180  # Set the timeout to 180 seconds (3 minutes)
