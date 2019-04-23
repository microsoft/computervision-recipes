#!/bin/sh
exec gunicorn -b :5000 --access-logfile - --error-logfile - application:app
# Start the Gunicorn web server and specify that the Flask application to be run is in file application.py file