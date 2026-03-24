import os
from flask import Flask, request, render_template

app = Flask(__name__)
# Ensure templates and static directories exist
os.makedirs('templates', exist_ok=True)
os.makedirs('static/uploads', exist_ok=True)
