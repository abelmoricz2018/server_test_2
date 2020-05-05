from flask import Flask 
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

def create_app():
    app = Flask(__name__, static_folder='../../build/static', template_folder='../../build')

    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'

    db.init_app(app)

 

    return app

from flask import Blueprint, jsonify, request, render_template

from models import Movie
import base64

import os

def run_script(layer, channel):
    my_cmd = 'cd ../DD && python deepdream_api.py -p ' + layer + ' ' + str(channel) + ' test.jpg'
    os.system(my_cmd)
    print("os cmd finshed")

def convert_img():
    with open("../DD/outputs/test.jpg", "rb") as img_file:
        my_string = base64.b64encode(img_file.read())
    return my_string

main = create_app()

@main.route('/')
def index():
	return render_template('index.html')




@main.route('/add_movie', methods=['POST'])
def add_movie():



    #print(my_string)
    #my_string = "myimage"


    movie_data = request.get_json()

    run_script(movie_data['title'], movie_data['rating'])

    my_string = str(convert_img())

    new_movie = Movie(title=movie_data['title'], rating=movie_data['rating'],
     image=my_string)

    db.session.add(new_movie)
    db.session.commit()

    return 'Done', 201

@main.route('/movies')
def movies():
    movie_list = Movie.query.all()
    movies = []

    for movie in movie_list:
        movies.append({'title' : movie.title, 'rating' : movie.rating, 'image' : movie.image})

    return jsonify({'movies' : movies})



if __name__=='__main__':
	main.run(host='0.0.0.0', port=80)
