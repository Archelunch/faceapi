from requests import get, post
from io import BytesIO
from json import loads, dumps

import api
from tools import get_secret_key, check_file, decode_image, mkdir, isExist, getFile, createFile
from api import detect_faces, compare_photos, extract_descriptor, get_landmarks, swap_face, swap_video

from flask import Flask, render_template, request, session, redirect, make_response, send_file, jsonify, url_for

#auth = HTTPBasicAuth()
#r = redis.StrictRedis(host='127.0.0.1', port=6379)
app = Flask(__name__)
app.secret_key = get_secret_key()
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 16

##@auth.get_password
# def get_password(username):
#     user_info = r.get(username)
#     if user_info == None:
#         return None
#     else:
#         user_info = loads(user_info)
#         return user_info[0]['password']


#@auth.error_handler
def unauthorized():
    return make_response(jsonify({'error': 'Unauthorized access'}), 401)


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


@app.route('/api/face_detect', methods=['POST'])
#@auth.login_required
def face_detect():
    img = request.files['img']
    if check_file(img):
        try:
            faces = detect_faces(decode_image(img))[1]
            return jsonify({'result': True, 'faces':{i: 4*faces[i].tolist() for i in range(len(faces))}})
        except Exception as e:
            return jsonify({"result": False, "error_msg": str(e)})
    else:
        return jsonify({'result': False, 'error_msg': 'File load failed'})


@app.route('/api/compare_faces', methods=['POST'])
#@auth.login_required
def compare_faces():
    img1 = request.files['img1']
    img2 = request.files['img2']
    if check_file(img1) and check_file(img2):
        try:
            mn = compare_photos(img1, img2, request.values.get("alg"))
            return jsonify({"result": True, "mn": mn})
        except Exception as e:
            return jsonify({"result": False, "error_msg": str(e)})
    else:
        return jsonify({'result': False, 'error_msg': 'File load failed'})


@app.route('/api/facetovec', methods=['POST'])
#@auth.login_required
def facetovec():
    img = request.files['img']
    if check_file(img):
        try:
            descs = extract_descriptor(decode_image(img))
            return jsonify({"result": True, 'descs':{i: list(descs[i]) for i in range(len(descs))}})
        except Exception as e:
            return jsonify({"result": False, "error_msg": str(e)})
    else:
        return jsonify({'result': False, 'error_msg': 'File load failed'})


@app.route('/api/face_landmarks', methods=['POST'])
#@auth.login_required
def face_landmarks():
    img = request.files['img']
    if check_file(img):
        try:
            descs = get_landmarks(decode_image(img), False)
            return jsonify({"result": True, 'descs':{i: descs[i].tolist() for i in range(len(descs))}})
        except Exception as e:
            return jsonify({"result": False, "error_msg": str(e)})
    else:
        return jsonify({'result': False, 'error_msg': 'File load failed'})


@app.route('/api/face_swap', methods=['POST'])
#@auth.login_required
def face_swap():
    img1 = request.files['img1']
    img2 = request.files['img2']
    #video = request.files['video']
    if check_file(img1) and check_file(img2):
        try:
            name = swap_face(createFile(auth.username(), img1), createFile(auth.username(), img2), auth.username())
            return jsonify({"result": True, 'URL':url_for('download_image',_external=True, username=auth.username(), filename=name)})
        except Exception as e:
            return jsonify({"result": False, "error_msg": str(e)})
    # elif check_file(video):
    #     try:
    #         swap_face(decode_image(img1), decode_image(img2), 1)
    #     except Exception as e:
    #         return jsonify({"result": False, "error_msg": str(e)})
    else:
        return jsonify({'result': False, 'error_msg': 'File load failed'})


@app.route('/api/nearest/<dataset>', methods=['POST'])
#@auth.login_required
def search_nearest(dataset):
    img = request.files['img']
    id_i = request.form['id']
    if check_file(img):
        try:
            descs = api.get_user(decode_image(img))
            return jsonify({"result": True, 'name':descs[0][1]['name'], 'id':id_i})
        except Exception as e:
            return jsonify({"result": False, "error_msg": str(e),'id':id_i})
    else:
        return jsonify({'result': False, 'error_msg': 'File load failed', 'id':id_i})


@app.route('/api/enable_dataset/<dataset>', methods=['POST'])
#@auth.login_required
def enable_dataset(dataset):
    api.user_tree = api.vptree.VPTree(api.loadDescriptors(dataset), api.cosine)
    return jsonify({'result': True})


# @app.route('/api/register', methods=['POST'])
# def register():
#     try:
#         username = request.values.get("username")
#         if get_password(username) == None:
#             password = get_secret_key()
#             user_info = dumps([{'password':password, 'datasets':[]}])
#             r.set(username, user_info)
#             mkdir(username)
#             return jsonify({"result": True, "username": username, "password":password, "datasets":[]})
#         else:
#             return jsonify({"result": False, "error_msg": 'This username already exists'})
#     except Exception as e:
#         return jsonify({"result": False, "error_msg": str(e)})


@app.route('/api/download/<username>/<filename>', methods=['GET'])
#@auth.login_required
def download_image(username, filename):
    try:
        if username != auth.username():
            return make_response(jsonify({'error': 'Not allowed acces'}), 403)
        if not isExist(username, filename):
            return make_response(jsonify({'error': 'File not found'}), 404)
        return send_file(getFile(username, filename))    
    except Exception as e:
        return jsonify({"result": False, "error_msg": str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050,  debug=True)
