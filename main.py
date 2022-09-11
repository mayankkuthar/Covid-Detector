from flask import Flask, render_template,redirect,url_for,request,flash
from wtforms_field import *
from models import *
from werkzeug.utils import secure_filename
import os
from support_function import *
from flask_login import LoginManager, login_user, current_user , login_required, logout_user
from Solution import ml_solution

app = Flask(__name__)

app.config['UPLOAD_LOCATION']=os.getcwd()+"\\static"
app.config['ALLOWED_EXTENSIONS']=set(['png', 'jpg', 'jpeg', 'gif','PNG','JPG','JPEG','GIF'])

'''Secret Key to keep our client session secure, it will be used to assign cookies used during the session.'''
app.secret_key=os.urandom(24)

app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://divstoccuiteph:be2481247739318660e5ec5093a985db404792d054c5e5d80000db6271adcf87@ec2-52-200-215-149.compute-1.amazonaws.com:5432/d7rkrhsqtqqjij'
db = SQLAlchemy(app)

# Configure flask login
login = LoginManager(app) 
login.init_app(app)

@login.user_loader
def load_user(id):
    return User.query.get(int(id))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/policy')
def policy():
    return render_template('policy.html')

@app.route('/upload', methods=['GET','POST'])
@login_required
def upload():
    status='No POST request triggered'
    image =''
    ans = ''
    file=''
    if request.method == 'POST':
        if 'chest_xray' not in request.files:
            status = "No object with name 'chest_xray' in the request"
        else:
            chest_xray=request.files["chest_xray"]
            if chest_xray.filename == "":
                status="No ppt template selected to upload"
            else:
                if is_file_image(chest_xray.filename, app.config['ALLOWED_EXTENSIONS']):
                    file = secure_filename(chest_xray.filename)
                    chest_xray.save(os.path.join(app.config['UPLOAD_LOCATION'],file))
                    status="Chest-XRAY successfully uploaded"
                    ans = ml_solution(file, os.path.join(app.config['UPLOAD_LOCATION'])+"/"+file)
                else:
                    status="File format do not match"
    return render_template('dashboard.html', status =status, image = file, user = current_user, ans=ans['answer'])

@app.route('/signup', methods=['GET','POST'])
def signup():

    signup_f = SignUpForm() # instance of my flask-form class 'wtforms_filed'
    if signup_f.validate_on_submit(): #this method will return true if form submitted using POST and all validators passes
        # Take data from form
        fullname = signup_f.fullname.data
        email = signup_f.email.data
        username = signup_f.username.data
        password = signup_f.password.data

        #hashing password using passlib
        hash_pwd = pbkdf2_sha256.hash(password) # default salt_size= 16 bytes rounds = 2900
        
        #Added User to Database
        user = User(fullname=fullname,username=username,email=email,password=hash_pwd) # create object of table
        db.session.add(user) # add object
        db.session.commit() # commit changes
        return redirect(url_for('login'))

    return render_template('signup.html', form=signup_f)

@app.route('/dashboard/<username>', methods=['GET','POST'])
def dashboard(username):
    if not current_user.is_authenticated:
        return redirect(url_for('unauthenticated'))
    #user = User.query.filter_by(username=username).first()
    return render_template('dashboard.html', user=current_user, status='')

@app.route('/unauthenticated', methods=['GET','POST'])
def unauthenticated():
    return render_template('unauthenticated.html')

@app.route('/login', methods=['GET','POST'])
def login():

    login_f = LogInForm() # instance of my flask-form class 'wtforms_filed'
    if login_f.validate_on_submit(): #this method will return true if form submitted using POST and all validators passes
        # Take data from form
        username = login_f.username.data
        user_object= User.query.filter_by(username = login_f.username.data).first()
        login_user(user_object)
        return redirect(url_for('dashboard',username=username))
    return render_template('login.html', form=login_f)
    
@app.route("/logout", methods=['GET'])
def logout():
    logout_user()
    return render_template('logout.html')

if __name__ == '__main__':
    app.run(port=9000, debug=True)
 