from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, EmailField, SubmitField
from wtforms.validators import InputRequired, Length, EqualTo, ValidationError
from models import *
from passlib.hash import pbkdf2_sha256

# Explicit Custom validator
def invalid_cred(form,field): # Here form is LogInForm and field is password as invalid_cred is called in password field of LogInForm
    '''Username and Password Checker'''
    username_entered = form.username.data # access username field through form as field is password
    password_entered = field.data # data in password field
    message = "Bad Username OR Password !!!"
    user_instance = User.query.filter_by(username=username_entered).first()
    if not user_instance: # if user not found in datatbase
        raise ValidationError(message=message)
    else:
        if not pbkdf2_sha256.verify(password_entered, user_instance.password): # if entered password is not same as stored password
            raise ValidationError(message=message)

class SignUpForm(FlaskForm):
    '''SignUp Form'''
    fullname = StringField('Fullname', validators=[InputRequired(message="Fullname Required"), Length(
        min=2, message="Fullname must be more than characters")])
    email = EmailField('Email',validators=[InputRequired(message="Email Required")])
    username = StringField('Username', validators=[InputRequired(message="Username Required"), Length(
        min=5, max=15, message="Usename must be between 5 to 15 characters")])
    password = PasswordField('Password', validators=[InputRequired(message="Password Required"), Length(
        min=8, max=25, message="Password must be between 8 to 25 characters")])
    confirm_password = PasswordField('Confirm_password', validators=[InputRequired(
        message="Confirm Password Required"), EqualTo('password',message="Password not matching")])
    submit_button = SubmitField('Create Account')

    #Inline Custom Validators

    def validate_username(self,username):
        user_instance = User.query.filter_by(username=username.data).first()
        if user_instance: # if user found in datatbase
            raise ValidationError("Username Already Taken, Please Select Another Username !!!")
    
    def validate_email(self,email):
        user_instance = User.query.filter_by(email=email.data).first()
        if user_instance: # if user found in datatbase
            raise ValidationError("Email Already Used, Please Use Another Email !!!")

class LogInForm(FlaskForm):
    '''LogIn Form'''
    username = StringField('Username', validators=[InputRequired(message="Username Required"), Length(
        min=5, max=15, message="Usename must be between 5 to 15 characters")])
    password = PasswordField('Password', validators=[InputRequired(message="Password Required"), Length(
        min=8, max=25, message="Password must be between 8 to 25 characters"), invalid_cred])
    submit_button = SubmitField('Log In')
