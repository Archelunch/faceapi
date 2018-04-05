from flask_wtf import Form
from wtforms import TextField, BooleanField, PasswordField, validators
from wtforms.validators import Required

class RegistrationForm(Form):
    username = TextField('Username', [validators.Length(min=4, max=20), validators.Required()])
    email = TextField('Email Address', [validators.Length(min=6, max=50), validators.Required()])
    password = PasswordField('New Password', [
        validators.Required(),
        validators.EqualTo('confirm', message='Passwords must match')
    ])
    confirm = PasswordField('Repeat Password')
    accept_tos = BooleanField('I accept the Terms of Service and Privacy Notice (updated Jan 22, 2015)', [validators.Required()])


class LoginForm(Form):
    username_email = TextField('Username or email', [validators.Length(min=4, max=50), validators.Required()])
    password = PasswordField('New Password', [
        validators.Required(),
        validators.EqualTo('confirm', message='Passwords must match')
    ])