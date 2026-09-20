# =============================================================================
# Heart Disease Web-based Prediction System — Flask Application (v2)
# Updated: Admin panel added | Algorithm selector removed from user interface
# =============================================================================

import os, joblib, numpy as np
from datetime import datetime
from functools import wraps

from flask import (Flask, render_template, request, redirect,
                   url_for, flash, abort)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (LoginManager, UserMixin, login_user,
                         logout_user, login_required, current_user)
from werkzeug.security import generate_password_hash, check_password_hash

# Flask Application Setup
app = Flask(__name__)
app.config['SECRET_KEY']               = 'hds-secret-key-change-in-production-2024'
app.config['SQLALCHEMY_DATABASE_URI']  = 'sqlite:///heart_disease.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Database and Login Manager Setup
db            = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view         = 'login'
login_manager.login_message      = 'Please log in to access this page.'
login_manager.login_message_category = 'warning'

# Best model (Logistic Regression — 87.78% accuracy on test set)
BEST_MODEL_NAME = 'Logistic Regression'
MODELS_DIR      = 'models'
best_model      = None
scaler          = None
models_loaded   = False

# Load the best model and scaler from disk
def load_models():
    global best_model, scaler, models_loaded
    try:
        best_model    = joblib.load(os.path.join(MODELS_DIR, 'best_model.pkl'))
        scaler        = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
        models_loaded = True
        print(f"[INFO] Model loaded: {BEST_MODEL_NAME}")
    except FileNotFoundError:
        print("[WARNING] Models not found. Run train_models.py first.")
        models_loaded = False

load_models()

# Database Models
class User(db.Model, UserMixin):
    __tablename__ = 'users'
    id            = db.Column(db.Integer,     primary_key=True)
    username      = db.Column(db.String(80),  unique=True, nullable=False)
    email         = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    is_admin      = db.Column(db.Boolean,     default=False)
    created_at    = db.Column(db.DateTime,    default=datetime.utcnow)
    predictions   = db.relationship('Prediction', backref='user',
                                    lazy=True, cascade='all, delete-orphan')

    def set_password(self, pw):
        self.password_hash = generate_password_hash(pw)
    def check_password(self, pw):
        return check_password_hash(self.password_hash, pw)

# Database model for storing prediction records
class Prediction(db.Model):
    __tablename__     = 'predictions'
    id                = db.Column(db.Integer, primary_key=True)
    user_id           = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    patient_name      = db.Column(db.String(100), default='Anonymous')
    age               = db.Column(db.Integer)
    sex               = db.Column(db.Integer)
    cp                = db.Column(db.Integer)
    trestbps          = db.Column(db.Integer)
    chol              = db.Column(db.Integer)
    fbs               = db.Column(db.Integer)
    restecg           = db.Column(db.Integer)
    thalach           = db.Column(db.Integer)
    exang             = db.Column(db.Integer)
    oldpeak           = db.Column(db.Float)
    slope             = db.Column(db.Integer)
    ca                = db.Column(db.Integer)
    thal              = db.Column(db.Integer)
    algorithm_used    = db.Column(db.String(60))
    prediction_result = db.Column(db.Integer)
    confidence_score  = db.Column(db.Float)
    created_at        = db.Column(db.DateTime, default=datetime.utcnow)

    @property
    def result_label(self):
        return 'Heart Disease Detected' if self.prediction_result == 1 else 'No Heart Disease'
    @property
    def result_class(self):
        return 'danger' if self.prediction_result == 1 else 'success'

# User Loader for Flask-Login
@login_manager.user_loader
def load_user(uid): return User.query.get(int(uid))

# Admin decorator
def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            abort(403)
        return f(*args, **kwargs)
    return decorated

# Feature decoders
SEX_MAP   = {0:'Female', 1:'Male'}
CP_MAP    = {0:'Typical Angina', 1:'Atypical Angina', 2:'Non-anginal Pain', 3:'Asymptomatic'}
FBS_MAP   = {0:'No (≤120 mg/dl)', 1:'Yes (>120 mg/dl)'}
ECG_MAP   = {0:'Normal', 1:'ST-T Wave Abnormality', 2:'LV Hypertrophy'}
EXANG_MAP = {0:'No', 1:'Yes'}
SLOPE_MAP = {0:'Upsloping', 1:'Flat', 2:'Downsloping'}
THAL_MAP  = {0:'Normal', 1:'Fixed Defect', 2:'Reversible Defect', 3:'Unknown'}

# Convert Prediction object to dictionary for display
def prediction_to_dict(p):
    return {
        'Age': p.age, 'Sex': SEX_MAP.get(p.sex, p.sex),
        'Chest Pain Type': CP_MAP.get(p.cp, p.cp),
        'Resting Blood Pressure': f'{p.trestbps} mm Hg',
        'Cholesterol': f'{p.chol} mg/dl',
        'Fasting Blood Sugar': FBS_MAP.get(p.fbs, p.fbs),
        'Resting ECG': ECG_MAP.get(p.restecg, p.restecg),
        'Max Heart Rate': f'{p.thalach} bpm',
        'Exercise Induced Angina': EXANG_MAP.get(p.exang, p.exang),
        'ST Depression (Oldpeak)': p.oldpeak,
        'ST Slope': SLOPE_MAP.get(p.slope, p.slope),
        'Major Vessels (CA)': p.ca,
        'Thalassemia': THAL_MAP.get(p.thal, p.thal),
    }

# ════════════════════════════════════════════════════════════════════════════
# USER ROUTES
# ════════════════════════════════════════════════════════════════════════════
@app.route('/')
def home():
    return render_template('home.html', best_model=BEST_MODEL_NAME)

# Registration route for new users
@app.route('/register', methods=['GET','POST'])
def register():
    if current_user.is_authenticated: return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = request.form.get('username','').strip()
        email    = request.form.get('email','').strip().lower()
        password = request.form.get('password','')
        confirm  = request.form.get('confirm_password','')
        errors   = []
        if len(username) < 3:              errors.append('Username must be at least 3 characters.')
        if '@' not in email:               errors.append('Enter a valid email address.')
        if len(password) < 6:             errors.append('Password must be at least 6 characters.')
        if password != confirm:            errors.append('Passwords do not match.')
        if User.query.filter_by(username=username).first(): errors.append('Username already taken.')
        if User.query.filter_by(email=email).first():       errors.append('Email already registered.')
        if errors:
            for e in errors: flash(e,'danger')
            return render_template('register.html', username=username, email=email)
        u = User(username=username, email=email)
        u.set_password(password)
        db.session.add(u); db.session.commit()
        flash('Account created! Please log in.','success')
        return redirect(url_for('login'))
    return render_template('register.html')

# Login route for existing users
@app.route('/login', methods=['GET','POST'])
def login():
    if current_user.is_authenticated: return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = request.form.get('username','').strip()
        password = request.form.get('password','')
        u = User.query.filter_by(username=username).first()
        if u and u.check_password(password):
            login_user(u, remember='remember' in request.form)
            if u.is_admin:
                flash(f'Welcome, Admin {u.username}!','success')
                return redirect(url_for('admin_dashboard'))
            flash(f'Welcome back, {u.username}!','success')
            return redirect(request.args.get('next') or url_for('dashboard'))
        flash('Invalid username or password.','danger')
    return render_template('login.html')

# Logout route for users
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully.','info')
    return redirect(url_for('home'))

# Dashboard route for Logged-in Users
@app.route('/dashboard')
@login_required
def dashboard():
    if current_user.is_admin: return redirect(url_for('admin_dashboard'))
    preds    = Prediction.query.filter_by(user_id=current_user.id)\
                               .order_by(Prediction.created_at.desc()).all()
    positive = sum(1 for p in preds if p.prediction_result == 1)
    return render_template('dashboard.html', predictions=preds[:5],
                           total=len(preds), positive=positive,
                           negative=len(preds)-positive)

# Prediction route for users to input features and get prediction
@app.route('/predict', methods=['GET','POST'])
@login_required
def predict():
    if current_user.is_admin: return redirect(url_for('admin_dashboard'))
    if not models_loaded:
        flash('Prediction model not loaded. Run train_models.py first.','warning')
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        try:
            patient_name = request.form.get('patient_name','Anonymous').strip() or 'Anonymous'
            features = {
                'age': int(request.form['age']),   'sex': int(request.form['sex']),
                'cp':  int(request.form['cp']),    'trestbps': int(request.form['trestbps']),
                'chol':int(request.form['chol']),  'fbs': int(request.form['fbs']),
                'restecg':int(request.form['restecg']), 'thalach':int(request.form['thalach']),
                'exang':int(request.form['exang']), 'oldpeak':float(request.form['oldpeak']),
                'slope':int(request.form['slope']), 'ca':int(request.form['ca']),
                'thal': int(request.form['thal']),
            }
            order = ['age','sex','cp','trestbps','chol','fbs','restecg',
                     'thalach','exang','oldpeak','slope','ca','thal']
            X        = np.array([[features[f] for f in order]])
            X_scaled = scaler.transform(X)
            pred     = int(best_model.predict(X_scaled)[0])
            conf     = float(best_model.predict_proba(X_scaled)[0][pred]) \
                       if hasattr(best_model,'predict_proba') else 1.0
            rec = Prediction(user_id=current_user.id, patient_name=patient_name,
                             algorithm_used=BEST_MODEL_NAME,
                             prediction_result=pred, confidence_score=round(conf,4),
                             **features)
            db.session.add(rec); db.session.commit()
            return redirect(url_for('result', prediction_id=rec.id))
        except (ValueError, KeyError) as e:
            flash(f'Input error: {e}. Check all fields.','danger')
    return render_template('predict.html', best_model=BEST_MODEL_NAME)

# Result route to display prediction outcome
@app.route('/result/<int:prediction_id>')
@login_required
def result(prediction_id):
    p = Prediction.query.get_or_404(prediction_id)
    if p.user_id != current_user.id and not current_user.is_admin: abort(403)
    return render_template('result.html', prediction=p, features=prediction_to_dict(p))

# History route to display user's prediction records
@app.route('/history')
@login_required
def history():
    if current_user.is_admin: return redirect(url_for('admin_all_predictions'))
    page  = request.args.get('page',1,type=int)
    preds = Prediction.query.filter_by(user_id=current_user.id)\
                            .order_by(Prediction.created_at.desc())\
                            .paginate(page=page, per_page=10, error_out=False)
    return render_template('history.html', predictions=preds)

# Delete a prediction record from history (only by the owner)
@app.route('/history/delete/<int:pid>', methods=['POST'])
@login_required
def delete_prediction(pid):
    p = Prediction.query.get_or_404(pid)
    if p.user_id != current_user.id: abort(403)
    db.session.delete(p); db.session.commit()
    flash('Prediction record deleted.','info')
    return redirect(url_for('history'))


# ════════════════════════════════════════════════════════════════════════════
# ADMIN ROUTES
# ════════════════════════════════════════════════════════════════════════════
@app.route('/admin')
@login_required
@admin_required
def admin_dashboard():
    total_users = User.query.count()
    total_preds = Prediction.query.count()
    positive    = Prediction.query.filter_by(prediction_result=1).count()
    negative    = total_preds - positive
    recent      = Prediction.query.order_by(Prediction.created_at.desc()).limit(5).all()
    return render_template('admin/dashboard.html',
                           total_users=total_users, total_preds=total_preds,
                           positive=positive, negative=negative, recent=recent,
                           best_model=BEST_MODEL_NAME)

# Admin route to view all registered users
@app.route('/admin/users')
@login_required
@admin_required
def admin_users():
    users = User.query.order_by(User.created_at.desc()).all()
    for u in users:
        u.pred_count = Prediction.query.filter_by(user_id=u.id).count()
    return render_template('admin/users.html', users=users)

# Admin route to delete a user account (except self and other admins)
@app.route('/admin/users/delete/<int:uid>', methods=['POST'])
@login_required
@admin_required
def admin_delete_user(uid):
    u = User.query.get_or_404(uid)
    if u.id == current_user.id:
        flash('You cannot delete your own account.','danger')
        return redirect(url_for('admin_users'))
    if u.is_admin:
        flash('Administrator accounts cannot be deleted.','danger')
        return redirect(url_for('admin_users'))
    db.session.delete(u); db.session.commit()
    flash(f'User "{u.username}" deleted.','info')
    return redirect(url_for('admin_users'))

# Admin route to view all prediction records in the system
@app.route('/admin/predictions')
@login_required
@admin_required
def admin_all_predictions():
    page  = request.args.get('page',1,type=int)
    preds = Prediction.query.order_by(Prediction.created_at.desc())\
                            .paginate(page=page, per_page=15, error_out=False)
    return render_template('admin/predictions.html', predictions=preds)

# Admin route to view model perfromance metrics (hardcoded for demsonstration)
@app.route('/admin/performance')
@login_required
@admin_required
def admin_performance():
    results = [
        {'name':'Logistic Regression','accuracy':87.78,'precision':94.29,
         'recall':78.57,'f1':85.71,'specificity':95.83,'cv':81.19,'best':True},
        {'name':'Decision Tree',      'accuracy':66.67,'precision':65.00,
         'recall':61.90,'f1':63.41,'specificity':70.83,'cv':71.60,'best':False},
        {'name':'Random Forest',      'accuracy':83.33,'precision':88.57,
         'recall':73.81,'f1':80.52,'specificity':91.67,'cv':81.19,'best':False},
        {'name':'K-Nearest Neighbors','accuracy':86.67,'precision':94.12,
         'recall':76.19,'f1':84.21,'specificity':95.83,'cv':78.79,'best':False},
        {'name':'Naive Bayes',        'accuracy':86.67,'precision':94.12,
         'recall':76.19,'f1':84.21,'specificity':95.83,'cv':80.24,'best':False},
        {'name':'Support Vector Machine','accuracy':86.67,'precision':94.12,
         'recall':76.19,'f1':84.21,'specificity':95.83,'cv':80.21,'best':False},
    ]
    return render_template('admin/performance.html', results=results,
                           best_model=BEST_MODEL_NAME)

# Admin route to view dataset information (hardcoded for demonstration)
@app.route('/admin/dataset')
@login_required
@admin_required
def admin_dataset():
    dataset_info = {
        'name':       'UCI Cleveland Heart Disease Dataset',
        'source':     'UCI Machine Learning Repository',
        'records':    297,
        'features':   13,
        'no_disease': 160,
        'disease':    137,
        'split_train':207,
        'split_test': 90,
        'file':       'data/heart.csv',
        'exists':     os.path.exists('data/heart.csv'),
    }
    return render_template('admin/dataset.html', dataset=dataset_info)

# Error handler for 403 Forbiden access
@app.errorhandler(403)
def forbidden(e):
    return render_template('403.html'), 403

# Error handler for 404 Not Found
@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        print("[INFO] Database ready.")
    print(f"[INFO] Best model: {BEST_MODEL_NAME}")
    print("[INFO] Open: http://127.0.0.1:5000")
    app.run(debug=True, host='127.0.0.1', port=5000)
