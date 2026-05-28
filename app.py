from flask import Flask, render_template, request, redirect, url_for, send_file, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import enc_dec_functions
import os

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(app.instance_path, 'users.db')
os.makedirs(app.instance_path, exist_ok=True)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'yoursecretkey')

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# ─── User Model ───────────────────────────────────────────────────────────────

class User(db.Model, UserMixin):
    id          = db.Column(db.Integer, primary_key=True)
    username    = db.Column(db.String(150), unique=True, nullable=False)
    password    = db.Column(db.String(150), nullable=False)
    private_key = db.Column(db.Text, nullable=True)
    public_key  = db.Column(db.Text, nullable=True)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ─── Auth Routes ──────────────────────────────────────────────────────────────

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        existing = User.query.filter_by(username=request.form['username']).first()
        if existing:
            flash('Username already taken. Please choose a different one.')
            return render_template('register.html')
        hashed_pw = bcrypt.generate_password_hash(request.form['password']).decode('utf-8')
        private_pem, public_pem = enc_dec_functions.generate_rsa_keypair()
        new_user = User(
            username=request.form['username'],
            password=hashed_pw,
            private_key=private_pem,
            public_key=public_pem
        )
        db.session.add(new_user)
        db.session.commit()
        flash('Account created! Your RSA key pair has been generated. Sign in to continue.')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form['username']).first()
        if user is None:
            flash('No account found with that username. Please register first.')
        elif not bcrypt.check_password_hash(user.password, request.form['password']):
            flash('Incorrect password. Please try again.')
        else:
            login_user(user)
            return redirect(url_for('index'))
    return render_template('login.html')

@app.route('/index')
@login_required
def index():
    return render_template('index.html')

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/my-keys')
@login_required
def my_keys():
    return jsonify({
        "public_key": current_user.public_key,
        "has_private_key": current_user.private_key is not None
    })

# ─── Encrypt (with RECIPIENT's public key) ───────────────────────────────────

@app.route('/encrypt', methods=['POST'])
@login_required
def encrypt_route():
    file = request.files['file']
    recipient_username = request.form.get('recipient_username', '').strip()

    recipient = User.query.filter_by(username=recipient_username).first()
    if not recipient:
        return jsonify({"error": f"No user found with username '{recipient_username}'."}), 404

    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)

    # Encrypt with RECIPIENT's public key — only they can decrypt
    out_path = enc_dec_functions.encrypt_file(file_path, recipient.public_key)
    download_name = os.path.splitext(file.filename)[0] + ".bin"
    return send_file(out_path, as_attachment=True, download_name=download_name)

# ─── Decrypt (with MY own private key) ───────────────────────────────────────

@app.route('/decrypt', methods=['POST'])
@login_required
def decrypt_route():
    file = request.files['file']
    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)

    # Decrypt using MY OWN private key — only works if file was encrypted for me
    try:
        out_path = enc_dec_functions.decrypt_file(file_path, current_user.private_key)
        return send_file(out_path, as_attachment=True)
    except Exception:
        return jsonify({"error": "Decryption failed. This file was not encrypted for you."}), 400

# ─── Encrypt + Sign ──────────────────────────────────────────────────────────

@app.route('/encrypt-sign', methods=['POST'])
@login_required
def encrypt_sign_route():
    file = request.files['file']
    recipient_username = request.form.get('recipient_username', '').strip()

    recipient = User.query.filter_by(username=recipient_username).first()
    if not recipient:
        return jsonify({"error": f"No user found with username '{recipient_username}'."}), 404

    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)

    # Encrypt with RECIPIENT's public key, sign with MY private key
    out_path = enc_dec_functions.encrypt_and_sign_file(
        file_path,
        recipient.public_key,
        current_user.private_key
    )
    download_name = os.path.splitext(file.filename)[0] + ".bin"
    return send_file(out_path, as_attachment=True, download_name=download_name)

# ─── Decrypt + Verify ────────────────────────────────────────────────────────

@app.route('/decrypt-verify', methods=['POST'])
@login_required
def decrypt_verify_route():
    file = request.files['file']
    sender_username = request.form.get('sender_username', '').strip()

    sender = User.query.filter_by(username=sender_username).first()
    if not sender:
        return jsonify({"success": False, "error": f"No user found with username '{sender_username}'."}), 404

    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)

    try:
        # Decrypt with MY private key, verify with SENDER's public key
        out_path, sig_valid = enc_dec_functions.decrypt_and_verify_file(
            file_path,
            current_user.private_key,
            sender.public_key
        )
        return jsonify({
            "success": True,
            "sig_valid": sig_valid,
            "download_url": url_for('download_temp', filename=os.path.basename(out_path))
        })
    except Exception as e:
        return jsonify({"success": False, "error": "Decryption failed. This file was not encrypted for you."}), 400

@app.route('/download-temp/<filename>')
@login_required
def download_temp(filename):
    path = os.path.join("uploads", filename)
    return send_file(path, as_attachment=True)


with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=False, threaded=True, port=8080, host='127.0.0.1')
