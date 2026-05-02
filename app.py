from flask import Flask, render_template, request, redirect, url_for, send_file, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import enc_dec_functions
import os

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SECRET_KEY'] = 'yoursecretkey'

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# ─── User Model ───────────────────────────────────────────────────────────────

class User(db.Model, UserMixin):
    id          = db.Column(db.Integer, primary_key=True)
    username    = db.Column(db.String(150), unique=True, nullable=False)
    password    = db.Column(db.String(150), nullable=False)
    private_key = db.Column(db.Text, nullable=True)   # RSA private key PEM
    public_key  = db.Column(db.Text, nullable=True)   # RSA public key PEM

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

        # Generate RSA key pair at registration time
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

# ─── Key Info Endpoint ────────────────────────────────────────────────────────

@app.route('/my-keys')
@login_required
def my_keys():
    """Return current user's public key (safe to expose) for display in UI."""
    return jsonify({
        "public_key": current_user.public_key,
        "has_private_key": current_user.private_key is not None
    })

# ─── Plain Encrypt / Decrypt (uses public key as symmetric password) ──────────

@app.route('/encrypt', methods=['POST'])
@login_required
def encrypt_route():
    file = request.files['file']
    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)
    # Use the user's RSA public key as the password for PBKDF2 derivation
    out_path = enc_dec_functions.encrypt_file(file_path, current_user.public_key)
    return send_file(out_path, as_attachment=True)

@app.route('/decrypt', methods=['POST'])
@login_required
def decrypt_route():
    file = request.files['file']
    sender_username = request.form.get('sender_username', '').strip()

    sender = User.query.filter_by(username=sender_username).first()
    if not sender:
        return jsonify({"error": f"No user found with username '{sender_username}'."}), 404

    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)

    # Decrypt using the SENDER's public key (they encrypted it with their own key)
    out_path = enc_dec_functions.decrypt_file(file_path, sender.public_key)
    return send_file(out_path, as_attachment=True)

# ─── Sign / Verify (HMAC-SHA256, user supplies their private key string) ──────

@app.route('/sign', methods=['POST'])
@login_required
def sign_file_route():
    file = request.files['file']
    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)
    # Use the stored RSA private key PEM as the HMAC key
    out_path = enc_dec_functions.sign_file(file_path, current_user.private_key)
    return send_file(out_path, as_attachment=True)

@app.route('/verify', methods=['POST'])
@login_required
def verify_file_route():
    file = request.files['file']
    sig_file = request.files['signature_file']
    sender_username = request.form.get('sender_username', '').strip()

    sender = User.query.filter_by(username=sender_username).first()
    if not sender:
        return jsonify({"valid": False, "error": f"No user found with username '{sender_username}'."}), 404

    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", file.filename)
    sig_path  = os.path.join("uploads", sig_file.filename)
    file.save(file_path)
    sig_file.save(sig_path)

    with open(file_path, "rb") as f:
        data = f.read()
    with open(sig_path, "r") as f:
        signature = f.read().strip()

    # Verify using the SENDER's private key (HMAC symmetric — same key signs and verifies)
    valid = enc_dec_functions.verify_signature(data, signature, sender.private_key)
    return jsonify({"valid": valid})

# ─── Encrypt + Sign (combined, RSA-PSS) ──────────────────────────────────────

@app.route('/encrypt-sign', methods=['POST'])
@login_required
def encrypt_sign_route():
    file = request.files['file']
    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)
    out_path = enc_dec_functions.encrypt_and_sign_file(
        file_path,
        current_user.public_key,
        current_user.private_key
    )
    return send_file(out_path, as_attachment=True)

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
        out_path, sig_valid = enc_dec_functions.decrypt_and_verify_file(
            file_path,
            sender.public_key   # SENDER's public key used for both decrypt and RSA-PSS verify
        )
        return jsonify({
            "success": True,
            "sig_valid": sig_valid,
            "download_url": url_for('download_temp', filename=os.path.basename(out_path))
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route('/download-temp/<filename>')
@login_required
def download_temp(filename):
    path = os.path.join("uploads", filename)
    return send_file(path, as_attachment=True)


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
