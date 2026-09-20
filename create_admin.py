# =============================================================================
# Create Admin Account Script
# Run this ONCE after train_models.py to create the administrator account.
# Usage:  python create_admin.py
# =============================================================================
from app import app, db, User

with app.app_context():
    db.create_all()
    admin_username = 'admin'
    admin_email    = 'admin@heartpredict.com'
    admin_password = 'Admin@1234'

    existing = User.query.filter_by(username=admin_username).first()
    if existing:
        print(f"[INFO] Admin account '{admin_username}' already exists.")
    else:
        admin = User(username=admin_username, email=admin_email, is_admin=True)
        admin.set_password(admin_password)
        db.session.add(admin)
        db.session.commit()
        print("=" * 50)
        print("  Admin account created successfully!")
        print("=" * 50)
        print(f"  Username : {admin_username}")
        print(f"  Password : {admin_password}")
        print(f"  Role     : Administrator")
        print("=" * 50)
        print("  You can change the password after first login.")
