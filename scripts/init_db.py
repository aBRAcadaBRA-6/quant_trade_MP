# scripts/init_db.py
from app.models.database import init_db, SessionLocal
from app.models.database import User
from sqlalchemy.exc import IntegrityError

def create_test_user():
    db = SessionLocal()
    try:
        # Check if test user exists
        existing = db.query(User).filter(User.email == "test@example.com").first()
        if existing:
            print(f"Test user already exists: {existing.name} (ID: {existing.id})")
            return

        user = User(
            name="Test User",
            email="test@example.com",
            risk_tolerance=0.5,
            capital=100000.0,
            max_assets=15,
            drawdown_limit=0.25
        )
        db.add(user)
        db.commit()
        print(f"Created user: {user.name} (ID: {user.id})")
    except IntegrityError as e:
        db.rollback()
        print("IntegrityError while creating test user:", e)
    finally:
        db.close()

if __name__ == "__main__":
    print("Initializing database...")
    init_db()
    print("Database initialized!")

    print("Creating test user...")
    create_test_user()
    print("Done!")
