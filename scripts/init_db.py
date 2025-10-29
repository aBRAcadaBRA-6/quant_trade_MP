
from app.models.database import init_db, SessionLocal, User

def create_test_user():
    db = SessionLocal()
    
    # Create a test user
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
    db.close()

if __name__ == "__main__":
    print("Initializing database...")
    init_db()
    print("Database initialized!")
    
    print("Creating test user...")
    create_test_user()
    print("Done!")
