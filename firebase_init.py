# firebase_init.py
import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase only if it hasn't been initialized already
if not firebase_admin._apps:
    service_account = "crop2x.json"
    cred = credentials.Certificate(service_account)
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://cropx-6f03a-default-rtdb.firebaseio.com',
        'storageBucket': 'cropx-6f03a.appspot.com'
    })

# Create and export Firestore client
firestore_client = firestore.client()

def get_firestore_client():
    """Returns the Firestore client instance."""
    return firestore_client
