from pymongo import MongoClient
from config import MONGODB_URI

DATABASE_NAME = "pmmnm"
COLLECTION_NAME = "dataset"

client = MongoClient(MONGODB_URI)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

def get_collection():
    return collection