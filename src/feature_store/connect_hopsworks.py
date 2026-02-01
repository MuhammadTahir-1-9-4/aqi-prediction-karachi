import os
from dotenv import load_dotenv
import hopsworks
import hsfs

load_dotenv()



_project = None

def get_feature_store():
    global _project
    if _project is None:
        _project = hopsworks.login(
            api_key_value=os.getenv("HOPSWORKS_API_KEY")
        )
    return _project.get_feature_store()