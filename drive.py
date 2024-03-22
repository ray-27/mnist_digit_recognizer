import json
import requests


def upload(file,filename="model"):
    print(f"uploading {filename} to Drive")
    header = {"Authorization": "Bearer ya29.a0Ad52N3-rPYFJtA8H3cKnd49CH_A6_7jQPn7WwwUdESoVr5bIXzPRw8mbZCRtzyWCQH_OZf-EXDeNSrT51Wz82BSRWXz6JgY5eQ3hxbmW85r9Jc0UeYIpmwCeCeBL0EwQehfMSEhv65621VIkEYXSFa2JoPn7GSg7oiYlaCgYKAc8SARASFQHGX2MirZLMGlLWarPURyi5FsQY_g0171"}

    para = {
        "name": filename
    }
    files = {
        'data': ('metadata',json.dumps(para), 'application/json; charset=utf-8'),
        'file': open(file, "rb")

    }
    r = requests.post(
        "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart",
        headers=header,
        files=files
    )
    print("Upload Complete")
    pass
