import json
import requests


def upload(file,filename="model"):
    print(f"uploading {filename} to Drive")
    header = {"Authorization": "Bearer ya29.a0Ad52N3943f3-VqZBEiOwuUw4Q5c2_5lRFMNhZKpMRdNbvleGG4oZp1xIbGbPpWJe9JsRR_HKMap922KEEg6ziwLiOwsnmnT7ysN8SfpCwXmffJdE237hmI_j558fgvaTI68bbBAimaUcHkDbfCO4S27A6R51xG9RQ_4ZaCgYKAWQSARASFQHGX2Mi-LMjJWP0wIxNLPMyfblwWg0171"}

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



