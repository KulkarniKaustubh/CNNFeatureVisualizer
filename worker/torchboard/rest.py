from typing import Callable
import json as jsonlib
import requests
import os

REST = os.getenv("REST") or "34.121.134.170.nip.io:80"


def _request_response(
    endpoint: str,
    req_method: Callable[..., requests.Response],
    data: dict,
    files: dict = None,
    json: dict = None,
    debug: bool = True,
) -> None:
    response = req_method(
        f"http://{REST}/{endpoint}", data=data, files=files, json=json
    )

    if response.status_code == 200:
        jsonResponse = jsonlib.dumps(response.json(), indent=4, sort_keys=True)
        if debug:
            print(jsonResponse)
    else:
        print(
            f"response code: {response.status_code}, raw response: {response.text}"
        )

    if debug:
        print(response.text)

    return
