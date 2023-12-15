from typing import Callable
import json as jsonlib
import requests
import os

REST = os.getenv("REST") or "34.171.207.21.nip.io:80"


def _request_response(
    endpoint: str,
    req_method: Callable[..., requests.Response],
    data: dict,
    files: dict = None,
    json: dict = None,
    debug: bool = False,
):
    response = req_method(
        f"http://{REST}/{endpoint}", data=data, files=files, json=json
    )

    if response.status_code == 200:
        if debug:
            print(response.text)
    else:
        print(
            f"response code: {response.status_code}, raw response: {response.text}"
        )

    return response
