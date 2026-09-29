"""Small Google Drive mirror used for resumable Kaggle training.

Credentials are read from environment variables populated by Kaggle Secrets;
no token is stored in the repository or checkpoint files.
"""
from __future__ import annotations
import json, mimetypes, os
from pathlib import Path
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
import io

FOLDER_MIME = "application/vnd.google-apps.folder"

def _service():
    refresh = os.environ.get("GOOGLE_REFRESH_TOKEN", "")
    client_id = os.environ.get("GOOGLE_CLIENT_ID", "")
    client_secret = os.environ.get("GOOGLE_CLIENT_SECRET", "")
    if refresh.startswith("{"):
        bundle = json.loads(refresh)
        refresh = bundle.get("refresh_token", refresh)
        client_id = client_id or bundle.get("client_id", "")
        client_secret = client_secret or bundle.get("client_secret", "")
    if not refresh or not client_id or not client_secret:
        raise RuntimeError("Drive sync requires GOOGLE_REFRESH_TOKEN plus GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET (or a JSON refresh-token bundle).")
    # Do not send a new scope in the refresh request. Google refresh tokens
    # are bound to the scopes granted during the original consent flow;
    # requesting a different scope causes invalid_scope. The Drive API client
    # will use exactly the scopes already attached to this refresh token.
    credentials = Credentials(None, refresh_token=refresh, token_uri="https://oauth2.googleapis.com/token", client_id=client_id, client_secret=client_secret)
    return build("drive", "v3", credentials=credentials, cache_discovery=False)

def _children(service, parent_id):
    result = {}; token = None
    while True:
        response = service.files().list(q=f"'{parent_id}' in parents and trashed=false", spaces="drive", fields="nextPageToken,files(id,name,mimeType)", pageToken=token).execute()
        result.update({item["name"]: item for item in response.get("files", [])})
        token = response.get("nextPageToken")
        if not token: return result

def _folder(service, parent_id, name, cache):
    item = cache.get(name)
    if item and item["mimeType"] == FOLDER_MIME: return item["id"]
    created = service.files().create(body={"name": name, "mimeType": FOLDER_MIME, "parents": [parent_id]}, fields="id").execute()
    cache[name] = {"id": created["id"], "name": name, "mimeType": FOLDER_MIME}
    return created["id"]

def _remote_root(service, root_id, parts):
    parent = root_id
    for name in [part for part in Path(parts).parts if part not in (".", "")]:
        parent = _folder(service, parent, name, _children(service, parent))
    return parent

def upload_tree(local_root, root_id, remote_path):
    service = _service(); parent = _remote_root(service, root_id, remote_path); local_root = Path(local_root)
    for path in sorted(local_root.rglob("*")):
        if not path.is_file(): continue
        relative_parent = path.parent.relative_to(local_root)
        remote_parent = _remote_root(service, parent, str(relative_parent)) if str(relative_parent) != "." else parent
        cache = _children(service, remote_parent); existing = cache.get(path.name)
        media = MediaFileUpload(str(path), mimetype=mimetypes.guess_type(path.name)[0] or "application/octet-stream", resumable=False)
        if existing and existing["mimeType"] != FOLDER_MIME:
            service.files().update(fileId=existing["id"], media_body=media).execute()
        else:
            service.files().create(body={"name": path.name, "parents": [remote_parent]}, media_body=media, fields="id").execute()

def download_tree(local_root, root_id, remote_path):
    service = _service(); parent = _remote_root(service, root_id, remote_path); local_root = Path(local_root); local_root.mkdir(parents=True, exist_ok=True)
    def walk(remote_id, local_dir):
        local_dir.mkdir(parents=True, exist_ok=True)
        for item in _children(service, remote_id).values():
            destination = local_dir / item["name"]
            if item["mimeType"] == FOLDER_MIME:
                walk(item["id"], destination); continue
            request = service.files().get_media(fileId=item["id"]); buffer = io.BytesIO(); downloader = MediaIoBaseDownload(buffer, request); done = False
            while not done: _, done = downloader.next_chunk()
            destination.write_bytes(buffer.getvalue())
    walk(parent, local_root)
