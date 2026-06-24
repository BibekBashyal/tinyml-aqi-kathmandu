"""
GitHub Repository Setup Script
================================
Run this ONCE to create the repo and push all files via GitHub API.

Usage:
    1. Create a GitHub Personal Access Token:
       github.com → Settings → Developer settings → Personal access tokens → Tokens (classic)
       → Generate new token → check 'repo' scope → copy the token

    2. Fill in your details below (GITHUB_USERNAME and GITHUB_TOKEN)

    3. Run:  python setup_github.py
"""

import base64
import json
import os
import requests

# ── FILL THESE IN ─────────────────────────────────────────────────────────────
GITHUB_USERNAME = "YOUR_GITHUB_USERNAME"   # e.g. "bibek-bashyal"
GITHUB_TOKEN    = "YOUR_GITHUB_TOKEN"      # e.g. "ghp_xxxxxxxxxxxx"
REPO_NAME       = "tinyml-aqi-kathmandu"
REPO_DESC       = "TinyML-Based AQI Classification on ESP32: Cross-Location Generalization in the Kathmandu Valley"
PRIVATE         = False   # Set True if you want a private repo
# ──────────────────────────────────────────────────────────────────────────────

BASE_URL = "https://api.github.com"
HEADERS  = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28"
}

def create_repo():
    print(f"Creating repository: {REPO_NAME} ...")
    r = requests.post(f"{BASE_URL}/user/repos", headers=HEADERS, json={
        "name": REPO_NAME,
        "description": REPO_DESC,
        "private": PRIVATE,
        "auto_init": False
    })
    if r.status_code == 201:
        print(f"  ✓ Repo created: {r.json()['html_url']}")
        return True
    elif r.status_code == 422:
        print(f"  ℹ Repo already exists — pushing files to existing repo")
        return True
    else:
        print(f"  ✗ Failed: {r.status_code} — {r.text}")
        return False

def push_file(path, content_str):
    encoded = base64.b64encode(content_str.encode("utf-8")).decode("utf-8")
    url = f"{BASE_URL}/repos/{GITHUB_USERNAME}/{REPO_NAME}/contents/{path}"

    # Check if file already exists (to get SHA for update)
    existing = requests.get(url, headers=HEADERS)
    sha = existing.json().get("sha") if existing.status_code == 200 else None

    payload = {"message": f"Add {path}", "content": encoded}
    if sha:
        payload["sha"] = sha

    r = requests.put(url, headers=HEADERS, json=payload)
    if r.status_code in (200, 201):
        print(f"  ✓ {path}")
    else:
        print(f"  ✗ {path} — {r.status_code}: {r.text[:80]}")

def push_binary(path, content_bytes):
    encoded = base64.b64encode(content_bytes).decode("utf-8")
    url = f"{BASE_URL}/repos/{GITHUB_USERNAME}/{REPO_NAME}/contents/{path}"
    existing = requests.get(url, headers=HEADERS)
    sha = existing.json().get("sha") if existing.status_code == 200 else None
    payload = {"message": f"Add {path}", "content": encoded}
    if sha:
        payload["sha"] = sha
    r = requests.put(url, headers=HEADERS, json=payload)
    if r.status_code in (200, 201):
        print(f"  ✓ {path}")
    else:
        print(f"  ✗ {path} — {r.status_code}: {r.text[:80]}")

def collect_files(base_dir):
    files = {}
    for root, dirs, filenames in os.walk(base_dir):
        # Skip __pycache__ and hidden folders
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        for fname in filenames:
            if fname.startswith('.') and fname != '.gitignore':
                continue
            full_path = os.path.join(root, fname)
            rel_path  = os.path.relpath(full_path, base_dir).replace("\\", "/")
            files[rel_path] = full_path
    return files

if __name__ == "__main__":
    if GITHUB_USERNAME == "YOUR_GITHUB_USERNAME" or GITHUB_TOKEN == "YOUR_GITHUB_TOKEN":
        print("ERROR: Please fill in GITHUB_USERNAME and GITHUB_TOKEN before running.")
        exit(1)

    if not create_repo():
        exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    files = collect_files(script_dir)

    # Remove this script from the push list
    files.pop("setup_github.py", None)

    print(f"\nPushing {len(files)} files to GitHub...")
    for rel_path, full_path in sorted(files.items()):
        try:
            with open(full_path, "rb") as f:
                raw = f.read()
            try:
                push_file(rel_path, raw.decode("utf-8"))
            except UnicodeDecodeError:
                push_binary(rel_path, raw)
        except Exception as e:
            print(f"  ✗ {rel_path} — {e}")

    print(f"\nDone! View your repo at:")
    print(f"  https://github.com/{GITHUB_USERNAME}/{REPO_NAME}")
