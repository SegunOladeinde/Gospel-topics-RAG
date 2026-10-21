"""Back up data/raw/ to the private Supabase Storage bucket "raw-data".

Uploads every file under data/raw/ plus a manifest.json listing each file's
path, sha256 and size. On later runs only files whose sha256 differs from
the stored manifest are uploaded, and files deleted locally are removed from
the bucket too. Needs SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in .env
(the service key bypasses Storage RLS; it's never printed).

Usage:
    uv run python scripts/backup_raw.py            # back up what changed
    uv run python scripts/backup_raw.py --dry-run  # show what would change
"""

import argparse
import asyncio
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
BUCKET = "raw-data"
MANIFEST = "manifest.json"
CONCURRENCY = 8


def local_manifest() -> dict[str, dict]:
    files = {}
    for path in sorted(RAW_DIR.rglob("*")):
        if path.is_file() and path.name != ".gitkeep":
            data = path.read_bytes()
            files[path.relative_to(RAW_DIR).as_posix()] = {
                "sha256": hashlib.sha256(data).hexdigest(),
                "size": len(data),
            }
    return files


class Storage:
    def __init__(self, client: httpx.AsyncClient, base_url: str):
        self.client = client
        self.base = f"{base_url.rstrip('/')}/storage/v1"

    async def ensure_private_bucket(self) -> str:
        r = await self.client.get(f"{self.base}/bucket/{BUCKET}")
        if r.status_code == 200:
            if r.json().get("public"):
                raise SystemExit(f'Bucket "{BUCKET}" exists but is PUBLIC; make it private first.')
            return "exists"
        r = await self.client.post(f"{self.base}/bucket", json={"id": BUCKET, "name": BUCKET, "public": False})
        r.raise_for_status()
        return "created"

    async def download_manifest(self) -> dict:
        r = await self.client.get(f"{self.base}/object/{BUCKET}/{MANIFEST}")
        if r.status_code in (400, 404):
            return {}
        r.raise_for_status()
        return r.json().get("files", {})

    async def upload(self, key: str, data: bytes, content_type: str) -> None:
        for attempt in range(4):
            r = await self.client.post(
                f"{self.base}/object/{BUCKET}/{key}",
                content=data,
                headers={"Content-Type": content_type, "x-upsert": "true"},
            )
            if r.status_code < 500 and r.status_code != 429:
                r.raise_for_status()
                return
            await asyncio.sleep(2**attempt)
        r.raise_for_status()

    async def delete(self, keys: list[str]) -> None:
        for start in range(0, len(keys), 100):
            r = await self.client.request(
                "DELETE", f"{self.base}/object/{BUCKET}", json={"prefixes": keys[start : start + 100]}
            )
            r.raise_for_status()


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="report changes without uploading")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    url, key = os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        sys.exit("Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in .env")

    local = local_manifest()
    headers = {"apikey": key, "Authorization": f"Bearer {key}"}
    async with httpx.AsyncClient(headers=headers, timeout=60) as client:
        storage = Storage(client, url)
        bucket_state = "skipped (dry run)" if args.dry_run else await storage.ensure_private_bucket()
        remote = {} if bucket_state == "created" else await storage.download_manifest()

        changed = [p for p, meta in local.items() if remote.get(p, {}).get("sha256") != meta["sha256"]]
        removed = [p for p in remote if p not in local]
        total = sum(m["size"] for m in local.values())
        print(f'bucket "{BUCKET}": {bucket_state}')
        print(f"local files: {len(local)} ({total / 2**20:.1f} MB); changed or new: {len(changed)}; removed: {len(removed)}")
        if args.dry_run:
            return

        semaphore = asyncio.Semaphore(CONCURRENCY)
        done = 0

        async def put(path: str):
            nonlocal done
            async with semaphore:
                await storage.upload(path, (RAW_DIR / path).read_bytes(), "text/plain; charset=utf-8")
            done += 1
            if done % 500 == 0:
                print(f"  uploaded {done}/{len(changed)}", flush=True)

        await asyncio.gather(*(put(p) for p in changed))
        if removed:
            await storage.delete(removed)
        manifest = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "file_count": len(local),
            "total_bytes": total,
            "files": local,
        }
        await storage.upload(MANIFEST, json.dumps(manifest, indent=1).encode(), "application/json")
        print(f"uploaded {len(changed)} files, deleted {len(removed)}, wrote {MANIFEST}")


if __name__ == "__main__":
    asyncio.run(main())
