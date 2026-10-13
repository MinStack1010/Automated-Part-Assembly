'''
Content-addressed SDF cache shared across planning jobs.

redmax computes the same SDF grid every time a physics job builds its
simulator: once through redmax.SDFMesh (auto collision threshold) and once
through BodySDFObj inside redmax.Simulation (contact model). For dense
meshes that dominates the job wall time. This module caches those grids in
the application storage directory, keyed by mesh file content and every
parameter that affects the grid, so repeat jobs load instead of recompute.

Two consumers are cached separately because they parse the same OBJ with
different vertex precision (trimesh double vs tinyobj float32); each entry
stores exactly what that consumer computed, so a hit is byte-identical to
a fresh build and planning behavior is unchanged.
'''
import hashlib
import json
import logging
import os
import shutil
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CACHE_VERSION = 1
CONSUMER_THRESHOLD = 'threshold'    # redmax.SDFMesh used for the collision threshold
CONSUMER_SIMULATION = 'simulation'  # BodySDFObj inside redmax.Simulation


def key_for(obj_path, consumer, sdf_dx, res=None):
    '''Return (cache_key, obj_sha256) for one OBJ file and SDF parameters.'''
    obj_sha = hashlib.sha256()
    with open(obj_path, 'rb') as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b''):
            obj_sha.update(chunk)
    digest = hashlib.sha256()
    digest.update(f'v{CACHE_VERSION}|{consumer}|dx={sdf_dx!r}|res={res!r}|'.encode())
    digest.update(obj_sha.digest())
    return digest.hexdigest(), obj_sha.hexdigest()


def _entry_paths(root, key):
    root = Path(root)
    return root / f'{key}.sdf', root / f'{key}.meta.json'


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _purge(root, key):
    sdf_path, meta_path = _entry_paths(root, key)
    for path in (sdf_path, meta_path):
        try:
            os.remove(path)
        except OSError:
            pass


def lookup(root, key):
    '''Return the cached .sdf path if the entry is complete and intact, else None.

    A missing, corrupt, truncated, or version-incompatible entry is deleted
    and reported as a miss so the caller recomputes; cache problems never
    fail the job.
    '''
    sdf_path, meta_path = _entry_paths(root, key)
    try:
        with open(meta_path) as fp:
            meta = json.load(fp)
        if meta.get('version') != CACHE_VERSION:
            _purge(root, key)
            return None
        expected_size = int(meta['sdf_bytes'])
        if os.path.getsize(sdf_path) != expected_size:
            _purge(root, key)
            return None
        if _sha256_file(sdf_path) != meta['sdf_sha256']:
            _purge(root, key)
            return None
        return sdf_path
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        _purge(root, key)
        return None


def begin_save(root, key):
    '''Return a staging path for a new entry; nothing is visible to readers yet.'''
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    return root / f'{key}.sdf.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}'


def finish_save(root, key, staging, *, consumer, obj_sha256, sdf_dx, res=None):
    '''Publish a fully written staging file atomically; only then is it readable.'''
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    sdf_path, meta_path = _entry_paths(root, key)
    meta = {
        'version': CACHE_VERSION,
        'consumer': consumer,
        'obj_sha256': obj_sha256,
        'sdf_dx': sdf_dx,
        'res': res,
        'sdf_bytes': os.path.getsize(staging),
        'sdf_sha256': _sha256_file(staging),
        'created_at': time.time(),
    }
    staging_meta = f'{meta_path}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}'
    try:
        with open(staging_meta, 'w') as fp:
            json.dump(meta, fp)
            fp.flush()
            os.fsync(fp.fileno())
        os.replace(staging, sdf_path)  # atomic on POSIX: readers see old or new, never partial
        os.replace(staging_meta, meta_path)
    finally:
        for leftover in (staging, staging_meta):
            if os.path.exists(leftover):
                try:
                    os.remove(leftover)
                except OSError:
                    pass


def publish_file(root, key, source_sdf, *, consumer, obj_sha256, sdf_dx, res=None):
    '''Copy an externally produced .sdf (e.g. BodySDFObj output) into the cache.

    Skips the copy when a valid entry already exists. Never raises: a cache
    write problem must not fail a job that already has its SDF.
    '''
    try:
        if lookup(root, key) is not None:
            return
        staging = begin_save(root, key)
        shutil.copyfile(source_sdf, staging)
        finish_save(root, key, staging, consumer=consumer, obj_sha256=obj_sha256, sdf_dx=sdf_dx, res=res)
    except Exception:
        logger.warning('SDF_CACHE_PUBLISH_FAILED key=%s', key[:16], exc_info=True)


def install_next_to_obj(root, key, obj_path):
    '''If a valid entry exists, copy it beside the OBJ where BodySDFObj loads it.

    Returns True on hit (file installed), False on miss. Any cache read error
    degrades to a miss.
    '''
    cached = lookup(root, key)
    if cached is None:
        return False
    target = os.path.splitext(obj_path)[0] + '.sdf'
    staging = f'{target}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}'
    try:
        shutil.copyfile(cached, staging)
        os.replace(staging, target)
        return True
    except OSError:
        logger.warning('SDF_CACHE_INSTALL_FAILED key=%s', key[:16], exc_info=True)
        try:
            os.remove(staging)
        except OSError:
            pass
        return False


@contextmanager
def entry_lock(root, key):
    '''Serialize concurrent misses for the same key so it is computed once.

    The lock is advisory (flock) and auto-releases if a worker dies; if the
    platform lacks fcntl the cache stays correct, only duplicate work remains
    possible.
    '''
    lock_path = Path(root) / f'{key}.lock'
    handle = None
    try:
        Path(root).mkdir(parents=True, exist_ok=True)
        handle = open(lock_path, 'w')
        try:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except ImportError:
            pass
        yield
    finally:
        if handle is not None:
            try:
                try:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except ImportError:
                    pass
            finally:
                handle.close()
