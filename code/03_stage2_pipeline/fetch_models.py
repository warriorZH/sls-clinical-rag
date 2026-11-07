#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""找到（必要时下载）模型，逐个文件核对后登记到 runs/models/registry.json。

    python fetch_models.py --key g1      --job fetch_g1       Qwen3-4B-Instruct-2507（问答模型 G1，也用于 LLM 零样本判断）
    python fetch_models.py --key qr4b    --job fetch_qr4b     Qwen3-Reranker-4B（可选基线）
    python fetch_models.py --key judge_a --job fetch_judge_a  评审 A：GLM-4-9B-0414
    python fetch_models.py --key judge_b --job fetch_judge_b  评审 B：Hunyuan-7B-Instruct
    python fetch_models.py --key g2      --job fetch_g2       G2：GLM-4-9B-0414（与评审 A 是同一个模型）

按以下顺序找模型，用第一份核对通过的：
  1. 上次登记、已经用过的那一份（保证同一模型在整个实验中用的是同一份文件）；
  2. 服务器上已有的副本：先找 /home/<user>/models（配置 paths.models_dir），再找 ~/models、ModelScope 与 Hugging Face 的缓存；
  3. 用 ModelScope 下载到 /home/<user>/models/<模型名>；
  4. 经 hf-mirror 下载（关闭 Xet）；还不行就按文件清单逐个直接下载。

单卡备选模型只在登记方案规定的条件下使用：服务器上没有 accelerate，或不足两张 GPU（方案第十四、十五节）。
找不到或下载失败不是改用备选模型的理由：这时作业标为“需要处理”。

核对：每个权重文件都完整读一遍（读不出说明磁盘有问题），safetensors 还核对文件长度是否覆盖文件头声明的全部张量。
Hugging Face 缓存里的副本还要与镜像站 API 给出的大小和 SHA256 一致；上次登记的副本要与上次记录的 SHA256 一致；
其他来源（如 ModelScope）的副本与 Hugging Face 的 SHA256 只作对照，结果记入登记表。
"""
import os

# hf-mirror 不能转发 Xet 存储的认证（10 月 28 日下载 GLM-4-9B-0414 时 cas-server.xethub.hf.co 返回 401）：
# 关闭 Xet，改用普通的 resolve 链接下载。必须在导入 huggingface_hub 之前设置。
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')

import argparse  # noqa: E402
import fnmatch  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import struct  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import urllib.parse  # noqa: E402
import urllib.request  # noqa: E402
from pathlib import Path  # noqa: E402

from common import (RUNS, WEIGHT_SUFFIXES, Log, fetch_model, load_config, mark_done, model_complete,  # noqa: E402
                    model_registry, now, read_json, register_model, sha256_file, write_json)

UA = {'User-Agent': 'eval_v2-fetch/1.2'}
ALLOW = ['*.json', '*.safetensors', '*.model', '*.txt', '*.tiktoken', '*.jinja', 'tokenizer*', 'vocab*', 'merges*', '*.py']
DIRECT_ROOT = Path(os.path.expanduser('~/.cache/eval_v2_models'))
MODELS_DIR = Path('/home/<user>/models')           # 由配置 paths.models_dir 覆盖
SKIP_NAMES = ('optimizer', 'scheduler', 'rng_state', 'training_args')


def endpoint():
    return (os.environ.get('HF_ENDPOINT') or 'https://huggingface.co').rstrip('/')


def fallback_allowed(env):
    """登记方案规定的单卡退路条件：没有 accelerate，或不足两张 GPU。"""
    return str(env.get('accelerate', 'missing')).startswith('missing') or int(env.get('n_gpus') or 0) < 2


def hf_repo_of(repos):
    """候选名里第一个 Hugging Face 仓库名（org/name），用于向镜像站查询官方文件清单。"""
    for r in repos:
        if '/' in r and not r.startswith(('/', '~', '.')):
            return r
    return None


def short_names(repos):
    out = set()
    for r in repos:
        n = str(r).rstrip('/').split('/')[-1].lower()
        out |= {n, n.replace('.', '___')}            # ModelScope 的本地目录名把“.”写成“___”
    return out


# ----------------------------------------------------------------------------- 镜像站的文件清单

def api_json(path, log, timeout=60):
    url = endpoint() + path
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
            return json.loads(r.read().decode('utf-8'))
    except Exception as e:  # noqa: BLE001
        log('  读取文件清单失败：{}（{}: {}）'.format(url, type(e).__name__, str(e)[:200]))
        return None


def hub_files(repo, rev, log):
    """镜像站 API 给出的文件清单：(版本号, {文件名: (大小, SHA256 或 None)})。取不到时清单为 None。"""
    info = api_json('/api/models/{}/revision/{}?blobs=true'.format(repo, rev or 'main'), log)
    if isinstance(info, dict) and info.get('siblings'):
        out = {}
        for s in info['siblings']:
            lfs = s.get('lfs') or {}
            size = lfs.get('size') or s.get('size')
            out[s['rfilename']] = (int(size) if size else None, lfs.get('sha256') or lfs.get('oid'))
        if any(v[0] for v in out.values()):
            return info.get('sha') or rev, out
    sha = (info or {}).get('sha') or rev or 'main'
    tree = api_json('/api/models/{}/tree/{}?recursive=true'.format(repo, sha), log)
    if not isinstance(tree, list):
        return (info or {}).get('sha') or rev, None
    out = {}
    for it in tree:
        if it.get('type') == 'file':
            lfs = it.get('lfs') or {}
            size = lfs.get('size') or it.get('size')
            out[it['path']] = (int(size) if size else None, lfs.get('sha256') or lfs.get('oid'))
    return sha, out


# ----------------------------------------------------------------------------- 服务器上已有的副本

def walk_dirs(root, max_depth):
    """root 之下 max_depth 层以内的目录（跳过隐藏目录，不跟随目录链接），按名称排序。"""
    root = Path(root)
    stack = [(root, 0)]
    while stack:
        d, depth = stack.pop(0)
        yield d
        if depth >= max_depth:
            continue
        try:
            subs = sorted(x for x in d.iterdir() if x.is_dir() and not x.is_symlink() and not x.name.startswith('.'))
        except OSError:
            continue
        stack.extend((x, depth + 1) for x in subs)


def model_dirs(root, max_depth=5):
    for d in walk_dirs(root, max_depth):
        if (d / 'config.json').is_file():
            yield d


def describe_models_dir(root, log):
    """记录 /home/<user>/models 里有哪些模型目录，便于核对。"""
    found = []
    for d in model_dirs(root, 4):
        try:
            arch = ','.join(json.loads((d / 'config.json').read_text(encoding='utf-8')).get('architectures') or ['?'])
        except Exception:  # noqa: BLE001
            arch = '?'
        found.append('{}（{}{}）'.format(d.relative_to(root), arch, '' if model_complete(d) else '，权重不全'))
    log('{} 中的模型目录：{}'.format(root, '；'.join(found) or '（无）'))


def find_local(repos, roots):
    """在各目录中按目录名（或 config.json 的 _name_or_path）找与候选名相同、权重齐全的模型。"""
    names = short_names(repos)
    out = []
    for r in repos:
        p = Path(os.path.expanduser(r))
        if p.is_absolute() and p.is_dir() and model_complete(p):
            out.append(p)
    for root in roots:
        root = Path(os.path.expanduser(str(root)))
        if not root.is_dir():
            continue
        for d in model_dirs(root):
            cand = {d.name.lower()}
            m = re.match(r'models--([^/]+?)--(.+)$', d.parent.parent.name) if d.parent.name == 'snapshots' else None
            if m:
                cand.add(m.group(2).lower())
            try:
                nop = json.loads((d / 'config.json').read_text(encoding='utf-8')).get('_name_or_path') or ''
                if nop:
                    cand.add(str(nop).rstrip('/').split('/')[-1].lower())
            except Exception:  # noqa: BLE001
                pass
            if cand & names and model_complete(d):
                out.append(d)
    return out


def source_repo(d, fallback):
    """副本对应的仓库名：Hugging Face 缓存目录名、直接下载时写的来源记录，或 ModelScope 下载时的记录。"""
    cr = cache_repo(d)
    if cr:
        return cr
    for f in ('.eval_v2_source.json',):
        info = read_json(Path(d) / f, {}) or {}
        if info.get('repo'):
            return info['repo']
    return fallback


# ----------------------------------------------------------------------------- 下载

def modelscope_download(ms_id, dest, log):
    """用 ModelScope 下载到 dest（/home/<user>/models/<模型名>）。服务器上没有 modelscope 或下载失败时返回 None。"""
    dest = Path(dest)
    try:
        dest.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        log('  无法创建 {}：{}'.format(dest, e))
        return None
    log('用 ModelScope 下载 {} → {}'.format(ms_id, dest))
    done = False
    try:
        try:
            from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot
        except ImportError:
            from modelscope import snapshot_download as ms_snapshot
        ms_snapshot(ms_id, local_dir=str(dest))
        done = True
    except ImportError:
        log('  当前 Python 环境里没有 modelscope 包，改用 modelscope 命令')
    except TypeError as e:
        log('  modelscope 版本较旧（{}），改用 modelscope 命令'.format(e))
    except Exception as e:  # noqa: BLE001
        log('  ModelScope 下载失败：{}: {}'.format(type(e).__name__, str(e)[:300]))
    if not done:
        exe = shutil.which('modelscope') or next((str(p) for p in (Path(os.path.expanduser(x)) for x in (
            '~/.local/bin/modelscope', '~/anaconda3/bin/modelscope', '~/miniconda3/bin/modelscope'))
            if p.exists()), None)
        if not exe:
            log('  服务器上找不到 modelscope 命令')
            return None
        r = subprocess.run([exe, 'download', '--model', ms_id, '--local_dir', str(dest)], capture_output=True,
                           text=True)
        tail = [x for x in (r.stdout + r.stderr).splitlines() if x.strip()][-5:]
        log('  modelscope download 退出码 {}：{}'.format(r.returncode, ' | '.join(t[:200] for t in tail)))
        if r.returncode != 0:
            return None
    if not model_complete(dest):
        log('  ModelScope 下载后权重文件不全：{}'.format(dest))
        return None
    write_json(dest / '.eval_v2_source.json', {'repo': ms_id, 'source': 'modelscope', 'time': now()})
    return dest


def http_download(repo, rev, name, dest, exp, log, tries=6):
    """按 resolve 链接直接下载一个文件（镜像站会重定向到带签名的下载地址）；支持断点续传，下载后核对大小与 SHA256。"""
    url = '{}/{}/resolve/{}/{}'.format(endpoint(), repo, rev or 'main', urllib.parse.quote(name))
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + '.part')
    size_exp, sha_exp = exp if exp else (None, None)
    last = None
    for attempt in range(1, tries + 1):
        try:
            have = part.stat().st_size if part.exists() else 0
            hdr = dict(UA)
            if have:
                hdr['Range'] = 'bytes={}-'.format(have)
            log('  直接下载 {}（第 {} 次{}）'.format(name, attempt, '，续传自 {:.2f} GB'.format(have / 1e9) if have else ''))
            with urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=120) as r:
                if have and getattr(r, 'status', 200) != 206:
                    have = 0                                  # 服务器不支持续传：从头下载
                with open(str(part), 'ab' if have else 'wb') as out:
                    while True:
                        b = r.read(1 << 23)
                        if not b:
                            break
                        out.write(b)
            got = part.stat().st_size
            if size_exp and got < size_exp:
                raise IOError('只收到 {} / {} 字节'.format(got, size_exp))
            if (size_exp and got != size_exp) or (sha_exp and sha256_file(part) != sha_exp):
                part.unlink()
                raise IOError('下载的文件与清单不符，已删除')
            if dest.is_symlink() or dest.exists():
                dest.unlink()
            os.replace(str(part), str(dest))
            return
        except Exception as e:  # noqa: BLE001
            last = e
            log('  失败：{}: {}'.format(type(e).__name__, str(e)[:200]))
            time.sleep(min(60, 10 * attempt))
    raise RuntimeError('无法下载 {}：{}'.format(name, last))


def direct_snapshot(repo, log):
    """huggingface_hub 下载不成功时，按文件清单逐个直接下载到 ~/.cache/eval_v2_models/<模型名>/。"""
    rev, files = hub_files(repo, 'main', log)
    if not files:
        raise RuntimeError('取不到 {} 的文件清单'.format(repo))
    dest = DIRECT_ROOT / repo.split('/')[-1]
    want = sorted(n for n in files if '/' not in n and any(fnmatch.fnmatch(n, p) for p in ALLOW))
    log('  直接下载 {}@{}：{} 个文件 → {}'.format(repo, rev, len(want), dest))
    for n in want:
        f = dest / n
        if f.exists() and check_file(f, files[n])[0] is None:
            continue
        http_download(repo, rev, n, f, files[n], log)
    write_json(dest / '.eval_v2_source.json', {'repo': repo, 'revision': rev, 'source': 'hf-mirror', 'time': now()})
    if not model_complete(dest):
        raise RuntimeError('下载后权重文件不完整')
    return dest, repo, rev


def clear_partial(repos, log):
    """删掉以前没有下载完的临时文件（*.incomplete）：旧版用 Xet 下载失败时留下的部分文件不能用来续传。"""
    try:
        from huggingface_hub import constants
        hub = Path(constants.HF_HUB_CACHE)
    except Exception:  # noqa: BLE001
        hub = Path(os.path.expanduser('~/.cache/huggingface/hub'))
    for r in repos:
        if '/' not in r or r.startswith(('/', '~', '.')):
            continue
        for f in (hub / 'models--{}'.format(r.replace('/', '--')) / 'blobs').glob('*.incomplete'):
            try:
                f.unlink()
                log('  删除未下载完的临时文件 {}'.format(f.name))
            except OSError:
                pass


def hf_cache_root():
    try:
        from huggingface_hub import constants
        return Path(constants.HF_HUB_CACHE)
    except Exception:  # noqa: BLE001
        return Path(os.path.expanduser('~/.cache/huggingface/hub'))


def candidates(repos, ms_id, prev, log, bad_cache):
    """按顺序给出候选副本 (来源, 目录, 仓库名, 版本号)；需要下载的候选只在前面的都不能用时才下载。"""
    names = short_names(repos)
    if prev.get('path') and str(prev.get('repo') or '').rstrip('/').split('/')[-1].lower() in names \
            and Path(prev['path']).is_dir():
        yield 'previous', Path(prev['path']), prev.get('repo'), prev.get('revision')
    roots = [MODELS_DIR, '~/models', '~/.cache/modelscope/hub', hf_cache_root(), DIRECT_ROOT]
    for d in find_local(repos, roots):
        yield 'local', d, source_repo(d, hf_repo_of(repos) or repos[0]), d.name if cache_repo(d) else None
    if ms_id:
        d = modelscope_download(ms_id, MODELS_DIR / ms_id.split('/')[-1], log)
        if d is not None:
            yield 'modelscope', d, ms_id, None
    hf = [r for r in repos if '/' in r and not r.startswith(('/', '~', '.'))]
    if not hf:
        return
    for p, names_bad in bad_cache.items():           # 缓存里核对不通过的文件先删掉，huggingface_hub 才会重新下载
        for n in names_bad:
            remove_cached(Path(p) / n)
    clear_partial(hf, log)
    try:
        p, r, rev = fetch_model(hf, log, search_dirs=[str(RUNS / '_no_local_search')])
        yield 'hf-mirror', Path(p), cache_repo(p) or r, rev
    except Exception as e:  # noqa: BLE001
        log('经 hf-mirror 下载失败：{}'.format(str(e)[:300]))
    for r in hf:
        try:
            p, r2, rev = direct_snapshot(r, log)
            yield 'direct', Path(p), r2, rev
        except Exception as e:  # noqa: BLE001
            log('  直接下载 {} 失败：{}'.format(r, str(e)[:300]))


# ----------------------------------------------------------------------------- 核对

def weight_names(d):
    return sorted(x.name for x in Path(d).iterdir() if x.is_file() and x.suffix in WEIGHT_SUFFIXES
                  and not any(s in x.name for s in SKIP_NAMES))


def check_files(d, hub):
    """要核对的文件：全部权重文件，加上清单里有 SHA256 的其他文件（如 tokenizer.json）。"""
    d = Path(d)
    names = set(weight_names(d))
    for n, (_, sha) in (hub or {}).items():
        if sha and '/' not in n and (d / n).is_file():
            names.add(n)
    return sorted(names)


def safetensors_span(f):
    """safetensors 文件头声明的总长度：文件至少要有这么长，否则读到末尾的张量时进程会因 SIGBUS 退出。"""
    with open(str(f), 'rb') as fh:
        n = struct.unpack('<Q', fh.read(8))[0]
        if n <= 0 or n > (100 << 20):
            raise ValueError('文件头长度异常：{}'.format(n))
        hdr = json.loads(fh.read(n).decode('utf-8'))
    ends = [v['data_offsets'][1] for k, v in hdr.items() if k != '__metadata__']
    return 8 + n + (max(ends) if ends else 0)


def check_file(f, exp):
    """返回 (问题, SHA256)；没有问题时问题为 None。exp = (大小, SHA256)，两项都可以是 None。"""
    f = Path(f)
    try:
        size = f.stat().st_size
        if exp and exp[0] and size != exp[0]:
            return '大小 {} 字节，应为 {}'.format(size, exp[0]), None
        if f.suffix == '.safetensors':
            need = safetensors_span(f)
            if size < need:
                return '文件不完整：{} 字节，文件头需要 {}'.format(size, need), None
        h = sha256_file(f)
    except (OSError, ValueError, struct.error) as e:
        return '读取出错：{}: {}'.format(type(e).__name__, str(e)[:200]), None
    if exp and exp[1] and h != exp[1]:
        return 'SHA256 不符', h
    return None, h


def cache_repo(path):
    """Hugging Face 缓存快照目录 models--<org>--<name>/snapshots/<commit> 对应的仓库名；不是缓存目录时返回 None。"""
    p = Path(path)
    m = re.match(r'models--([^/]+?)--(.+)$', p.parent.parent.name) if p.parent.name == 'snapshots' else None
    return '{}/{}'.format(m.group(1), m.group(2)) if m else None


def remove_cached(f):
    """删掉缓存快照里的文件（符号链接）和它指向的 blob。"""
    f = Path(f)
    target = None
    try:
        target = f.resolve(strict=True)
    except Exception:  # noqa: BLE001
        pass
    for p in (f, target):
        try:
            if p is not None and (p.is_symlink() or p.exists()):
                p.unlink()
        except FileNotFoundError:
            pass


def file_stamp(f):
    """文件的大小、修改时间（纳秒）与 inode：三者都没变，才认为文件自上次核对后没有被改动或替换。"""
    st = Path(f).stat()
    return {'size': st.st_size, 'mtime_ns': st.st_mtime_ns, 'ino': st.st_ino}


def reuse_verification(path, key):
    """同一目录已由另一个作业核对过、文件没有变化（例如评审 A 与 G2 共用 GLM-4-9B-0414）：直接沿用。"""
    for k, info in model_registry().items():
        v = info.get('verified') or {}
        if k == key or info.get('path') != str(path) or not v.get('files'):
            continue
        try:
            same = all(file_stamp(Path(path) / n) == s for n, s in v['files'].items())
        except OSError:
            same = False
        if same:
            return k, info
    return None, None


def verify(path, repo, rev, key, origin, prev, hf_repo, log):
    """核对一份副本。返回 (问题, 权重 SHA256, 核对记录, 版本号)；问题为空表示可以使用。"""
    path = Path(path)
    k, other = reuse_verification(path, key)
    if other:
        log('核对：与 {} 是同一目录，已核对过且文件未变，沿用其结果'.format(k))
        return {}, other.get('weights_sha256') or {}, dict(other['verified'], reused_from=k, origin=origin), \
            other.get('revision') or rev
    crepo = cache_repo(path)
    strict, basis = None, 'none'
    if crepo or origin in ('hf-mirror', 'direct'):
        rev_hub, hub = hub_files(crepo or repo, rev or path.name, log)
        if hub:
            strict, basis = hub, 'hub'
    if strict is None and prev.get('path') == str(path) and prev.get('weights_sha256'):
        strict, basis = {n: (None, h) for n, h in prev['weights_sha256'].items()}, 'previous_registry'
    ref = None
    if basis != 'hub' and hf_repo:
        _, ref = hub_files(hf_repo, 'main', log)               # 只作对照：与 Hugging Face 上的官方文件是否相同
    names = check_files(path, strict if basis == 'hub' else None)
    log('核对 {}（{}）的 {} 个文件；依据：{} …'.format(path, origin, len(names), {
        'hub': '镜像站给出的大小与 SHA256', 'previous_registry': '上次登记的 SHA256',
        'none': '能否完整读出与 safetensors 文件头'}[basis]))
    problems, hashes, files = {}, {}, {}
    for n in names:
        prob, h = check_file(path / n, (strict or {}).get(n))
        if prob:
            problems[n] = prob
            continue
        hashes[n] = h
        files[n] = file_stamp(path / n)
    if not problems and not model_complete(path):
        problems['model.safetensors.index.json'] = '索引中列出的权重文件不全'
    weights = {n: hashes[n] for n in weight_names(path) if n in hashes}
    match = None
    if ref is not None and not problems:
        match = 'identical' if weights and all(n in ref and ref[n][1] == h for n, h in weights.items()) else 'different'
        log('  与 Hugging Face 上 {} 的官方文件{}'.format(hf_repo, '逐字节相同' if match == 'identical' else '不完全相同（只作记录）'))
    record = {'time': now(), 'source': basis, 'origin': origin, 'files': files, 'hf_reference': hf_repo,
              'matches_hf': match}
    return problems, weights, record, rev


# ----------------------------------------------------------------------------- 主程序

def main():
    global MODELS_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument('--key', required=True)
    ap.add_argument('--job', required=True)
    a = ap.parse_args()
    cfg = load_config()
    MODELS_DIR = Path(os.path.expanduser(str((cfg.get('paths') or {}).get('models_dir') or '/home/<user>/models')))
    log = Log(RUNS / 'logs' / 'fetch_{}.log'.format(a.key))
    spec = cfg['models'][a.key]
    repos, ms_id, device_map, fallback = spec['repos'], spec.get('modelscope'), spec.get('device_map'), False
    env = read_json(RUNS / 'env.json', {}) or {}
    if device_map == 'auto' and spec.get('fallback_repos') and fallback_allowed(env):
        log('没有 accelerate 或不足两张 GPU（登记方案规定的条件）：{} 改用单卡模型 {}'.format(a.key, spec['fallback_repos']))
        repos, ms_id, device_map, fallback = spec['fallback_repos'], spec.get('fallback_modelscope'), None, True
    log('获取 {}：候选 {}，ModelScope {}（HF_ENDPOINT={}，Xet {}）'.format(
        a.key, repos, ms_id or '无', endpoint(), '已关闭' if os.environ.get('HF_HUB_DISABLE_XET') else '开启'))
    if MODELS_DIR.is_dir():
        describe_models_dir(MODELS_DIR, log)
    prev = model_registry().get(a.key) or {}
    hf_repo = hf_repo_of(repos)
    chosen, seen, bad_cache, rejected = None, set(), {}, {}
    try:
        for origin, path, repo, rev in candidates(repos, ms_id, prev, log, bad_cache):
            rp = str(Path(path).resolve())
            if origin in ('previous', 'local') and rp in seen:
                continue                     # 下载得到的候选即使目录相同也要重新核对（文件可能刚被替换）
            seen.add(rp)
            problems, weights, record, rev = verify(path, repo, rev, a.key, origin, prev, hf_repo, log)
            if problems and origin == 'modelscope':
                log('  ModelScope 下载的文件有问题，删除后重新下载一次：{}'.format('，'.join(problems)))
                for n in problems:
                    try:
                        (Path(path) / n).unlink()
                    except OSError:
                        pass
                if modelscope_download(ms_id, path, log) is not None:
                    problems, weights, record, rev = verify(path, repo, rev, a.key, origin, prev, hf_repo, log)
            if not problems:
                chosen = (origin, Path(path), repo, rev, weights, record)
                break
            desc = '；'.join('{}：{}'.format(n, p) for n, p in problems.items())
            log('  不能用：{}（{}）'.format(path, desc))
            rejected[str(path)] = desc
            if cache_repo(path):
                bad_cache[str(path)] = list(problems)
    except Exception as e:  # noqa: BLE001
        log('出错：{}: {}'.format(type(e).__name__, e))
    if chosen is None:
        log('失败：没有找到可用的 {}（候选 {}）。按登记方案不能改用备选模型。'.format(a.key, repos))
        log('可以在服务器上运行 modelscope download --model {} --local_dir {}/{} 下载后，再双击“一键启动实验.bat”。'.format(
            ms_id or hf_repo, MODELS_DIR, (ms_id or hf_repo or '').split('/')[-1]))
        sys.exit(3)
    origin, path, repo, rev, weights, record = chosen
    if rejected:
        record['rejected'] = rejected
    info = {'path': str(path), 'repo': repo, 'revision': rev, 'device_map': device_map, 'fallback': fallback,
            'origin': origin, 'weights_sha256': weights, 'verified': record, 'time': now()}
    register_model(a.key, info)
    log('完成：{}'.format(json.dumps({k: v for k, v in info.items() if k not in ('weights_sha256', 'verified')},
                                   ensure_ascii=False)))
    mark_done(a.job, {k: v for k, v in info.items() if k not in ('weights_sha256', 'verified')})


if __name__ == '__main__':
    main()
