#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在 P40 服务器上运行（由本机“导出S4训练数据.bat”上传并调用）：把 stage4 的训练数据 prepared.pt 与 stage4 自带的
评测函数（evaluate_v2.py、policy_stage4.py 及它们导入的本地模块）打包成 /tmp/stage4_export.tar.gz。
目录顺序与 eval_v2 的 train_bge.py 相同：stage4 目录、项目根目录、experiments 目录。只读，不修改服务器上的任何文件。"""
import hashlib, io, json, os, platform, sys, tarfile, time

S4 = '/home/<user>/UUM/EEE_research/experiments/stage4_retraining'
roots = [S4, os.path.dirname(os.path.dirname(S4)), os.path.dirname(S4)]
roots = [os.path.realpath(r) for r in roots]
for r in roots:
    if os.path.isdir(r) and r not in sys.path:
        sys.path.append(r)
before = set(sys.modules)
import evaluate_v2, policy_stage4  # noqa: E401,E402
try:
    import torch
    data = torch.load(os.path.join(S4, 'prepared.pt'), map_location='cpu', weights_only=False)
    counts = {k: len(data[k]) for k in ('train', 'dev', 'test') if k in data}
    fields = sorted(data['train'][0].keys())
    torch_v = torch.__version__
except Exception as e:  # noqa: BLE001
    counts, fields, torch_v = {'error': repr(e)}, [], None
local, third, skipped = {}, {}, []
for name, mod in list(sys.modules.items()):
    if name in before:
        continue
    f = getattr(mod, '__file__', None)
    if not f or not isinstance(f, str):
        continue
    f = os.path.realpath(f)
    if not os.path.isfile(f):          # 有些模块的 __file__ 是虚的或相对路径（如 torch 的 _ops），磁盘上并无此文件，跳过
        skipped.append('{} -> {}'.format(name, f))
        continue
    owner = next((i for i, r in enumerate(roots) if f.startswith(r + os.sep)), None)
    if owner is not None:
        local[f] = 'lib{}/{}'.format(owner, os.path.relpath(f, roots[owner]))
    elif 'site-packages' in f or 'dist-packages' in f:
        top = name.split('.')[0]
        try:
            from importlib.metadata import version
            third[top] = version(top)
        except Exception:  # noqa: BLE001
            third.setdefault(top, None)
files = {os.path.join(S4, 'prepared.pt'): 'prepared.pt'}
files.update(local)
sha = {}
out = '/tmp/stage4_export.tar.gz'
with tarfile.open(out, 'w:gz') as t:
    for src, arc in files.items():
        t.add(src, arcname='stage4/' + arc)
        h = hashlib.sha256()
        with open(src, 'rb') as fh:
            for b in iter(lambda: fh.read(1 << 20), b''):
                h.update(b)
        sha[arc] = h.hexdigest()
    man = {'created': time.strftime('%Y-%m-%d %H:%M:%S'), 'host': platform.node(), 'python': sys.version.split()[0],
           'torch': torch_v, 'stage4_dir': S4, 'search_roots': roots, 'counts': counts, 'train_fields': fields,
           'third_party_modules': third, 'skipped_modules_without_file': skipped, 'files': sha}
    b = json.dumps(man, ensure_ascii=False, indent=1).encode('utf-8')
    ti = tarfile.TarInfo('stage4/EXPORT_MANIFEST.json')
    ti.size = len(b)
    ti.mtime = int(time.time())
    t.addfile(ti, io.BytesIO(b))
print('EXPORT_OK', json.dumps({'counts': counts, 'n_files': len(sha), 'third_party': third}, ensure_ascii=False))
print('SIZE', os.path.getsize(out))
