#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""核对一键包文件与登记时的 FREEZE_MANIFEST_sup.json；有改动时记入 logs/变更记录_sup.txt（不阻止运行）。"""
import hashlib, json, os, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent.parent
man = json.loads((HERE / 'FREEZE_MANIFEST_sup.json').read_text(encoding='utf-8'))
changed = []
for rel, h in man['files'].items():
    p = HERE / rel
    if not p.exists():
        changed.append((rel, '缺失'))
    elif hashlib.sha256(p.read_bytes()).hexdigest() != h:
        changed.append((rel, '已改动'))
log = Path(os.environ.get('SUP_WORK', HERE)) / 'logs' / '变更记录_sup.txt'
if changed:
    with open(log, 'a', encoding='utf-8') as f:
        f.write('{} 与登记版本不同的文件：{}\n'.format(time.strftime('%Y-%m-%d %H:%M:%S'), '；'.join('{}（{}）'.format(a, b) for a, b in changed)))
    print('注意：{} 个文件与登记版本不同，已记入 {}'.format(len(changed), log))
else:
    print('一键包文件与登记版本一致（{} 个文件）'.format(len(man['files'])))
