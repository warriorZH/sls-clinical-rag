#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S4：解压从 P40 服务器导出的 stage4 训练数据包（stage4_export.tar.gz），核对哈希与样本数，并确认 stage4 自带的
评测函数可以载入。导出包由本机的“导出S4训练数据.bat”生成，请与 eval_v2_sup.zip 放在同一目录上传。"""
import hashlib
import json
import os
import shutil
import tarfile
from pathlib import Path

from common_sup import ROOT, WORK, log, write_json
from s4_common import S4, STAGE4, stage4_modules


def find_pack():
    for d in (ROOT.parent, ROOT, WORK, Path.home()):
        p = d / 'stage4_export.tar.gz'
        if p.exists():
            return p
    return None


def main():
    if (STAGE4 / 'READY.json').exists():
        log('stage4 训练数据已就绪', 's4')
        return
    pack = find_pack()
    if not pack:
        raise SystemExit('找不到 stage4_export.tar.gz。请在本机双击“导出S4训练数据.bat”，把生成的 stage4_export.tar.gz '
                         '上传到与 eval_v2_sup.zip 相同的目录（如 /root/autodl-tmp/），再运行 bash run_all.sh。'
                         '若不做 S4，可把 config/sup.json 中 s4 的 enabled 改为 false。')
    if STAGE4.exists():
        shutil.rmtree(STAGE4)
    STAGE4.parent.mkdir(parents=True, exist_ok=True)
    tmp = STAGE4.parent / 'stage4_extract'
    if tmp.exists():
        shutil.rmtree(tmp)
    with tarfile.open(pack) as t:
        for m in t.getmembers():                 # 只允许相对路径，防止解压到别处
            if m.name.startswith('/') or '..' in Path(m.name).parts:
                raise SystemExit('导出包中有不安全的路径：{}'.format(m.name))
        try:
            t.extractall(tmp, filter='data')
        except TypeError:
            t.extractall(tmp)
    shutil.move(str(tmp / 'stage4'), str(STAGE4))
    shutil.rmtree(tmp)
    man = json.loads((STAGE4 / 'EXPORT_MANIFEST.json').read_text(encoding='utf-8'))
    bad = []
    for rel, h in man['files'].items():
        p = STAGE4 / rel
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != h:
            bad.append(rel)
    if bad:
        raise SystemExit('导出包文件与清单不符：{}'.format(', '.join(bad)))
    import torch
    data = torch.load(str(STAGE4 / 'prepared.pt'), map_location='cpu', weights_only=False)
    counts = {k: len(data[k]) for k in ('train', 'dev', 'test')}
    row = data['train'][0]
    for f in ('question', 'text', 'y_support', 'y_relevance'):
        if f not in row:
            raise SystemExit('prepared.pt 的训练样本缺少字段 {}'.format(f))
    stage4_modules()
    info = {'pack': str(pack), 'pack_sha256': hashlib.sha256(pack.read_bytes()).hexdigest(), 'counts': counts,
            'export_manifest': {k: v for k, v in man.items() if k != 'files'},
            'train_matches_expected': counts['train'] == int(S4['train']['expected_train_pairs'])}
    write_json(STAGE4 / 'READY.json', info)
    log('stage4 训练数据就绪：{}'.format(json.dumps(counts)), 's4')
    if not info['train_matches_expected']:
        log('注意：训练对数 {} 与预期 {} 不同'.format(counts['train'], S4['train']['expected_train_pairs']), 's4')


if __name__ == '__main__':
    main()
