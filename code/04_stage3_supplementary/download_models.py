#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 ModelScope 下载模型（三个服务模型；S4 另加两个重排模型）到 $SUP_MODELS，计算每个文件的 SHA256，写入 models/registry.json。
已下载且文件大小与登记表一致的模型跳过。"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from common_sup import CFG, WORK, log

MODELS = Path(os.environ.get('SUP_MODELS') or (WORK / 'models'))
REG = MODELS / 'registry.json'


def sha256_file(p, bs=8 << 20):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while True:
            b = f.read(bs)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def files_of(d):
    return sorted(p for p in Path(d).rglob('*') if p.is_file() and '/.' not in str(p.relative_to(d)) and not p.name.startswith('.'))


def main():
    MODELS.mkdir(parents=True, exist_ok=True)
    reg = json.loads(REG.read_text(encoding='utf-8')) if REG.exists() else {}
    from modelscope import snapshot_download
    import modelscope
    todo = [(r, CFG['models'][r]) for r in ('gen', 'rm', 'judge')]
    if CFG.get('s4', {}).get('enabled'):
        todo += [(k, v) for k, v in CFG['s4']['models'].items()]
    if '--only' in sys.argv:                      # 先下载小模型做 vLLM 预检：--only gen,q06
        only = sys.argv[sys.argv.index('--only') + 1].split(',')
        todo = [(r, m) for r, m in todo if r in only]
    for role, m in todo:
        d = MODELS / m['dir']
        old = reg.get(role)
        if old and all((d / f).exists() and (d / f).stat().st_size == v['size'] for f, v in old['files'].items()) \
                and any(f.endswith('.safetensors') for f in old['files']):
            log('{}（{}）已下载并核对过，跳过'.format(role, m['modelscope']))
            continue
        for attempt in range(1, 4):
            try:
                log('下载 {}：{} → {}（第 {} 次）'.format(role, m['modelscope'], d, attempt))
                snapshot_download(m['modelscope'], local_dir=str(d))
                break
            except Exception as e:  # noqa: BLE001
                log('下载失败：{}'.format(repr(e)[:300]))
                if attempt == 3:
                    raise
                time.sleep(30)
        fs = files_of(d)
        if not any(p.suffix == '.safetensors' for p in fs):
            raise SystemExit('{} 目录中没有 safetensors 权重文件：{}'.format(role, d))
        idx = d / 'model.safetensors.index.json'
        if idx.exists():                          # 分片是否齐全
            need = set(json.loads(idx.read_text(encoding='utf-8'))['weight_map'].values())
            miss = [f for f in need if not (d / f).exists()]
            if miss:
                raise SystemExit('{} 缺少权重分片：{}（重新运行 bash run_all.sh 会续传）'.format(role, ', '.join(sorted(miss))))
        if not (d / 'config.json').exists() or not any((d / f).exists() for f in ('tokenizer.json', 'vocab.json')):
            raise SystemExit('{} 缺少 config.json 或分词器文件：{}'.format(role, d))
        log('计算 {} 的文件 SHA256（{} 个文件）……'.format(role, len(fs)))
        reg[role] = {'modelscope': m['modelscope'], 'dir': str(d), 'time': time.strftime('%Y-%m-%d %H:%M:%S'),
                     'modelscope_version': modelscope.__version__,
                     'files': {str(p.relative_to(d)): {'size': p.stat().st_size, 'sha256': sha256_file(p)} for p in fs}}
        REG.write_text(json.dumps(reg, ensure_ascii=False, indent=1), encoding='utf-8')
        log('{} 完成：{:.1f} GB'.format(role, sum(v['size'] for v in reg[role]['files'].values()) / 1e9))


if __name__ == '__main__':
    main()
