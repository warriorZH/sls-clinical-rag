#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""收集服务器上以前问答实验的代码、提示词和模型信息（只读，供论文与对照使用；失败不影响主流程）。"""
import argparse
import json
import os
import shutil
from pathlib import Path

from common import RUNS, Log, load_config, mark_done, now, write_json

QA_DIRS = ['multimodel_qa_v1', 'multimodel_qa_v2', 'single_gpu_fp16_v1', 'qa_validation_v1', 'model_availability_20261015']
KEEP = ('.py', '.sh', '.json', '.md', '.txt', '.yaml', '.yml', '.csv')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--job', required=True)
    a = ap.parse_args()
    cfg = load_config()
    log = Log(RUNS / 'logs' / 'inspect.log')
    root = Path(cfg['paths']['experiments_root'])
    out = RUNS / 'inspect'
    code = out / 'code'
    rep = {'time': now(), 'qa_dirs': {}, 'models': [], 'adapters': []}
    budget = 40 * 2 ** 20
    used = 0
    for name in QA_DIRS:
        d = root / name
        if not d.exists():
            rep['qa_dirs'][name] = 'missing'
            continue
        files = []
        for p in sorted(d.rglob('*')):
            if 'env' in p.parts and 'site-packages' in str(p):
                continue
            if p.is_file() and p.suffix in KEEP:
                try:
                    size = p.stat().st_size
                except OSError:
                    continue
                files.append({'path': str(p.relative_to(d)), 'size': size})
                if size <= 2 * 2 ** 20 and used + size <= budget:
                    dest = code / name / p.relative_to(d)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(p), str(dest))
                    used += size
            if len(files) > 3000:
                break
        rep['qa_dirs'][name] = {'n_files': len(files), 'files': files[:400]}
    for base in ('/home/<user>/models', '~/.cache/huggingface/hub', '~/.cache/modelscope/hub', '/home/<user>/UUM'):
        b = Path(os.path.expanduser(base))
        if not b.exists():
            continue
        for cfgf in b.rglob('config.json'):
            if len(cfgf.parts) - len(b.parts) > 7 or 'site-packages' in str(cfgf):
                continue
            try:
                c = json.loads(cfgf.read_text(encoding='utf-8'))
            except Exception:  # noqa: BLE001
                continue
            arch = c.get('architectures') or []
            if any('CausalLM' in x or 'ConditionalGeneration' in x for x in arch):
                rep['models'].append({'path': str(cfgf.parent), 'architectures': arch,
                                      'model_type': c.get('model_type'), 'name_or_path': c.get('_name_or_path')})
        for ad in b.rglob('adapter_config.json'):
            try:
                c = json.loads(ad.read_text(encoding='utf-8'))
                rep['adapters'].append({'path': str(ad.parent), 'base_model': c.get('base_model_name_or_path'),
                                        'r': c.get('r'), 'target_modules': c.get('target_modules')})
            except Exception:  # noqa: BLE001
                continue
    write_json(out / 'inspect_qa.json', rep)
    log('以前的问答实验目录：{}'.format({k: (v if isinstance(v, str) else v['n_files']) for k, v in rep['qa_dirs'].items()}))
    log('找到生成模型 {} 个，LoRA 适配器 {} 个'.format(len(rep['models']), len(rep['adapters'])))
    mark_done(a.job, {'models': len(rep['models']), 'adapters': len(rep['adapters'])})


if __name__ == '__main__':
    main()
