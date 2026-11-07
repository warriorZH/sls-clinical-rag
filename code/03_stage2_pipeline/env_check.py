#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S0 环境检查：GPU、Python 包、输入文件与哈希、网络（hf-mirror）、磁盘。结果写入 runs/env.json。"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

from common import EVAL_V1, HERE, RUNS, Log, environment, load_config, load_target, mark_done, now, resolve, write_json
import conditions as C


def check_url(url, timeout=15):
    try:
        req = urllib.request.Request(url, method='GET', headers={'User-Agent': 'eval_v2-env-check'})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return 'ok (HTTP {})'.format(r.status)
    except urllib.error.HTTPError as e:
        return 'reachable (HTTP {})'.format(e.code)      # 401/404 也说明网络可达
    except Exception as e:  # noqa: BLE001
        return 'unreachable ({}: {})'.format(type(e).__name__, str(e)[:120])


def nvidia_smi_diag():
    """nvidia-smi -L 的输出（GPU 不可用时帮助判断原因，例如“Failed to initialize NVML”通常要重启服务器）。"""
    try:
        r = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True, timeout=30)
        out = ' '.join((r.stdout + ' ' + r.stderr).split())
        return out[:240] or '无输出'
    except Exception as e:  # noqa: BLE001
        return '无法运行（{}）'.format(type(e).__name__)


def accelerate_importable(lib):
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join([str(lib)] + ([env['PYTHONPATH']] if env.get('PYTHONPATH') else []))
    r = subprocess.run([sys.executable, '-c', 'import accelerate, sys; sys.stdout.write(accelerate.__version__)'],
                       capture_output=True, text=True, env=env, timeout=120)
    return r.stdout.strip() if r.returncode == 0 else None


def ensure_accelerate(log):
    """G2（9B，fp32）需要用 accelerate 把模型分到两张卡上。缺少时装到 eval_v2/pylib（不改动系统环境），
    装完必须能真正导入才算可用；否则 G2 自动改用单卡模型。"""
    lib = HERE / 'pylib'
    v = accelerate_importable(lib)
    if v:
        return 'ok ({})'.format(v)
    log('  未找到可用的 accelerate，尝试安装到 {}（不改动系统环境）'.format(lib))
    for pkgs in (['accelerate'], ['psutil', 'pyyaml']):
        try:
            r = subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--target', str(lib)] + pkgs,
                               capture_output=True, text=True, timeout=600)
            if r.returncode != 0:
                log('  安装 {} 失败：{}'.format(pkgs, (r.stderr or r.stdout)[-300:]))
        except Exception as e:  # noqa: BLE001
            log('  安装 {} 失败：{}'.format(pkgs, e))
        v = accelerate_importable(lib)
        if v:
            return 'ok ({}, eval_v2/pylib)'.format(v)
    return 'missing'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--job', required=True)
    a = ap.parse_args()
    cfg = load_config()
    log = Log(RUNS / 'logs' / 'env.log')
    rep = {'time': now(), 'env': environment(), 'problems': [], 'warnings': []}
    log('环境：' + json.dumps(rep['env'], ensure_ascii=False))
    try:
        import torch
        n = torch.cuda.device_count()
        rep['n_gpus'] = n
        if n == 0 and not cfg.get('allow_cpu'):
            rep['nvidia_smi'] = nvidia_smi_diag()
            rep['problems'].append('服务器上没有可用的 GPU（PyTorch 看到 0 张卡；nvidia-smi：{}）。这通常是显卡驱动出了问题，'
                                   '重启服务器或重新加载驱动后，再次双击“一键启动实验.bat”即可从这里继续。'.format(rep['nvidia_smi']))
        caps = [torch.cuda.get_device_capability(i) for i in range(n)]
        rep['capabilities'] = caps
        if n:
            x = torch.randn(64, 64, device='cuda')
            rep['cuda_matmul_ok'] = bool(torch.isfinite(x @ x).all())
    except Exception as e:  # noqa: BLE001
        rep['nvidia_smi'] = nvidia_smi_diag()
        rep['problems'].append('PyTorch 无法使用 GPU：{}（nvidia-smi：{}）。重启服务器或重新加载驱动后再次启动。'.format(
            str(e)[:200], rep['nvidia_smi']))
    for mod in ('transformers', 'sentencepiece', 'huggingface_hub', 'numpy', 'safetensors'):
        try:
            m = __import__(mod)
            rep.setdefault('packages', {})[mod] = getattr(m, '__version__', 'ok')
        except Exception as e:  # noqa: BLE001
            rep['problems'].append('缺少 Python 包 {}（{}）'.format(mod, type(e).__name__))
    try:
        import transformers
        v = tuple(int(x) for x in transformers.__version__.split('.')[:2])
        if v < (4, 51):
            rep['problems'].append('transformers 版本 {} 低于 4.51（Qwen3 需要）'.format(transformers.__version__))
    except Exception:  # noqa: BLE001
        pass
    rep['accelerate'] = ensure_accelerate(log)      # 两张卡合放一个模型（评审 A、B 与 G2）需要它
    if rep['accelerate'] == 'missing':
        rep['warnings'].append('没有可用的 accelerate：评审与 G2 将改用单卡可放下的较小模型（见方案 v2.2）')
    # 输入文件
    for target in ('calib', 'sealed'):
        try:
            qs, pools = load_target(cfg, target)
            rep[target] = {'questions': len(qs), 'pairs': sum(len(v) for v in pools.values())}
        except Exception as e:  # noqa: BLE001
            rep['problems'].append('{} 输入检查失败：{}'.format(target, e))
    v1 = C.v1_models()
    if not v1:
        rep['problems'].append('找不到 eval_v1/models.json')
    for name, m in v1.items():
        p = m.get('path')
        if m.get('kind') == 'eee_ce_pt' and not Path(p).is_file():
            rep['problems'].append('找不到 CE 检查点 {}'.format(p))
    for target in ('calib', 'sealed'):
        for name in ('support_ce_s42', 'relevance_ce_s42', 'bge_reranker_v2_m3', 'qwen3_reranker_0_6b', 'bm25'):
            p = EVAL_V1 / 'scores' / target / '{}.jsonl'.format(name)
            if not p.exists():
                rep['problems'].append('缺少 eval_v1 分数文件 {}'.format(p))
    if not (EVAL_V1 / 'server_score.py').exists():
        rep['problems'].append('缺少 eval_v1/server_score.py')
    s4 = resolve(cfg['e2']['stage4_dir'])
    if cfg['optional'].get('e2_finetune', True):
        dirs = [s4, s4.parents[1], s4.parent]
        for f in ('prepared.pt', 'policy_stage4.py', 'evaluate_v2.py'):
            found = [str(d / f) for d in dirs if (d / f).exists()]
            rep.setdefault('e2_files', {})[f] = found[0] if found else None
            if not found:
                rep['warnings'].append('E2 需要的 {} 没有找到（查找了 {}）；E2 作业会失败，其余实验不受影响'.format(
                    f, '、'.join(str(d) for d in dirs)))
    if not (HERE / 'data' / 'oracle_lists.jsonl').exists():
        rep['problems'].append('缺少 data/oracle_lists.jsonl')
    ce = cfg.get('external')
    if ce and cfg['optional'].get('external', True):
        paths = {k: resolve(ce[k]) for k in ('questions_all', 'reference', 'corpus', 'stage4_dataset', 'split_manifest',
                                              'dataset_root')}
        rep['external_inputs'] = {k: str(v) for k, v in paths.items()}
        missing = [k for k, v in paths.items() if not v.exists()]
        if missing:
            rep['warnings'].append('外部验证需要的文件没有找到：{}；外部验证作业会失败，其余实验不受影响'.format(
                '、'.join(str(paths[k]) for k in missing)))
    if not (HERE / 'REGISTRATION_v2.json').exists():
        rep['warnings'].append('没有 REGISTRATION_v2.json（本机启动器会上传）')
    du = shutil.disk_usage(str(HERE))
    rep['disk_free_gib'] = round(du.free / 2 ** 30, 1)
    if du.free < 100 * 2 ** 30:
        rep['warnings'].append('磁盘剩余 {:.0f} GB，少于建议的 100 GB（四个模型约 50 GB，E2 检查点约 15 GB）'.format(
            du.free / 2 ** 30))
    hf = os.environ.get('HF_ENDPOINT', 'https://huggingface.co')
    rep['network'] = {'hf_endpoint': hf, 'hf': check_url(hf.rstrip('/') + '/api/models/BAAI/bge-reranker-v2-m3')}
    for jn, jc in cfg['judges'].items():
        if isinstance(jc, dict) and jc.get('backend', 'api') == 'api':
            rep['network']['judge_' + jn] = check_url(jc['base_url'].rstrip('/') + '/models')
    write_json(RUNS / 'env.json', rep)
    for p in rep['problems']:
        log('问题：' + p)
    for w in rep['warnings']:
        log('提醒：' + w)
    log('网络：' + json.dumps(rep['network'], ensure_ascii=False))
    if rep['problems']:
        sys.exit(3)                     # 环境问题重试也不会好：直接标为“需要处理”，进度表里会列出原因
    mark_done(a.job, {'n_gpus': rep.get('n_gpus'), 'accelerate': rep['accelerate']})


if __name__ == '__main__':
    main()
