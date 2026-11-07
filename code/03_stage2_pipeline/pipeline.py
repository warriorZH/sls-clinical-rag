#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_v2 自动流水线：按依赖关系运行全部作业，同时使用两张 GPU，中断后再次启动会从断点继续。

    python pipeline.py run       由 run_v2.sh start 在后台调用
    python pipeline.py status    打印进度表（也写入 status.txt）
    python pipeline.py plan      只列出作业，不运行

调度规则：
  - GPU 作业按优先级分配空闲的卡；生成和打分作业占一张卡，评审作业与第二个问答模型占两张卡。
  - 严格按优先级：排在前面的作业在等卡时，后面的作业不会抢先占用空出来的卡（避免两卡作业一直等不到卡）。
  - 主线（E0、E1、E3、E4、外部验证及其评审）优先；主线完成之前不开始 E2 微调，之后两张卡可同时微调。
  - 作业失败自动重试一次；退出码 3 表示需要你处理，不重试。
  - 每个作业成功后写 runs/done/<作业>.json；再次启动时已完成的作业直接跳过。
"""
import datetime as dt
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from common import EVAL_V1, HERE, RUNS, read_json, write_json, now, load_config, DONE_DIR
import conditions as C

STATE = RUNS / 'pipeline_state.json'
STOP = RUNS / 'STOP'
PIDFILE = RUNS / 'pipeline.pid'
LOGS = RUNS / 'logs'
TERMINAL = ('done', 'failed', 'needs_user', 'blocked', 'skipped')
MAX_CPU_JOBS = 3
CODE_FILE = RUNS / 'pipeline.code'
HEAVY_GIB = 10          # 预计主机内存峰值 ≥ 10 GiB 的作业（把大模型读进内存、建 BM25 索引）
LOAD_WINDOW = 240       # 大作业启动后 4 分钟内视为仍在加载模型：内存读数还没反映它的峰值


def J(jid, title, script, args=(), gpus=0, deps=(), after=(), prio=50, group=None, gate=False, optional=False,
      stage='', see_gpus=False, mainline=False, mem=2, mem_all_run=False):
    """mem：预计的主机内存峰值（GiB）；mem_all_run：整个运行期间都可能出现峰值（依次加载几个大模型的作业）。"""
    return {'id': jid, 'title': title, 'cmd': [script] + list(args) + ['--job', jid], 'gpus': gpus, 'deps': list(deps),
            'after': list(after), 'prio': prio, 'group': group or jid, 'gate': gate, 'optional': optional,
            'stage': stage, 'see_gpus': see_gpus, 'mainline': mainline, 'mem': mem, 'mem_all_run': mem_all_run}


def chunk(conds, n):
    """按估计成本把条件分成 n 份（贪心，结果固定）。"""
    parts = [[] for _ in range(n)]
    load = [0.0] * n
    for c in sorted(conds, key=lambda c: (-C.cost(c), c['id'])):
        i = min(range(n), key=lambda j: (load[j], j))
        parts[i].append(c['id'])
        load[i] += C.cost(c)
    return [p for p in parts if p]


def build_jobs(cfg):
    opt = cfg['optional']
    cs = C.conditions(cfg)
    g2 = opt.get('g2_generator', True)
    jobs = [
        J('env', '环境检查', 'env_check.py', prio=100, stage='准备', see_gpus=True, mainline=True),
        J('fetch_g1', '下载问答模型 Qwen3-4B-Instruct-2507', 'fetch_models.py', ['--key', 'g1'], deps=['env'], prio=99,
          stage='准备', mainline=True),
        J('fetch_judge_a', '下载评审 A：GLM-4-9B-0414（与 G2 同一模型）', 'fetch_models.py', ['--key', 'judge_a'],
          deps=['env'], prio=98, stage='准备', mainline=True),
        J('fetch_judge_b', '下载评审 B：Hunyuan-7B-Instruct', 'fetch_models.py', ['--key', 'judge_b'], deps=['env'],
          prio=95, optional=True, stage='准备'),
        J('inspect', '收集以前问答实验的代码与模型信息', 'inspect_qa.py', deps=['env'], prio=10, optional=True,
          stage='准备'),
    ]
    if opt.get('qwen3_reranker_4b', True):
        jobs.append(J('fetch_qr4b', '下载 Qwen3-Reranker-4B', 'fetch_models.py', ['--key', 'qr4b'], deps=['env'],
                      prio=94, optional=True, stage='准备'))
    if g2:
        jobs.append(J('fetch_g2', '登记 G2：GLM-4-9B-0414（与评审 A 同一模型）', 'fetch_models.py',
                      ['--key', 'g2'], deps=['env'], after=['fetch_judge_a'], prio=96, optional=True, stage='准备'))
    jobs += [
        J('e0_gen', 'E0 生成冒烟（校准集，跑两遍查可复现性）', 'generate.py', ['--smoke'], gpus=1, deps=['fetch_g1'],
          prio=95, stage='E0', mainline=True),
        J('e0_judge', 'E0 评审冒烟（加载两位本地评审试评）', 'judge.py', ['--smoke'], gpus=2,
          deps=['e0_gen', 'fetch_judge_a'], after=['fetch_judge_b'], prio=94, group='judge', stage='E0',
          mainline=True),
        J('e1_calib_ce', 'E1 校准集：CE 完整题干重打分', 'score_v2.py', ['--stage', 'e1_ce', '--target', 'calib'],
          gpus=1, deps=['env'], prio=92, stage='E1', mainline=True),
        J('e1_calib_pub', 'E1 校准集：公开模型文本对等打分', 'score_v2.py', ['--stage', 'e1_pub', '--target', 'calib'],
          gpus=1, deps=['env'], after=['fetch_g1', 'fetch_qr4b'], prio=91, stage='E1', mainline=True),
        J('e1_sealed_ce', 'E1 封存集：CE 完整题干重打分', 'score_v2.py', ['--stage', 'e1_ce', '--target', 'sealed'],
          gpus=1, deps=['e1_calib_ce'], prio=90, gate=True, stage='E1', mainline=True),
        J('e1_sealed_pub', 'E1 封存集：公开模型文本对等打分', 'score_v2.py', ['--stage', 'e1_pub', '--target', 'sealed'],
          gpus=1, deps=['e1_calib_pub'], prio=89, gate=True, stage='E1', mainline=True),
        J('e3_select', 'E3 按各方法分数选证据', 'select_evidence.py', ['--sets', 'main,ksens,palt'],
          deps=['e1_sealed_ce', 'e1_sealed_pub'], prio=88, stage='E3', mainline=True),
    ]
    gen_ids = []
    for i, part in enumerate(chunk(cs['main'] + cs['ksens'] + cs['palt'], int(cfg.get('e3_chunks', 4)))):
        jid = 'e3_gen_{}'.format(i + 1)
        gen_ids.append(jid)
        jobs.append(J(jid, 'E3/E5 生成（第 {} 份，{} 个条件）'.format(i + 1, len(part)), 'generate.py',
                      ['--conds', ','.join(part)], gpus=1, deps=['e3_select', 'e0_gen'], prio=80, stage='E3',
                      mainline=True))
    jobs += [
        J('e3_freeze', '冻结非 O 条件的答案', 'freeze.py', ['--sets', 'main,ksens,palt'], deps=gen_ids, prio=79,
          stage='E3', mainline=True),
        J('e4_gen', 'E4 生成（金标构造证据 O1–O4）', 'generate.py',
          ['--conds', ','.join(c['id'] for c in cs['oracle'])], gpus=1, deps=['e3_freeze'], prio=78, stage='E4',
          mainline=True),
        J('judge_main_A', '评审 A：E3 主条件答案（全部）', 'judge.py', ['--judge', 'A', '--sets', 'main'], gpus=2,
          deps=['e3_freeze', 'e0_judge'], prio=76, group='judge', stage='评审', mainline=True),
        J('judge_oracle_A', '评审 A：E4 答案（全部）', 'judge.py', ['--judge', 'A', '--sets', 'oracle'], gpus=2,
          deps=['e4_gen', 'e0_judge'], prio=75, group='judge', stage='评审', mainline=True),
        J('summary_main', '汇总主要结果', 'summarize.py', [], deps=['judge_main_A', 'judge_oracle_A'], prio=74,
          stage='汇总', mainline=True),
    ]
    b_after = ['judge_main_A', 'judge_oracle_A', 'judge_e5_A']
    b_sets = ['main', 'oracle', 'ksens', 'palt']
    final_after = ['summary_main', 'judge_e5_A']
    if cs['ext']:
        # 外部验证（TCM-BEST4SDT，方案第十六节）：属于主线，排在封存集主要结果之后、E5 之前
        jobs += [
            J('ext_build', '外部验证：排除近重复，建立 BM25 候选池', 'build_external.py', [], deps=['env'], prio=90,
              optional=True, stage='外部', mainline=True),
            J('e1_ext', '外部验证：CE 与公开模型打分', 'score_v2.py', ['--stage', 'ext', '--target', 'external'],
              gpus=1, deps=['ext_build', 'e1_calib_ce'], prio=87, gate=True, optional=True, stage='外部',
              mainline=True),
            J('ext_select', '外部验证：按各方法分数选证据', 'select_evidence.py', ['--sets', 'ext'], deps=['e1_ext'],
              prio=86, optional=True, stage='外部', mainline=True),
        ]
        ext_gen = []
        for i, part in enumerate(chunk(cs['ext'], int(cfg.get('ext_chunks', 2)))):
            jid = 'ext_gen_{}'.format(i + 1)
            ext_gen.append(jid)
            jobs.append(J(jid, '外部验证：生成（第 {} 份，{} 个条件）'.format(i + 1, len(part)), 'generate.py',
                          ['--conds', ','.join(part)], gpus=1, deps=['ext_select', 'e0_gen'], prio=77, optional=True,
                          stage='外部', mainline=True))
        jobs += [
            J('judge_ext_A', '评审 A：外部验证答案（全部）', 'judge.py', ['--judge', 'A', '--sets', 'ext'], gpus=2,
              deps=ext_gen + ['e0_judge'], after=['judge_main_A', 'judge_oracle_A'], prio=73, group='judge',
              optional=True, stage='评审', mainline=True),
            J('summary_ext', '汇总外部验证结果', 'summarize.py', [], deps=['judge_ext_A'], after=['summary_main'],
              prio=72, optional=True, stage='汇总', mainline=True),
        ]
        b_after.append('judge_ext_A')
        b_sets.append('ext')
        final_after += ['judge_ext_A', 'summary_ext']
    jobs.append(J('judge_e5_A', '评审 A：E5 答案（k 敏感度、第二套提示词）', 'judge.py',
                  ['--judge', 'A', '--sets', 'ksens,palt'], gpus=2, deps=['e3_freeze', 'e0_judge'], prio=70,
                  group='judge', stage='评审'))
    if g2:
        jobs += [
            J('g2_gen', 'E5 第二个问答模型生成（占两张卡）', 'generate.py', ['--conds', ','.join(c['id'] for c in cs['g2'])],
              gpus=2, deps=['fetch_g2', 'e3_freeze', 'e3_select'], prio=66, optional=True, stage='E5'),
            J('judge_g2_A', '评审 A：第二个问答模型的答案', 'judge.py', ['--judge', 'A', '--sets', 'g2'], gpus=2,
              deps=['g2_gen', 'e0_judge'], prio=64, group='judge', optional=True, stage='评审'),
        ]
        b_after.append('judge_g2_A')
        b_sets.append('g2')
        final_after.append('judge_g2_A')
    jobs.append(J('judge_B', '评审 B：25% 子集（一致性与敏感性）', 'judge.py', ['--judge', 'B', '--sets', ','.join(b_sets)],
                  gpus=2, deps=['e0_judge'], after=b_after, prio=60, group='judge', optional=True, stage='评审'))
    final_after.append('judge_B')
    if opt.get('e2_finetune', True):
        trains = []
        jobs.append(J('e2_smoke', 'E2 bge 微调冒烟（50 步）', 'train_bge.py', ['--smoke'], gpus=1, deps=['env'], prio=93,
                      stage='E2'))
        for obj in C.OBJECTIVES:
            for seed in C.SEEDS:
                jid = 'e2_train_{}_s{}'.format(obj, seed)
                trains.append(jid)
                jobs.append(J(jid, 'E2 微调 bge（{}标签，种子 {}）'.format('支持' if obj == 'support' else '相关性', seed),
                              'train_bge.py', ['--objective', obj, '--seed', str(seed)], gpus=1, deps=['e2_smoke'],
                              prio=50, group='e2_train', stage='E2'))
        jobs += [
            J('e2_calib', 'E2 校准集打分', 'score_v2.py', ['--stage', 'e2', '--target', 'calib'], gpus=1, deps=trains,
              prio=49, stage='E2'),
            J('e2_sealed', 'E2 封存集打分', 'score_v2.py', ['--stage', 'e2', '--target', 'sealed'], gpus=1,
              deps=['e2_calib'], prio=48, gate=True, stage='E2'),
            J('e2_select', 'E2 选证据', 'select_evidence.py', ['--sets', 'e2'], deps=['e2_sealed'], prio=47, stage='E2'),
            J('e2_gen', 'E2 生成（G10、G11）', 'generate.py', ['--conds', ','.join(c['id'] for c in cs['e2'])], gpus=1,
              deps=['e2_select', 'e3_freeze'], prio=46, stage='E2'),
            J('judge_e2_A', '评审 A：E2 答案（全部）', 'judge.py', ['--judge', 'A', '--sets', 'e2'], gpus=2,
              deps=['e2_gen', 'e0_judge'], prio=45, group='judge', stage='评审'),
            J('judge_e2_B', '评审 B：E2 答案（25% 子集）', 'judge.py', ['--judge', 'B', '--sets', 'e2'], gpus=2,
              deps=['e0_judge'], after=['judge_e2_A', 'judge_B'], prio=44, group='judge', optional=True, stage='评审'),
        ]
        final_after += ['judge_e2_A', 'judge_e2_B']
    jobs.append(J('summary_final', '最终汇总', 'summarize.py', [], after=final_after, prio=1, stage='汇总'))
    # 主机内存：fp32 的 4B–9B 模型读进内存时峰值约 16–20 GiB，放到 GPU 上之后常驻只剩几 GiB
    for j in jobs:
        jid = j['id']
        if jid in ('e0_judge', 'e1_calib_pub', 'e1_sealed_pub'):
            j.update(mem=20, mem_all_run=True)           # 依次加载两个大模型
        elif jid == 'ext_build':
            j.update(mem=10, mem_all_run=True)           # 全语料的 BM25 稀疏矩阵
        elif jid.startswith(('judge_', 'e3_gen_', 'ext_gen_')) or jid in ('e0_gen', 'e4_gen', 'g2_gen', 'e2_gen'):
            j['mem'] = 20
        elif jid.startswith(('e2_train_', 'e2_smoke')):
            j['mem'] = 6
        elif jid.startswith('e1_') or jid in ('e2_calib', 'e2_sealed'):
            j['mem'] = 4
    ids = {j['id'] for j in jobs}
    for j in jobs:
        j['deps'] = [d for d in j['deps'] if d in ids]
        j['after'] = [d for d in j['after'] if d in ids]
    return jobs


# ----------------------------------------------------------------------------- 状态

def load_state(jobs, finished=None):
    st = read_json(STATE, {}) or {}
    st.setdefault('jobs', {})
    for j in jobs:
        s = st['jobs'].setdefault(j['id'], {'status': 'pending', 'attempts': 0})
        marker = DONE_DIR / '{}.json'.format(j['id'])
        if marker.exists():
            if s.get('status') != 'done':
                # 在没有调度程序运行时结束的作业（例如换用新版本调度程序的那几秒）：结束时间取完成标记的时间
                try:
                    s['ended'] = dt.datetime.fromtimestamp(marker.stat().st_mtime).astimezone().isoformat(
                        timespec='seconds')
                except Exception:  # noqa: BLE001
                    pass
                if finished is not None and s.get('status') == 'running':
                    finished.append(j['id'])
            s['status'] = 'done'
        elif s.get('status') == 'done':
            s['status'] = 'pending'          # 完成标记被删除：重新运行该作业
    return st


def save_state(st):
    st['updated'] = now()
    write_json(STATE, st)


def pid_alive(pid, jid=None):
    try:
        os.kill(int(pid), 0)
    except Exception:  # noqa: BLE001
        return False
    if jid:
        try:
            cmd = Path('/proc/{}/cmdline'.format(pid)).read_bytes().replace(b'\0', b' ').decode('utf-8', 'ignore')
            return jid in cmd
        except Exception:  # noqa: BLE001
            return True
    return True


def n_gpus():
    env = read_json(RUNS / 'env.json', {}) or {}
    if 'n_gpus' in env:
        return int(env['n_gpus'])
    return None


def hf_endpoint():
    for p in (EVAL_V1 / '.hf_endpoint', HERE / '.hf_endpoint'):
        if p.exists():
            v = p.read_text(encoding='utf-8').strip()
            if v and v != 'https://huggingface.co':
                return v
    return os.environ.get('HF_ENDPOINT', '')


def gpu_busy_by_others(own):
    """用 nvidia-smi 查看各卡显存占用；不是本流水线占用、且已用超过 1.5 GB 的卡视为被别人占用。"""
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=20).stdout
    except Exception:  # noqa: BLE001
        return set()
    busy = set()
    for line in out.strip().splitlines():
        try:
            i, used = [x.strip() for x in line.split(',')[:2]]
            if int(i) not in own and float(used) > 1500:
                busy.add(int(i))
        except Exception:  # noqa: BLE001
            continue
    return busy


def _read_int(path):
    try:
        v = Path(path).read_text().strip()
        return None if v == 'max' else int(v)
    except Exception:  # noqa: BLE001
        return None


def _stat_value(path, key):
    try:
        for line in Path(path).read_text().splitlines():
            k, _, v = line.partition(' ')
            if k == key:
                return int(v)
    except Exception:  # noqa: BLE001
        pass
    return 0


def mem_info():
    """(可用内存, 总内存)，单位 GiB；取容器（cgroup）限制与整机中较小的那个。读不到时返回 (None, None)。"""
    host_total = host_avail = None
    try:
        info = {}
        for line in Path('/proc/meminfo').read_text().splitlines():
            k, _, v = line.partition(':')
            info[k.strip()] = int(v.split()[0]) * 1024
        host_total, host_avail = info.get('MemTotal'), info.get('MemAvailable')
    except Exception:  # noqa: BLE001
        pass
    cg_total = cg_avail = None
    lim = _read_int('/sys/fs/cgroup/memory.max')                                   # cgroup v2
    if lim is not None:
        cur = _read_int('/sys/fs/cgroup/memory.current') or 0
        cg_total, cg_avail = lim, lim - cur + _stat_value('/sys/fs/cgroup/memory.stat', 'inactive_file')
    else:
        lim = _read_int('/sys/fs/cgroup/memory/memory.limit_in_bytes')            # cgroup v1
        if lim is not None and lim < (1 << 60):
            cur = _read_int('/sys/fs/cgroup/memory/memory.usage_in_bytes') or 0
            cg_total, cg_avail = lim, lim - cur + _stat_value('/sys/fs/cgroup/memory/memory.stat', 'total_inactive_file')
    totals = [x for x in (host_total, cg_total) if x]
    avails = [x for x in (host_avail, cg_avail) if x is not None]
    if not totals or not avails:
        return None, None
    return max(0.0, min(avails)) / 2 ** 30, min(totals) / 2 ** 30


def audit_models(jobs, st):
    """登记表里的模型还没有逐文件核对过，或在登记方案规定的条件之外用了单卡备选模型：让对应的下载作业重新运行
    （10 月 28 日 hf-mirror 下载 GLM-4-9B-0414 失败后，旧版下载程序错误地改用了单卡备选模型）。"""
    reg = read_json(RUNS / 'models' / 'registry.json', {}) or {}
    env = read_json(RUNS / 'env.json', {}) or {}
    redo = []
    for j in jobs:
        if j['cmd'][0] != 'fetch_models.py' or st['jobs'][j['id']]['status'] != 'done' or '--key' not in j['cmd']:
            continue
        info = reg.get(j['cmd'][j['cmd'].index('--key') + 1]) or {}
        bad_fallback = bool(info.get('fallback')) and 'n_gpus' in env and not (
            str(env.get('accelerate', 'missing')).startswith('missing') or int(env.get('n_gpus') or 0) < 2)
        if info.get('verified') and not bad_fallback:
            continue
        try:
            (DONE_DIR / '{}.json'.format(j['id'])).unlink()
        except FileNotFoundError:
            pass
        st['jobs'][j['id']].update(status='pending', attempts=0, reason='')
        redo.append('{}（{}）'.format(j['id'], '单卡备选模型不符合登记条件' if bad_fallback else '模型文件尚未逐个核对'))
    return redo


def code_hash():
    """调度程序自身代码的哈希；run_v2.sh start 据此判断要不要换用新版本的调度程序。"""
    import hashlib
    return hashlib.sha256(b''.join((HERE / f).read_bytes() for f in ('pipeline.py', 'conditions.py', 'common.py'))
                          ).hexdigest()[:16]


def iso_epoch(s):
    try:
        return dt.datetime.fromisoformat(s).timestamp()
    except Exception:  # noqa: BLE001
        return 0.0


def find_job_procs():
    """本流水线正在运行的作业进程：命令行以 --job <作业号> 结尾、工作目录是本目录的会话首进程。
    返回 {作业号: (pid, GPU 列表, 开始时间)}。调度程序重启时用它接管状态文件里没有记下的作业，避免重复启动。"""
    out = {}
    try:
        btime = next(int(x.split()[1]) for x in Path('/proc/stat').read_text().splitlines() if x.startswith('btime'))
        tck = os.sysconf('SC_CLK_TCK')
    except Exception:  # noqa: BLE001
        btime, tck = None, 100
    here = str(HERE.resolve())
    try:
        entries = [d for d in Path('/proc').iterdir() if d.name.isdigit()]
    except Exception:  # noqa: BLE001
        return out
    for d in entries:
        try:
            argv = [a.decode('utf-8', 'ignore') for a in (d / 'cmdline').read_bytes().split(b'\0') if a]
            if len(argv) < 3 or argv[-2] != '--job' or os.readlink(str(d / 'cwd')) != here:
                continue
            pid = int(d.name)
            stat = (d / 'stat').read_text()
            f = stat[stat.rindex(')') + 2:].split()
            if int(f[3]) != pid:             # 只认会话首进程：作业派生的子进程命令行与作业相同
                continue
            env = {}
            for e in (d / 'environ').read_bytes().split(b'\0'):
                k, _, v = e.partition(b'=')
                env[k] = v.decode('utf-8', 'ignore')
            spec = env.get(b'EVAL_V2_GPUS', env.get(b'CUDA_VISIBLE_DEVICES', ''))
            gpus = [int(x) for x in spec.split(',') if x.strip().isdigit()]
            t0 = btime + int(f[19]) / tck if btime else time.time()
            out[argv[-1]] = (pid, gpus, t0)
        except Exception:  # noqa: BLE001
            continue
    return out


def job_env(gpus, fake=False, see_all=False):
    env = dict(os.environ)
    env['PYTHONUNBUFFERED'] = '1'
    env['TOKENIZERS_PARALLELISM'] = 'false'
    env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    env['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    env['PYTHONPATH'] = os.pathsep.join([str(HERE), str(HERE / 'pylib')] +
                                        ([env['PYTHONPATH']] if env.get('PYTHONPATH') else []))
    hf = hf_endpoint()
    if hf:
        env['HF_ENDPOINT'] = hf
    env['EVAL_V2_GPUS'] = ','.join(str(g) for g in gpus) if gpus else ''     # 调度程序重启时据此接管作业
    env['PYTHONFAULTHANDLER'] = '1'      # 进程因 SIGBUS、SIGSEGV 等信号退出时，把 Python 调用栈写进作业日志
    if see_all:
        env.pop('CUDA_VISIBLE_DEVICES', None)
    elif fake:
        env['CUDA_VISIBLE_DEVICES'] = ''
    elif gpus is None:
        env['CUDA_VISIBLE_DEVICES'] = ''          # CPU 作业不占用 GPU
    else:
        env['CUDA_VISIBLE_DEVICES'] = ','.join(str(g) for g in gpus)
    return env


# ----------------------------------------------------------------------------- 调度

def run(jobs):
    cfg = load_config()
    LOGS.mkdir(parents=True, exist_ok=True)
    import fcntl
    lockf = open(str(RUNS / 'pipeline.lock'), 'w')
    try:
        fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)      # 同一时间只允许一个调度程序
    except OSError:
        print('ALREADY_RUNNING')
        sys.exit(3)
    if STOP.exists():
        STOP.unlink()
    PIDFILE.write_text(str(os.getpid()), encoding='utf-8')
    CODE_FILE.write_text(code_hash(), encoding='utf-8')
    window = float(cfg.get('heavy_load_window_s', LOAD_WINDOW))
    # 收到 SIGTERM（换用新版本的调度程序）时在两轮调度之间退出，不会停在“作业已启动、状态还没记下”的中间
    term = []
    signal.signal(signal.SIGTERM, lambda *_: term.append(1))
    # 最终汇总在最后重新生成
    try:
        (DONE_DIR / 'summary_final.json').unlink()
    except FileNotFoundError:
        pass
    finished = []
    st = load_state(jobs, finished)
    st['started'] = now()
    st['orchestrator'] = 'running'
    byid = {j['id']: j for j in jobs}
    procs = {}
    orphans = []
    live = find_job_procs()
    # 上次中断时仍在运行的作业：进程还在就接管，否则重新排队
    for jid, s in st['jobs'].items():
        if s.get('status') == 'running' and s.get('pid') and pid_alive(s['pid'], jid):
            procs[jid] = {'pid': int(s['pid']), 'gpus': s.get('gpus') or [], 'adopted': True, 'proc': None,
                          't0': iso_epoch(s.get('started', ''))}
            continue
        if jid in byid and s.get('status') != 'done' and jid in live:
            # 已经在运行、但状态文件里没有记下的作业（旧调度程序恰好在启动它的瞬间退出）：接管，不重复启动
            pid, gpus, t0 = live[jid]
            s.update(status='running', pid=pid, gpus=gpus, log=str(LOGS / '{}.log'.format(jid)), reason='',
                     started=dt.datetime.fromtimestamp(t0).astimezone().isoformat(timespec='seconds'))
            procs[jid] = {'pid': pid, 'gpus': gpus, 'adopted': True, 'proc': None, 't0': t0}
            orphans.append(jid)
            continue
        if s.get('status') == 'running':
            s['status'] = 'pending'
        if s.get('status') in ('failed', 'needs_user', 'blocked', 'skipped') and jid in byid:
            s['status'] = 'pending'          # 重新启动时再试一次
            s['attempts'] = 0
    # 环境检查：没有正在运行的 GPU 作业时每次启动都重做（GPU 可能在两次启动之间出过故障或已经修好）；
    # 有 GPU 作业正在运行时 GPU 显然可用，沿用本次运行的检查结果（这时卡上显存已被占用，不宜重测）
    if not any(p['gpus'] for p in procs.values()) and 'env' not in procs:
        for p in (DONE_DIR / 'env.json', RUNS / 'env.json'):
            try:
                p.unlink()
            except FileNotFoundError:
                pass
        if 'env' in st['jobs']:
            st['jobs']['env']['status'] = 'pending'
    redo_models = audit_models(jobs, st)
    save_state(st)
    log = open(str(LOGS / 'pipeline.log'), 'a', encoding='utf-8')

    def note(msg):
        line = '[{}] {}'.format(dt.datetime.now().strftime('%m-%d %H:%M:%S'), msg)
        log.write(line + '\n')
        log.flush()
        print(line, flush=True)

    note('流水线启动，作业 {} 个'.format(len(jobs)))
    for jid in finished:
        note('完成：{}（{}）（在调度程序重启期间结束）'.format(jid, byid[jid]['title']))
    if redo_models:
        note('重新获取并核对模型：' + '，'.join(redo_models))
    if procs:
        note('接管正在运行的作业：' + '，'.join(sorted(procs)) +
             ('（其中 {} 由进程列表找回）'.format('，'.join(orphans)) if orphans else ''))
    retry_at = {}
    while True:
        try:
            if PIDFILE.read_text(encoding='utf-8').strip() != str(os.getpid()):
                PIDFILE.write_text(str(os.getpid()), encoding='utf-8')      # 被覆盖时自行恢复，进度表据此判断是否在运行
        except Exception:  # noqa: BLE001
            PIDFILE.write_text(str(os.getpid()), encoding='utf-8')
        if term:
            note('调度程序退出（换用新版本）；正在运行的作业继续运行，由新的调度程序接管')
            st['orchestrator'] = 'stopped'
            save_state(st)
            break
        if STOP.exists():
            note('收到停止请求，结束正在运行的作业')
            for jid, p in procs.items():
                try:
                    os.killpg(int(p['pid']), signal.SIGTERM)
                except Exception:  # noqa: BLE001
                    pass
                st['jobs'][jid]['status'] = 'pending'
            st['orchestrator'] = 'stopped'
            save_state(st)
            STOP.unlink()
            break
        # 1. 回收结束的作业
        for jid in list(procs):
            p = procs[jid]
            if p['proc'] is not None:
                code = p['proc'].poll()
                if code is None:
                    continue
            else:
                if pid_alive(p['pid'], jid):
                    continue
                code = 0 if (DONE_DIR / '{}.json'.format(jid)).exists() else 1
            s = st['jobs'][jid]
            s['ended'] = now()
            s['exit'] = code
            done_ok = (DONE_DIR / '{}.json'.format(jid)).exists()
            if code == 0 and done_ok:
                s['status'] = 'done'
                note('完成：{}（{}）'.format(jid, byid[jid]['title']))
            elif code == 3:
                s['status'] = 'needs_user'
                note('需要处理：{}，见日志 {}'.format(jid, s.get('log')))
            else:
                s['attempts'] = s.get('attempts', 0) + 1
                if s['attempts'] < 2:
                    s['status'] = 'pending'
                    retry_at[jid] = time.time() + 60
                    note('失败（退出码 {}），1 分钟后重试：{}'.format(code, jid))
                else:
                    s['status'] = 'failed'
                    note('失败（已重试）：{}，见日志 {}'.format(jid, s.get('log')))
            del procs[jid]
            save_state(st)
        # 2. 依赖失败的作业标为受阻；需要的 GPU 数超过实有数的作业不能运行
        ng = n_gpus()
        if ng == 0 and cfg.get('allow_cpu'):
            ng = int(cfg['allow_cpu'])            # 仅用于在没有 GPU 的机器上测试流程
        if ng is not None:
            for j in jobs:
                s = st['jobs'][j['id']]
                if s['status'] == 'pending' and j['gpus'] > ng:
                    s['status'] = 'skipped' if j['optional'] else 'blocked'
                    s['reason'] = '需要 {} 张 GPU，实有 {} 张'.format(j['gpus'], ng)
        changed = True
        while changed:
            changed = False
            for j in jobs:
                s = st['jobs'][j['id']]
                if s['status'] != 'pending':
                    continue
                bad = [d for d in j['deps'] if st['jobs'][d]['status'] in ('failed', 'needs_user', 'blocked',
                                                                            'skipped')]
                if bad:
                    s['status'] = 'skipped' if j['optional'] else 'blocked'
                    s['reason'] = '依赖未完成：' + ','.join(bad)
                    changed = True
        # 3. 就绪的作业
        ready = []
        for j in jobs:
            s = st['jobs'][j['id']]
            if s['status'] != 'pending' or j['id'] in procs:
                continue
            if retry_at.get(j['id'], 0) > time.time():
                continue
            if any(st['jobs'][d]['status'] != 'done' for d in j['deps']):
                continue
            if any(st['jobs'][d]['status'] not in TERMINAL for d in j['after']):
                continue
            if j['gate'] and not (HERE / 'REGISTRATION_v2.json').exists():
                s['reason'] = '等待本机上传 REGISTRATION_v2.json'
                continue
            ready.append(j)
        ready.sort(key=lambda j: -j['prio'])
        used = set(g for p in procs.values() for g in p['gpus'])
        others = gpu_busy_by_others(used) if (ng and not cfg.get('allow_cpu')) else set()
        free = [g for g in range(ng or 0) if g not in used and g not in others]
        st['gpus_busy_by_others'] = sorted(others)
        mainline_left = any(st['jobs'][j['id']]['status'] not in TERMINAL and j.get('mainline') for j in jobs)
        running_groups = {}
        for jid in procs:
            running_groups[byid[jid]['group']] = running_groups.get(byid[jid]['group'], 0) + 1
        cpu_running = sum(1 for jid in procs if byid[jid]['gpus'] == 0)
        # 主机内存：大作业（加载大模型、建索引）启动前确认内存够用；还在加载中的大作业按其峰值预留
        avail, total = mem_info()
        st['memory_gib'] = None if avail is None else {'available': round(avail, 1), 'total': round(total, 1)}
        tnow = time.time()
        loading_gib = sum(min(byid[jid]['mem'], 0.8 * total if total else byid[jid]['mem'])
                          for jid, p in procs.items() if byid[jid]['mem'] >= HEAVY_GIB and
                          (byid[jid]['mem_all_run'] or tnow - p.get('t0', 0) < window))
        reserved = False                    # 有更高优先级的 GPU 作业在等卡：其后的 GPU 作业不再占卡
        for j in ready:
            g = j['group']
            if g == 'e2_train':
                limit = 0 if mainline_left else max(1, ng or 1)
                if running_groups.get(g, 0) >= limit:
                    continue
            elif g == 'judge' and running_groups.get(g, 0) >= 1:
                continue
            need = 0.0
            if st['jobs'][j['id']].get('reason', '').startswith(('等内存', '等另一个作业加载完模型')):
                st['jobs'][j['id']]['reason'] = ''
            if j['mem'] >= HEAVY_GIB and procs:        # 没有任何作业在运行时不受此限制，避免一直等下去
                need = min(j['mem'], 0.8 * total) if total else j['mem']
                wait = ''
                if avail is None:
                    if loading_gib > 0:
                        wait = '等另一个作业加载完模型'
                elif avail < need + loading_gib:
                    wait = '等内存：可用 {:.1f} GiB，需要 {:.1f} GiB'.format(avail, need + loading_gib)
                if wait:
                    st['jobs'][j['id']]['reason'] = wait
                    if j['gpus']:
                        reserved = True                # 优先级更低的 GPU 作业不趁机占卡，保持排队顺序
                    continue
            if j['gpus'] == 0:
                if cpu_running >= MAX_CPU_JOBS:
                    continue
                alloc = None
            else:
                if ng is None or (ng or 0) < j['gpus']:
                    continue
                if reserved or len(free) < j['gpus']:
                    reserved = True
                    continue
                alloc = free[:j['gpus']]
                free = free[j['gpus']:]
            s = st['jobs'][j['id']]
            logf = LOGS / '{}.log'.format(j['id'])
            fh = open(str(logf), 'a', encoding='utf-8')
            fh.write('\n===== {} 启动 {}（GPU {}）=====\n'.format(now(), j['id'], alloc))
            fh.flush()
            fake = n_gpus() == 0 and cfg.get('allow_cpu')
            proc = subprocess.Popen([sys.executable] + j['cmd'], cwd=str(HERE),
                                    env=job_env(alloc, fake, see_all=j.get('see_gpus')), stdout=fh,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            procs[j['id']] = {'pid': proc.pid, 'gpus': alloc or [], 'proc': proc, 'adopted': False, 't0': time.time()}
            if need:
                loading_gib += need
            running_groups[g] = running_groups.get(g, 0) + 1
            if j['gpus'] == 0:
                cpu_running += 1
            s.update(status='running', started=now(), pid=proc.pid, gpus=alloc or [], log=str(logf), reason='')
            note('启动：{}（{}），GPU {}'.format(j['id'], j['title'], alloc if alloc is not None else '无'))
            save_state(st)
        # 4. 是否全部结束
        if not procs and all(st['jobs'][j['id']]['status'] in TERMINAL for j in jobs):
            st['orchestrator'] = 'finished'
            st['finished'] = now()
            save_state(st)
            note('全部作业已结束')
            break
        if not procs and not ready and not any(st['jobs'][j['id']]['status'] == 'pending' and
                                               retry_at.get(j['id'], 0) > time.time() for j in jobs):
            waiting = [j['id'] for j in jobs if st['jobs'][j['id']]['status'] == 'pending']
            if waiting and all(j['gate'] for j in jobs if j['id'] in waiting):
                st['orchestrator'] = 'waiting_registration'
            elif waiting and ng is None and st['jobs']['env']['status'] in TERMINAL:
                st['orchestrator'] = 'no_gpu_info'
                save_state(st)
                note('没有 GPU 信息，无法继续')
                break
        write_status(jobs, st)
        save_state(st)
        for _ in range(10):
            if term:
                break
            time.sleep(1)
    try:
        PIDFILE.unlink()
    except Exception:  # noqa: BLE001
        pass
    write_status(jobs, st)
    log.close()


# ----------------------------------------------------------------------------- 进度表

def width(s):
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(ch) in ('W', 'F') else 1 for ch in str(s))


def pad(s, w, right=False):
    s = str(s)
    while width(s) > w:
        s = s[:-1]
    fill = ' ' * (w - width(s))
    return fill + s if right else s + fill


NAMES = {'pending': '等待', 'running': '运行中', 'done': '完成', 'failed': '失败', 'needs_user': '需要处理',
         'blocked': '受阻', 'skipped': '已跳过（可选）'}


def fmt_dur(a, b=None):
    try:
        t0 = dt.datetime.fromisoformat(a)
        t1 = dt.datetime.fromisoformat(b) if b else dt.datetime.now().astimezone()
        s = int((t1 - t0).total_seconds())
        return '{}:{:02d}'.format(s // 3600, (s % 3600) // 60)
    except Exception:  # noqa: BLE001
        return ''


def last_line(path, limit=70):
    """作业日志的最后一行有效内容（去掉时间戳），用于在进度表里显示正在做什么。"""
    try:
        with open(str(path), 'rb') as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 4000))
            lines = f.read().decode('utf-8', 'ignore').splitlines()
    except Exception:  # noqa: BLE001
        return ''
    for ln in reversed(lines):
        ln = ln.strip()
        if not ln or 'Warning' in ln or 'warn' in ln or ln.startswith('=====') or '%|' in ln:
            continue
        if ln.startswith('[') and '] ' in ln:
            ln = ln.split('] ', 1)[1]
        return ln[:limit]
    return ''


def orchestrator_alive():
    if not PIDFILE.exists():
        return False
    try:
        return pid_alive(int(PIDFILE.read_text().strip()), 'pipeline.py')
    except Exception:  # noqa: BLE001
        return False


def done_note(jid, s):
    """已完成作业的补充说明：例如评审 B 在 E0 中不可用、相应作业没有实际运行。"""
    if s.get('status') != 'done':
        return ''
    try:
        dm = read_json(DONE_DIR / '{}.json'.format(jid), {}) or {}
    except Exception:  # noqa: BLE001
        return ''
    if dm.get('skipped'):
        return '未运行：' + {'judge B unavailable in E0': '评审 B 在 E0 中不可用'}.get(dm['skipped'], str(dm['skipped']))
    missing = [k for k, v in (dm.get('judges') or {}).items() if not v]
    return '评审 {} 不可用，只用评审 A'.format('、'.join(missing)) if missing else ''


def write_status(jobs, st):
    lines = []
    alive = orchestrator_alive()
    orch = st.get('orchestrator', '')
    bad = [j['id'] for j in jobs if st['jobs'][j['id']]['status'] in ('failed', 'needs_user', 'blocked')]
    if alive:
        overall = '运行中'
    elif orch == 'finished':
        overall = '已结束，但有作业需要处理（见下方）' if bad else '已全部结束'
    else:
        overall = {'stopped': '已停止'}.get(orch, '未运行')
    lines.append('eval_v2 实验进度（{}）    流水线：{}'.format(now()[:16].replace('T', ' '), overall))
    cnt = {}
    for j in jobs:
        s = st['jobs'][j['id']]['status']
        cnt[s] = cnt.get(s, 0) + 1
    lines.append('作业 {} 个：'.format(len(jobs)) + '，'.join('{} {}'.format(NAMES[k], v) for k, v in cnt.items()))
    avail, total = mem_info()
    if avail is not None:
        lines.append('服务器内存：可用 {:.0f} GiB，共 {:.0f} GiB'.format(avail, total))
    lines.append('')
    gpu_line = []
    for j in jobs:
        s = st['jobs'][j['id']]
        if s['status'] == 'running':
            gpu_line.append('{} → GPU {}（已运行 {}）{}'.format(
                j['title'], ','.join(map(str, s.get('gpus') or [])) or '无', fmt_dur(s.get('started', '')),
                ('\n      最近：' + last_line(s.get('log'))) if last_line(s.get('log')) else ''))
    if st.get('gpus_busy_by_others'):
        lines.append('注意：GPU {} 正被其他程序占用，本实验暂不使用这些卡。'.format(st['gpus_busy_by_others']))
    if gpu_line:
        lines.append('正在运行：')
        lines += ['  ' + x for x in gpu_line]
        lines.append('')
    lines.append('{} {} {} {}  {}'.format(pad('阶段', 6), pad('作业', 44), pad('状态', 14), pad('用时', 6, True), '说明'))
    for j in jobs:
        s = st['jobs'][j['id']]
        dur = fmt_dur(s.get('started', ''), s.get('ended') if s['status'] != 'running' else None) \
            if s.get('started') else ''
        lines.append('{} {} {} {}  {}'.format(pad(j['stage'], 6), pad(j['title'], 44),
                                             pad(NAMES.get(s['status'], s['status']), 14), pad(dur, 6, True),
                                             s.get('reason', '') or done_note(j['id'], s)))
    envrep = read_json(RUNS / 'env.json', {}) or {}
    if envrep.get('problems') and st['jobs'].get('env', {}).get('status') in ('failed', 'needs_user'):
        lines.append('')
        lines.append('环境检查发现的问题：')
        lines += ['  - ' + p for p in envrep['problems']]
    todo = []
    for j in jobs:
        s = st['jobs'][j['id']]
        if s['status'] in ('failed', 'needs_user'):
            todo.append('{}：{}。日志 {}。请把“查看进度并下载结果”窗口的内容发给 Claude。'.format(
                j['title'], NAMES[s['status']], s.get('log', '')))
    if todo:
        lines.append('')
        lines.append('需要你处理的事项：')
        lines += ['  - ' + x for x in todo]
    text = '\n'.join(lines) + '\n'
    (HERE / 'status.txt').write_text(text, encoding='utf-8')
    return text


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'status'
    cfg = load_config()
    jobs = build_jobs(cfg)
    if cmd == 'plan':
        for j in jobs:
            print('{:<26} gpus={} prio={:<3} deps={} after={} {}'.format(j['id'], j['gpus'], j['prio'], j['deps'],
                                                                      j['after'], j['title']))
        return
    if cmd == 'status':
        st = load_state(jobs)
        print(write_status(jobs, st))
        return
    if cmd == 'run':
        if orchestrator_alive() and int(PIDFILE.read_text().strip()) != os.getpid():
            print('ALREADY_RUNNING')
            sys.exit(3)
        run(jobs)
        return
    print(__doc__)


if __name__ == '__main__':
    main()
