#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""启动 / 检查 / 停止 vLLM 服务：评审 Qwen3-32B、奖励模型 FangZheng-RM、被测模型 Qwen3-4B-Instruct-2507。

两种模式：
  all   —— S1/S2 官方评测与 S4 生成答案、选择题用：评审、奖励模型、被测模型三个服务；
  judge —— S3 重评与 S4 答案评审用：只开评审（两卡张量并行；失败时两卡各一个评审副本）。

显卡分配（按实际显卡自动选择，见 config/sup.json 的 serve 段与补充登记第六节）：
  C. 2 张、每张 ≥76 GiB（RTX PRO 6000 96 GB、RTX 6000D 84 GB、A800/H800 80 GB 等）：
       all   评审独占 0 号卡（显存比例 0.92）；奖励模型（0.50）与被测模型（0.40）共用 1 号卡；
       judge 评审在 0、1 号卡张量并行（0.92）；不行则两卡各开一个评审（端口 8002、8012）。
  A. ≥4 张、每张 ≥44 GB：
       all   评审占 0、1 号卡（张量并行），奖励模型 2 号卡，被测模型 3 号卡；
       judge 两个评审副本：0、1 号卡与 2、3 号卡。
  其他：停止并说明原因。

每个服务按顺序尝试三种启动方式：default → eager（--enforce-eager，不用 CUDA graph）→ triton（再加
VLLM_ATTENTION_BACKEND=TRITON_ATTN）。若 preflight 已在小模型上验证出可用的方式，优先用它。
每个服务就绪后做一次简单问答核对（“1+1”），答错也视为启动失败。实际使用的方式记入 state/servers.json 与
logs/servers_history.jsonl。

用法：python serve.py start all|judge  |  stop  |  status
"""
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from common_sup import CFG, LOGS, WORK, log

STATE = WORK / 'state'
PIDS = STATE / 'servers.json'
HIST = LOGS / 'servers_history.jsonl'
MODELS = Path(os.environ.get('SUP_MODELS') or str(WORK / 'models'))
SV = CFG['serve']
VARIANTS = {
    'default': ([], {}),
    'eager': (['--enforce-eager'], {}),
    'triton': (['--enforce-eager'], {'VLLM_ATTENTION_BACKEND': 'TRITON_ATTN'}),
}


# ----------------------------------------------------------------------------- 显卡
def gpus():
    out = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.total,memory.used,name', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True, check=True).stdout
    res = []
    for line in out.strip().splitlines():
        i, tot, used, name = [x.strip() for x in line.split(',', 3)]
        res.append({'index': int(i), 'total_mib': int(float(tot)), 'used_mib': int(float(used)), 'name': name})
    return res


def describe(gs):
    return ', '.join('{}号 {} {:.1f}GiB'.format(g['index'], g['name'], g['total_mib'] / 1024) for g in gs)


def layout(gs=None):
    """返回 (方案名, 显卡数)。"""
    gs = gs if gs is not None else gpus()
    gib = [g['total_mib'] / 1024 for g in gs]
    if len(gs) >= 4 and min(gib[:4]) >= SV['min_gib_4card']:
        return 'A', 4
    if len(gs) >= 2 and min(gib[:2]) >= SV['min_gib_2card']:
        return 'C', 2
    raise SystemExit('显卡不满足要求：需要 2 张 ≥{} GiB 的卡（如 RTX PRO 6000 96 GB、RTX 6000D 84 GB、A800 80 GB），'
                     '或 4 张 ≥{} GB 的卡；当前为 {}'.format(SV['min_gib_2card'], SV['min_gib_4card'], describe(gs)))


def spec(name, role, devs, util, port=None):
    m = CFG['models'][role]
    return {'name': name, 'role': role, 'model': str(MODELS / m['dir']), 'served': m['served_name'],
            'port': int(port or m['port']), 'devices': list(devs), 'util': float(util),
            'max_len': int(m['max_model_len']), 'reasoning_parser': m.get('reasoning_parser')}


def plans(mode, gs=None):
    """返回 (方案名, [候选部署，每个是若干服务的列表]，按优先顺序)。"""
    name, _ = layout(gs)
    pre = preflight_info()
    j2 = SV['judge2_port']
    if name == 'A':
        u = SV['util_4card']
        if mode == 'all':
            return name, [[spec('judge', 'judge', [0, 1], u), spec('rm', 'rm', [2], u), spec('gen', 'gen', [3], u)]]
        return name, [[spec('judge', 'judge', [0, 1], u), spec('judge2', 'judge', [2, 3], u, j2)]]
    uj = SV['util_judge_solo']
    if mode == 'all':
        return name, [[spec('judge', 'judge', [0], uj), spec('rm', 'rm', [1], SV['util_rm_shared']),
                       spec('gen', 'gen', [1], SV['util_gen_shared'])]]
    tp2 = [spec('judge', 'judge', [0, 1], uj)]
    dp2 = [spec('judge', 'judge', [0], uj), spec('judge2', 'judge', [1], uj, j2)]
    if pre.get('tp2') is False:
        return name, [dp2]
    return name, [tp2, dp2]


def preflight_info():
    p = STATE / 'vllm_variants.json'
    if p.exists():
        try:
            return json.loads(p.read_text(encoding='utf-8'))
        except Exception:  # noqa: BLE001
            return {}
    return {}


LEARNED = STATE / 'learned_variants.json'


def learned():
    try:
        return json.loads(LEARNED.read_text(encoding='utf-8'))
    except Exception:  # noqa: BLE001
        return {}


def variant_order(tp, role=None):
    """启动方式的尝试顺序：本次运行中该服务已成功用过的方式 → 预检得出的方式 → 其余。"""
    pre = preflight_info()
    firsts = [learned().get('{}_tp{}'.format(role, tp)), pre.get('tp2_variant' if tp > 1 else 'tp1_variant')]
    order = list(SV['variants'])
    for first in reversed([f for f in firsts if f in order]):
        order.remove(first)
        order.insert(0, first)
    if tp > 1 and pre.get('tp2_variant') in order:
        # 两卡并行的问题多在 NCCL 通信，换启动方式无济于事：只试预检通过的那一种，失败就改用备用部署
        order = [firsts[0] if firsts[0] in order else pre['tp2_variant']]
    return order


# ----------------------------------------------------------------------------- 单个服务
def http_json(url, body=None, timeout=10):
    data = None if body is None else json.dumps(body).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        txt = r.read().decode('utf-8')
        return r.status, (json.loads(txt) if txt.strip().startswith(('{', '[')) else txt)


def healthy_port(port, timeout=5):
    try:
        st, _ = http_json('http://127.0.0.1:{}/health'.format(port), timeout=timeout)
        return st == 200
    except Exception:  # noqa: BLE001
        return False


def healthy(role, timeout=5):
    """兼容旧接口：按角色检查（judge 检查全部评审副本）。"""
    ports = [v['port'] for v in load_state().get('instances', {}).values() if v['role'] == role] or \
            [CFG['models'][role]['port']]
    return all(healthy_port(p, timeout) for p in ports)


def launch(s, variant, log_name=None):
    args, env_extra = VARIANTS[variant]
    cmd = [sys.executable, '-m', 'vllm.entrypoints.openai.api_server', '--model', s['model'],
           '--served-model-name', s['served'], '--host', '127.0.0.1', '--port', str(s['port']),
           '--dtype', CFG['dtype'], '--tensor-parallel-size', str(len(s['devices'])),
           '--gpu-memory-utilization', str(s['util']), '--max-model-len', str(s['max_len']),
           '--seed', str(CFG['vllm_seed'])]
    if s.get('reasoning_parser'):
        cmd += ['--reasoning-parser', s['reasoning_parser']]
    if len(s['devices']) > 1:
        cmd += ['--disable-custom-all-reduce']        # 无 NVLink 的 PCIe 卡：只用 NCCL，避免自定义 all-reduce 的兼容问题
    cmd += args
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(str(d) for d in s['devices']), **env_extra)
    lf_path = LOGS / 'vllm_{}.log'.format(log_name or s['name'])
    lf = open(lf_path, 'a', encoding='utf-8')
    lf.write('\n===== {} 启动（方式 {}，环境 {}）：{}\n'.format(time.strftime('%Y-%m-%d %H:%M:%S'), variant, env_extra,
                                                            ' '.join(cmd)))
    lf.flush()
    p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    lf.close()
    return p, cmd, env_extra, lf_path


def wait_ready(p, port, limit):
    t0 = time.time()
    while time.time() - t0 < limit:
        if p.poll() is not None:
            return False
        if healthy_port(port):
            return True
        time.sleep(5)
    return False


def _group_alive(pid):
    try:                                   # 若是本进程启动的子进程，先回收（否则僵尸进程会被当作仍在运行）
        os.waitpid(pid, os.WNOHANG)
    except (ChildProcessError, OSError):
        pass
    try:
        os.killpg(pid, 0)
        return True
    except OSError:
        return False


def kill_group(pid, grace=60):
    try:
        os.killpg(pid, signal.SIGTERM)
    except OSError:
        return
    t0 = time.time()
    while time.time() - t0 < grace:
        if not _group_alive(pid):
            return
        time.sleep(1)
    try:
        os.killpg(pid, signal.SIGKILL)
    except OSError:
        pass
    time.sleep(3)


def tail(path, n=25):
    """日志中最近一次启动以来的最后 n 行。"""
    try:
        lines = Path(path).read_text(encoding='utf-8', errors='replace').splitlines()
        k = max([i for i, x in enumerate(lines) if x.startswith('===== ')] or [0])
        return '\n'.join(lines[k:][-n:])
    except Exception:  # noqa: BLE001
        return ''


def sanity(s, timeout=600):
    """一次简单问答核对：评审与被测模型须答出 2；奖励模型须有非空输出。返回 (是否通过, 说明)。"""
    q = '1+1等于几？只回答一个阿拉伯数字。'
    body = {'model': s['served'], 'messages': [{'role': 'user', 'content': q}], 'temperature': 0.0,
            'max_tokens': 1024 if s['role'] != 'gen' else 16}
    if s.get('reasoning_parser'):
        body['chat_template_kwargs'] = {'enable_thinking': False}
    try:
        st, r = http_json('http://127.0.0.1:{}/v1/chat/completions'.format(s['port']), body, timeout=timeout)
    except Exception as e:  # noqa: BLE001
        return False, '请求失败：{}'.format(repr(e)[:300])
    try:
        msg = r['choices'][0]['message']
    except Exception:  # noqa: BLE001
        return False, '返回格式异常：{}'.format(str(r)[:300])
    content = (msg.get('content') or '').strip()
    reasoning = (msg.get('reasoning_content') or '').strip()
    if s['role'] == 'rm':
        return bool(content or reasoning), '输出：{}'.format((content or reasoning)[:80])
    return '2' in content, '输出：{}'.format(content[:80])


def wait_mem_release(devs, baseline, limit=180):
    """等指定显卡的显存占用回落到 baseline（MiB，按卡）附近。"""
    t0 = time.time()
    while time.time() - t0 < limit:
        try:
            used = {g['index']: g['used_mib'] for g in gpus()}
        except Exception:  # noqa: BLE001
            return True
        if all(used.get(d, 0) <= baseline.get(d, 0) + 1500 for d in devs):
            return True
        time.sleep(5)
    return False


def start_instance(s, record):
    """按启动方式依次尝试，直到服务就绪且通过问答核对。成功返回实例信息，否则 None。"""
    limit = int(SV['start_timeout_sec'].get(s['role'], 900))
    for variant in variant_order(len(s['devices']), s['role']):
        base = {g['index']: g['used_mib'] for g in gpus()}
        t0 = time.time()
        p, cmd, env_extra, lf = launch(s, variant)
        log('启动 {}（卡 {}，显存比例 {}，端口 {}，方式 {}），等待就绪……'.format(s['name'], s['devices'], s['util'], s['port'], variant))
        ok = wait_ready(p, s['port'], limit)
        why = '' if ok else ('进程已退出' if p.poll() is not None else '超时 {} 秒未就绪'.format(limit))
        if ok:
            ok, why = sanity(s)
        if ok:
            info = dict(s, pid=p.pid, variant=variant, cmd=cmd, env_extra=env_extra,
                        startup_sec=round(time.time() - t0, 1), started=time.strftime('%Y-%m-%d %H:%M:%S'))
            log('{} 已就绪（{:.0f} 秒，方式 {}；核对{}）'.format(s['name'], time.time() - t0, variant, why))
            lv = learned()
            lv['{}_tp{}'.format(s['role'], len(s['devices']))] = variant
            STATE.mkdir(parents=True, exist_ok=True)
            LEARNED.write_text(json.dumps(lv, ensure_ascii=False), encoding='utf-8')
            return info
        log('{} 以方式 {} 启动失败：{}。日志最后几行：\n{}'.format(s['name'], variant, why, tail(lf)))
        record.append({'name': s['name'], 'variant': variant, 'error': why})
        kill_group(p.pid)
        wait_mem_release(s['devices'], base)
    return None


# ----------------------------------------------------------------------------- 状态
def load_state():
    if PIDS.exists():
        try:
            return json.loads(PIDS.read_text(encoding='utf-8'))
        except Exception:  # noqa: BLE001
            return {}
    return {}


def save_state(st):
    STATE.mkdir(parents=True, exist_ok=True)
    tmp = PIDS.with_suffix('.tmp')
    tmp.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding='utf-8')
    os.replace(tmp, PIDS)


def history(rec):
    with open(HIST, 'a', encoding='utf-8') as f:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')


def all_healthy(st):
    inst = st.get('instances') or {}
    return bool(inst) and all(healthy_port(v['port']) for v in inst.values())


def stale_vllm_pids():
    """本工作目录模型的 vLLM 服务进程（例如上次异常退出后残留的）。"""
    out = []
    me = os.getpid()
    for d in Path('/proc').iterdir():
        if not d.name.isdigit() or int(d.name) == me:
            continue
        try:
            cmd = (d / 'cmdline').read_bytes().replace(b'\0', b' ').decode('utf-8', 'replace')
        except Exception:  # noqa: BLE001
            continue
        if ('vllm.entrypoints.openai.api_server' in cmd and str(MODELS) in cmd) or cmd.startswith('VLLM::'):
            out.append(int(d.name))
    return out


def stop(quiet=False):
    st = load_state()
    pids = [v['pid'] for v in (st.get('instances') or {}).values()]
    pids += [p for p in stale_vllm_pids() if p not in pids]
    for pid in pids:
        try:
            pg = os.getpgid(pid)
        except OSError:
            continue
        kill_group(pg)
    if PIDS.exists():
        PIDS.unlink()
    # 等显存释放（下一个服务启动时 vLLM 要求空闲显存不少于“比例 × 总显存”）
    t0 = time.time()
    while time.time() - t0 < 180:
        try:
            gs = gpus()
        except Exception:  # noqa: BLE001
            break
        if all(g['used_mib'] < 2048 for g in gs):
            break
        time.sleep(5)
    else:
        log('注意：停止服务 3 分钟后显存仍未完全释放：{}'.format(
            ', '.join('{}号 已用 {:.1f}GiB'.format(g['index'], g['used_mib'] / 1024) for g in gpus())))
    if pids and not quiet:
        log('已停止全部 vLLM 服务')


def start(mode):
    if mode not in ('all', 'judge'):
        raise SystemExit('未知模式 {}'.format(mode))
    st = load_state()
    if st.get('mode') == mode and all_healthy(st):
        log('{} 模式的服务已在运行（{}）'.format(mode, ', '.join('{}:{}'.format(k, v['variant'])
                                                         for k, v in st['instances'].items())))
        return st
    stop(quiet=True)
    gs = gpus()
    name, cands = plans(mode, gs)
    log('显卡：{}；分配方案 {}，模式 {}'.format(describe(gs), name, mode))
    busy = [g for g in gs if g['used_mib'] > 4096]
    if busy:
        raise SystemExit('显卡上已有其他程序占用显存：{}。请先结束它们（可运行 bash stop.sh）'.format(
            ', '.join('{}号 已用 {:.1f}GiB'.format(g['index'], g['used_mib'] / 1024) for g in busy)))
    fails = []
    lv = learned()
    for k, cand in enumerate(cands):
        if lv.get('skip_{}_{}'.format(mode, k)) and k + 1 < len(cands):
            log('部署 {} 在本次运行中已失败过，直接改用备用部署'.format(k + 1))
            continue
        st = {'mode': mode, 'layout': name, 'deployment': k, 'gpus': gs, 'instances': {},
              'started': time.strftime('%Y-%m-%d %H:%M:%S')}
        ok = True
        for s in cand:              # 依次启动（同卡共用时，后启动的按剩余显存分配）
            info = start_instance(s, fails)
            if info is None:
                ok = False
                break
            st['instances'][s['name']] = info
            save_state(st)
        if ok:
            st['failed_attempts'] = fails
            save_state(st)
            history({k2: v for k2, v in st.items() if k2 != 'gpus'})
            if k > 0:
                log('注意：采用了备用部署（{}）'.format(', '.join('{} 卡{}'.format(n, v['devices'])
                                                        for n, v in st['instances'].items())))
            return st
        stop(quiet=True)
        if k + 1 < len(cands):
            log('部署 {} 失败，改用备用部署'.format(k + 1))
            lv = learned()
            lv['skip_{}_{}'.format(mode, k)] = True
            LEARNED.write_text(json.dumps(lv, ensure_ascii=False), encoding='utf-8')
    history({'mode': mode, 'layout': name, 'failed': True, 'attempts': fails, 'time': time.strftime('%Y-%m-%d %H:%M:%S')})
    raise SystemExit('{} 模式的服务启动失败（已尝试：{}）。请把 bash status.sh 的输出和 {} 下 vllm_*.log 的最后几十行发给 Claude。'.format(
        mode, '；'.join('{} {}：{}'.format(f['name'], f['variant'], f['error']) for f in fails), LOGS))


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'status'
    if cmd == 'start':
        start(sys.argv[2] if len(sys.argv) > 2 else 'all')
    elif cmd == 'stop':
        stop()
    else:
        st = load_state()
        print(json.dumps({'mode': st.get('mode'), 'instances': {k: {'port': v['port'], 'devices': v['devices'],
                                                                     'variant': v['variant'],
                                                                     'healthy': healthy_port(v['port'])}
                                                                 for k, v in (st.get('instances') or {}).items()}},
                         ensure_ascii=False))
