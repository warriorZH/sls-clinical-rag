#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""运行前检查：尽早发现问题，避免在租用的机器上反复失败。三个阶段由 run_all.sh 自动调用：

  python preflight.py gpu         装好环境后、下载大模型之前（约 2 分钟）：
                                  显卡与分配方案、PyTorch 对本卡架构（Blackwell 为 sm_120）的支持、bf16 矩阵乘与注意力
                                  的数值、Triton 与 torch.compile（需要 C 编译器与 Python 头文件）、双卡 NCCL 通信
                                  （失败时改试 NCCL_P2P_DISABLE=1 并记下）、vLLM 导入、磁盘、共享内存。
  python preflight.py vllm        下载小模型之后（约 5–15 分钟，首次含内核编译）：用 Qwen3-4B-Instruct-2507 实际启动
                                  vLLM：单卡、同卡两个服务（与正式部署相同的显存比例）、两卡张量并行；
                                  依次试 default / eager / triton 三种方式，记下可用的方式（state/vllm_variants.json）。
  python preflight.py api all|judge   正式服务启动后（约 2–5 分钟）：评审在思考模式下（官方评测的用法，含流式）
                                  输出 JSON 且不含 <think>；非思考模式（S3 重评的用法）输出 JSON；奖励模型按官方
                                  方证匹配提示词返回“匹配分数”；被测模型正常作答。

任何一项不通过：退出码 2，run_all.sh 停下并在 status.sh 中显示原因（此时尚未开始长时间运行）。
结果写入 state/preflight_*.json 与 logs/。
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from common_sup import CFG, LOGS, OFFICIAL, WORK, judge_ports, log, write_json

STATE = WORK / 'state'
HERE = Path(__file__).resolve().parent


class Report:
    def __init__(self, name):
        self.name = name
        self.rep = {'stage': name, 'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'checks': {}}
        self.fatal = []
        self.warn = []

    def add(self, key, ok, detail='', fatal=True):
        self.rep['checks'][key] = {'ok': bool(ok), 'detail': detail}
        mark = '通过' if ok else ('未通过' if fatal else '注意')
        log('[预检 {}] {}：{} {}'.format(self.name, key, mark, detail), 'preflight')
        if not ok:
            (self.fatal if fatal else self.warn).append('{}（{}）'.format(key, detail))
        return ok

    def finish(self, path):
        self.rep['fatal'] = self.fatal
        self.rep['warnings'] = self.warn
        write_json(path, self.rep)
        if self.fatal:
            log('[预检 {}] 未通过：{}'.format(self.name, '；'.join(self.fatal)), 'preflight')
            print('预检未通过：' + '；'.join(self.fatal))
            sys.exit(2)
        log('[预检 {}] 全部通过{}'.format(self.name, '（注意：{}）'.format('；'.join(self.warn)) if self.warn else ''),
            'preflight')


# ----------------------------------------------------------------------------- gpu
NCCL_CODE = r'''
import os, sys, torch, torch.distributed as dist, torch.multiprocessing as mp
def run(rank, port):
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', init_method='tcp://127.0.0.1:%d' % port, rank=rank, world_size=2)
    x = torch.full((16 * 1024 * 1024,), float(rank + 1), device='cuda')
    for _ in range(3):
        y = x.clone(); dist.all_reduce(y)
    torch.cuda.synchronize()
    assert abs(float(y[0]) - 3.0) < 1e-6 and abs(float(y[-1]) - 3.0) < 1e-6, float(y[0])
    dist.barrier(); dist.destroy_process_group()
if __name__ == '__main__':
    mp.spawn(run, args=(int(sys.argv[1]),), nprocs=2, join=True)
    print('NCCL_OK')
'''


def nccl_test(extra_env, timeout=240):
    p = WORK / 'tmp_nccl_test.py'
    p.write_text(NCCL_CODE, encoding='utf-8')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1', **extra_env)
    try:
        r = subprocess.run([sys.executable, str(p), '29517'], capture_output=True, text=True, timeout=timeout, env=env)
        ok = r.returncode == 0 and 'NCCL_OK' in r.stdout
        return ok, (r.stdout + r.stderr)[-600:]
    except subprocess.TimeoutExpired:
        subprocess.run(['pkill', '-f', str(p)], capture_output=True)
        return False, '超时 {} 秒（多半卡在 P2P 通信）'.format(timeout)
    finally:
        p.unlink(missing_ok=True)


def stage_gpu():
    R = Report('gpu')
    import serve
    gs = serve.gpus()
    try:
        name, ncard = serve.layout(gs)
        R.add('显卡与分配方案', True, '{}；方案 {}'.format(serve.describe(gs), name))
    except SystemExit as e:
        R.add('显卡与分配方案', False, str(e))
        R.finish(STATE / 'preflight_gpu.json')
    R.rep['layout'] = name
    busy = [g for g in gs if g['used_mib'] > 4096]
    R.add('显卡空闲', not busy, '已占用：{}'.format(', '.join('{}号 {:.1f}GiB'.format(g['index'], g['used_mib'] / 1024)
                                                         for g in busy)) if busy else '')

    import torch
    R.add('PyTorch 看到显卡', torch.cuda.is_available() and torch.cuda.device_count() >= ncard,
          'torch {}，CUDA {}，{} 张'.format(torch.__version__, torch.version.cuda, torch.cuda.device_count()))
    arch = torch.cuda.get_arch_list()
    for i in range(min(torch.cuda.device_count(), ncard)):
        cap = torch.cuda.get_device_capability(i)
        sm = 'sm_{}{}'.format(*cap)
        R.add('{} 号卡架构 {} 受支持'.format(i, sm), sm in arch, '本 PyTorch 支持：{}'.format(' '.join(arch)))
    R.rep['capabilities'] = [list(torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())]

    # 数值：bf16 矩阵乘、注意力
    import torch.nn.functional as F
    for i in range(min(torch.cuda.device_count(), ncard)):
        try:
            d = 'cuda:{}'.format(i)
            g = torch.Generator(device='cpu').manual_seed(0)
            a = torch.randn(1024, 1024, generator=g).to(d, torch.bfloat16)
            c = (a @ a.T).float()
            ref = a.float() @ a.float().T
            err = float((c - ref).abs().max() / ref.abs().max())
            q = torch.randn(1, 8, 256, 128, generator=g).to(d, torch.bfloat16)
            o = F.scaled_dot_product_attention(q, q, q, is_causal=True).float()
            oref = F.scaled_dot_product_attention(q.float(), q.float(), q.float(), is_causal=True)
            aerr = float((o - oref).abs().max())
            R.add('{} 号卡 bf16 计算'.format(i), err < 2e-2 and aerr < 5e-2 and torch.isfinite(o).all().item(),
                  '矩阵乘相对误差 {:.1e}，注意力误差 {:.1e}'.format(err, aerr))
        except Exception as e:  # noqa: BLE001
            R.add('{} 号卡 bf16 计算'.format(i), False, repr(e)[:300])

    # Triton 与 torch.compile（vLLM 默认用它们；需要 C 编译器与 Python 头文件）
    R.add('C 编译器', bool(shutil.which('gcc') or shutil.which('cc')), shutil.which('gcc') or shutil.which('cc') or '找不到 gcc')
    try:
        import triton
        import triton.language as tl

        @triton.jit
        def _add(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            m = offs < n
            tl.store(o_ptr + offs, tl.load(x_ptr + offs, mask=m) + tl.load(y_ptr + offs, mask=m), mask=m)

        x = torch.arange(4096, device='cuda:0', dtype=torch.float32)
        o = torch.empty_like(x)
        _add[(4,)](x, x, o, 4096, BLOCK=1024)
        R.add('Triton 内核编译', bool(torch.equal(o, 2 * x)), 'triton {}'.format(triton.__version__))
    except Exception as e:  # noqa: BLE001
        R.add('Triton 内核编译', False, repr(e)[:400])
    try:
        f = torch.compile(lambda t: torch.sin(t) * 2 + 1)
        t = torch.randn(4096, device='cuda:0')
        R.add('torch.compile', bool(torch.allclose(f(t), torch.sin(t) * 2 + 1, atol=1e-5)), '')
    except Exception as e:  # noqa: BLE001
        R.add('torch.compile', False, repr(e)[:400])

    # 双卡通信（judge 模式两卡张量并行要用；不通时自动改用两个评审副本，不影响结果，只是慢一些）
    nccl_env = {}
    ok, out = nccl_test({})
    if not ok:
        log('NCCL 默认设置不通：{}；改试 NCCL_P2P_DISABLE=1'.format(out[-200:]), 'preflight')
        ok, out = nccl_test({'NCCL_P2P_DISABLE': '1'})
        if ok:
            nccl_env = {'NCCL_P2P_DISABLE': '1'}
    (STATE / 'nccl.env').write_text(''.join('{}={}\n'.format(k, v) for k, v in nccl_env.items()), encoding='utf-8')
    R.rep['nccl_ok'], R.rep['nccl_env'] = ok, nccl_env
    R.add('双卡 NCCL 通信', ok, '设置 {}'.format(nccl_env or '默认') if ok else out[-300:], fatal=(name == 'A'))

    try:
        import vllm
        import vllm._C  # noqa: F401  载入 vLLM 的 CUDA 扩展
        R.add('vLLM 导入', vllm.__version__ == '0.11.0', 'vllm {}'.format(vllm.__version__))
    except Exception as e:  # noqa: BLE001
        R.add('vLLM 导入', False, repr(e)[:400])

    # 磁盘：还要下载的模型 + 余量 30 GB
    need_gb = 30.0
    reg = {}
    regp = Path(os.environ.get('SUP_MODELS') or (WORK / 'models')) / 'registry.json'
    if regp.exists():
        reg = json.loads(regp.read_text(encoding='utf-8'))
    sizes = {'judge': 65.6, 'rm': 29.6, 'gen': 8.1, 'q06': 1.2, 'q4b': 8.1}
    for k, v in sizes.items():
        if k not in reg and (k not in ('q06', 'q4b') or CFG.get('s4', {}).get('enabled')):
            need_gb += v
    free_gb = shutil.disk_usage(str(WORK)).free / 1e9
    R.add('磁盘空间', free_gb >= need_gb, '剩余 {:.0f} GB，需要约 {:.0f} GB'.format(free_gb, need_gb))
    try:
        shm = shutil.disk_usage('/dev/shm').total / 2 ** 30
        R.add('共享内存 /dev/shm', shm >= 1, '{:.1f} GiB'.format(shm), fatal=False)
    except Exception:  # noqa: BLE001
        pass
    try:
        mem = int([x for x in open('/proc/meminfo') if x.startswith('MemTotal')][0].split()[1]) / 2 ** 20
        R.add('内存', mem >= 60, '{:.0f} GiB'.format(mem), fatal=False)
    except Exception:  # noqa: BLE001
        pass
    R.finish(STATE / 'preflight_gpu.json')


# ----------------------------------------------------------------------------- vllm
def stage_vllm():
    import serve
    R = Report('vllm')
    gs = serve.gpus()
    name, _ = serve.layout(gs)
    pre = json.loads((STATE / 'preflight_gpu.json').read_text(encoding='utf-8')) if (STATE / 'preflight_gpu.json').exists() else {}
    serve.stop(quiet=True)
    m = CFG['models']['gen']
    model = serve.MODELS / m['dir']
    if not model.exists():
        R.add('小模型已下载', False, str(model))
        R.finish(STATE / 'preflight_vllm.json')

    def sp(port, devs, util):
        return {'name': 'preflight', 'role': 'gen', 'model': str(model), 'served': m['served_name'], 'port': port,
                'devices': devs, 'util': util, 'max_len': 4096, 'reasoning_parser': None}

    def try_up(specs, variant, label, limit=900):
        """依次启动 specs（同一种方式），全部就绪且答对后返回 (True, 用时)；最后全部停止。"""
        procs, t0, why = [], time.time(), ''
        base = {g['index']: g['used_mib'] for g in serve.gpus()}
        try:
            for s in specs:
                p, _, _, lf = serve.launch(s, variant, log_name='preflight_{}'.format(label))
                procs.append(p)
                if not serve.wait_ready(p, s['port'], limit):
                    why = '未就绪：' + serve.tail(lf, 15)
                    return False, why, time.time() - t0
                ok, why = serve.sanity(s)
                if not ok:
                    return False, '问答核对未通过：' + why, time.time() - t0
            return True, why, time.time() - t0
        finally:
            for p in procs:
                serve.kill_group(p.pid)
            serve.wait_mem_release(sorted({d for s in specs for d in s['devices']}), base)

    out = {'layout': name, 'tested_with': m['dir'], 'time': time.strftime('%Y-%m-%d %H:%M:%S')}
    # 1) 单卡
    tp1 = None
    for v in CFG['serve']['variants']:
        ok, why, sec = try_up([sp(8101, [0], 0.40)], v, 'tp1_' + v)
        R.add('单卡启动（方式 {}）'.format(v), ok, '{:.0f} 秒；{}'.format(sec, why[-300:]), fatal=False)
        if ok:
            tp1 = v
            break
    out['tp1_variant'] = tp1
    if tp1 is None:
        R.add('vLLM 单卡可用', False, '三种启动方式都失败，见 logs/vllm_preflight_tp1_*.log')
        write_json(STATE / 'vllm_variants.json', out)
        R.finish(STATE / 'preflight_vllm.json')
    # 2) 同卡两个服务（与正式部署相同的显存比例；vLLM 要求后启动者的空闲显存 ≥ 比例 × 总显存）
    if name == 'C':
        S = CFG['serve']
        ok, why, sec = try_up([sp(8102, [1], S['util_rm_shared']), sp(8103, [1], S['util_gen_shared'])], tp1, 'colocate')
        out['colocate'] = ok
        R.add('同卡两个服务（{} + {}）'.format(S['util_rm_shared'], S['util_gen_shared']), ok,
              '{:.0f} 秒；{}'.format(sec, why[-300:]))
    # 3) 两卡张量并行
    tp2 = None
    if pre.get('nccl_ok', True):
        # 只用单卡可用的方式试一次：两卡并行的问题多在 NCCL 通信，换启动方式无济于事，且卡住时很费时间
        ok, why, sec = try_up([sp(8104, [0, 1], 0.40)], tp1, 'tp2_' + tp1, limit=600)
        R.add('两卡张量并行（方式 {}）'.format(tp1), ok, '{:.0f} 秒；{}'.format(sec, why[-300:]), fatal=False)
        if ok:
            tp2 = tp1
    out['tp2'] = tp2 is not None
    out['tp2_variant'] = tp2
    if name == 'A' and tp2 is None:
        R.add('两卡张量并行', False, '4 卡方案的评审必须两卡并行')
    elif tp2 is None:
        R.add('两卡张量并行', False, 'S3/S4 评审改用两个评审副本（结果相同，速度稍慢）', fatal=False)
    write_json(STATE / 'vllm_variants.json', out)
    R.rep['result'] = out
    R.finish(STATE / 'preflight_vllm.json')


# ----------------------------------------------------------------------------- api
def stage_api(mode):
    import re
    from openai import OpenAI
    R = Report('api_' + mode)
    jm = CFG['models']['judge']['served_name']
    claim = '请判断下面的说法是否正确，并且只输出一个 JSON 对象，格式为 {"正确": true 或 false, "理由": "不超过20字"}。' \
            '说法：麻黄汤由麻黄、桂枝、杏仁、炙甘草四味药组成。'

    def js(text):
        mm = re.search(r'\{[\s\S]*\}', text or '')
        if not mm:
            return None
        try:
            return json.loads(mm.group(0))
        except Exception:  # noqa: BLE001
            return None

    for port in judge_ports():
        c = OpenAI(api_key='EMPTY', base_url='http://127.0.0.1:{}/v1'.format(port), timeout=900, max_retries=0)
        tag = '评审(端口 {})'.format(port)
        try:   # 思考模式（官方评测的默认用法，max_tokens 4096）
            t0 = time.time()
            r = c.chat.completions.create(model=jm, messages=[{'role': 'user', 'content': claim}], temperature=0.0,
                                          max_tokens=4096)
            ch = r.choices[0]
            content = ch.message.content or ''
            rc = getattr(ch.message, 'reasoning_content', None) or ''
            R.add(tag + ' 思考模式', js(content) is not None and '<think>' not in content and ch.finish_reason == 'stop',
                  '思考 {} 字，正文 {}，{:.0f} 秒'.format(len(rc), content[:60].replace('\n', ' '), time.time() - t0))
        except Exception as e:  # noqa: BLE001
            R.add(tag + ' 思考模式', False, repr(e)[:300])
        try:   # 流式（官方幻觉评估用）
            parts = []
            for ev in c.chat.completions.create(model=jm, messages=[{'role': 'user', 'content': claim}], temperature=0.0,
                                                max_tokens=8192, stream=True):
                if ev.choices and ev.choices[0].delta and ev.choices[0].delta.content:
                    parts.append(ev.choices[0].delta.content)
            text = ''.join(parts)
            R.add(tag + ' 流式', js(text) is not None and '<think>' not in text, text[:60].replace('\n', ' '))
        except Exception as e:  # noqa: BLE001
            R.add(tag + ' 流式', False, repr(e)[:300])
        try:   # 非思考模式（S3 重评用法）
            r = c.chat.completions.create(model=jm, messages=[{'role': 'user', 'content': claim}], temperature=0.0,
                                          max_tokens=int(CFG['rejudge']['max_tokens']),
                                          extra_body={'chat_template_kwargs': {'enable_thinking': False}})
            ch = r.choices[0]
            content = ch.message.content or ''
            rc = getattr(ch.message, 'reasoning_content', None) or ''
            R.add(tag + ' 非思考模式', js(content) is not None and len(rc) < 50 and ch.finish_reason == 'stop',
                  '思考 {} 字，正文 {}'.format(len(rc), content[:60].replace('\n', ' ')))
        except Exception as e:  # noqa: BLE001
            R.add(tag + ' 非思考模式', False, repr(e)[:300])

    if mode == 'all':
        g = CFG['models']['gen']
        c = OpenAI(api_key='EMPTY', base_url='http://127.0.0.1:{}/v1'.format(g['port']), timeout=600, max_retries=0)
        try:
            r = c.chat.completions.create(model=g['served_name'], temperature=0.0, max_tokens=512,
                                          messages=[{'role': 'user', 'content': '简述麻黄汤的功效，50字以内。'}])
            content = r.choices[0].message.content or ''
            R.add('被测模型', bool(content.strip()) and '<think>' not in content, content[:60].replace('\n', ' '))
        except Exception as e:  # noqa: BLE001
            R.add('被测模型', False, repr(e)[:300])
        # 奖励模型：官方方证匹配提示词，经官方解析函数取“匹配分数”
        try:
            sys.path.insert(0, str(OFFICIAL))
            from data_loader import TCMDataLoader
            from evaluators import RewardModelEvaluator
            from common_sup import make_client
            case = sorted(TCMDataLoader(str(OFFICIAL / 'TCM_SDT.json')).load_cases(), key=lambda x: int(x['case_id']))[0]
            rmc = CFG['models']['rm']
            ev = RewardModelEvaluator('127.0.0.1', rmc['port'], rmc['served_name'], api_key='EMPTY')
            ev.client = make_client('rm')
            prompt = ev._build_fangzheng_match_prompt(case, '肝热动风证', '- 羚羊角 3g\n- 钩藤 9g\n- 桑叶 6g\n- 菊花 9g\n'
                                                                          '- 生地黄 12g\n- 白芍 9g\n- 川贝母 6g\n- 竹茹 6g\n'
                                                                          '- 茯神 9g\n- 甘草 3g')
            res = ev._call_reward_model_json(prompt)
            score = res.get('匹配分数')
            R.add('奖励模型（方证匹配）', isinstance(score, (int, float)) and 0 <= float(score) <= 100, json.dumps(res, ensure_ascii=False)[:120])
        except Exception as e:  # noqa: BLE001
            R.add('奖励模型（方证匹配）', False, repr(e)[:300])
    R.finish(STATE / 'preflight_api_{}.json'.format(mode))


if __name__ == '__main__':
    STATE.mkdir(parents=True, exist_ok=True)
    what = sys.argv[1] if len(sys.argv) > 1 else ''
    if what == 'gpu':
        stage_gpu()
    elif what == 'vllm':
        stage_vllm()
    elif what == 'api':
        stage_api(sys.argv[2] if len(sys.argv) > 2 else 'all')
    else:
        raise SystemExit('用法：python preflight.py gpu | vllm | api all|judge')
