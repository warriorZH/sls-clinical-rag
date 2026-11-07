#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S4 的 GPU 调度：每张卡同一时间只跑一个作业；作业失败重试 1 次。

    python s4_orchestrate.py smoke | train | score | select
"""
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

from common_sup import LOGS, log, read_json
from s4_common import SCORE_DIR, TRAIN_DIR, score_models, train_runs

HERE = Path(__file__).resolve().parent


def n_gpus():
    out = subprocess.run(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'], capture_output=True, text=True,
                         check=True).stdout
    return len([x for x in out.split() if x.strip()])


def run_jobs(jobs, label):
    """jobs: [(名称, 命令参数列表, 是否已完成的判断函数)]"""
    todo = [j for j in jobs if not j[2]()]
    log('S4 {}：共 {} 个作业，待做 {}'.format(label, len(jobs), len(todo)), 's4')
    if not todo:
        return
    gpus = queue.Queue()
    for g in range(n_gpus()):
        gpus.put(g)
    failed = []
    lock = threading.Lock()

    def work(job):
        name, args, done = job
        for attempt in (1, 2):
            g = gpus.get()
            try:
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTHONUNBUFFERED='1')
                lf = open(LOGS / 's4_{}.log'.format(name), 'a', encoding='utf-8')
                t0 = time.time()
                log('S4 开始 {}（{} 号卡，第 {} 次）'.format(name, g, attempt), 's4')
                rc = subprocess.run([sys.executable] + args, cwd=str(HERE), env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
                lf.close()
            finally:
                gpus.put(g)
            if rc == 0 and done():
                log('S4 完成 {}，{:.1f} 分钟'.format(name, (time.time() - t0) / 60), 's4')
                return
            log('S4 {} 失败（返回码 {}），日志：{}'.format(name, rc, LOGS / 's4_{}.log'.format(name)), 's4')
        with lock:
            failed.append(name)

    threads = []
    pending = list(todo)
    # 简单的工作池：线程数 = 卡数
    def worker():
        while True:
            with lock:
                if not pending:
                    return
                job = pending.pop(0)
            work(job)
    for _ in range(n_gpus()):
        t = threading.Thread(target=worker)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    if failed:
        raise SystemExit('S4 {} 有作业失败：{}'.format(label, ', '.join(failed)))


def main():
    stage = sys.argv[1]
    if stage == 'smoke':
        jobs = [('train_q06_smoke', ['s4_train.py', '--model', 'q06', '--objective', 'support', '--seed', '42', '--smoke'],
                 lambda: (TRAIN_DIR / 'q06_support_s42_smoke' / 'DONE.json').exists()),
                ('train_q4b_smoke', ['s4_train.py', '--model', 'q4b', '--objective', 'support', '--seed', '42', '--smoke'],
                 lambda: (TRAIN_DIR / 'q4b_support_s42_smoke' / 'DONE.json').exists()),
                ('score_smoke', ['s4_score.py', '--name', 'q06_smoke', '--smoke'],
                 lambda: (SCORE_DIR / 'smoke' / 'DONE.json').exists())]
        run_jobs(jobs, '冒烟测试')
        est = {}
        for key in ('q06', 'q4b'):
            try:
                est[key] = float(read_json(TRAIN_DIR / '{}_support_s42_smoke'.format(key) / 'DONE.json')['estimated_hours_full_run'])
            except Exception:  # noqa: BLE001
                est[key] = None
        if all(v is not None for v in est.values()):
            n = max(1, n_gpus())
            runs = len(train_runs()) // 2
            log('S4 冒烟测试通过。按试跑速度估计：0.6B 每次约 {:.1f} 小时，4B 每次约 {:.1f} 小时（3 轮上限；早停会更短）；'
                '12 次微调在 {} 张卡上约 {:.1f} 小时'.format(est['q06'], est['q4b'], n,
                                                     runs * (est['q06'] + est['q4b']) / n), 's4')
    elif stage == 'train':
        jobs = [('train_' + name, ['s4_train.py', '--model', key, '--objective', obj, '--seed', str(seed)],
                 (lambda n=name: (TRAIN_DIR / n / 'DONE.json').exists())) for key, obj, seed, name in train_runs()]
        run_jobs(jobs, '微调')
    elif stage == 'score':
        jobs = [('score_' + name, ['s4_score.py', '--name', name],
                 (lambda n=name: (SCORE_DIR / n / 'DONE.json').exists())) for name, _, _ in score_models()]
        run_jobs(jobs, '打分')
    else:
        raise SystemExit('未知阶段 {}'.format(stage))


if __name__ == '__main__':
    main()
