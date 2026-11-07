#!/usr/bin/env bash
# 查看补充分析进度
source "$(cd "$(dirname "$0")" && pwd)/tools/env.sh"
echo "工作目录：$SUP_WORK"
if pgrep -f "run_all.sh --fg" >/dev/null 2>&1; then echo "状态：正在运行（当前步骤：$(cat "$STATE/CURRENT" 2>/dev/null)）"
elif [ -f "$STATE/ALL_DONE" ]; then echo "状态：全部完成。结果包：$(ls -t "$SUP_WORK"/results_sup_*.tar.gz 2>/dev/null | head -1)"
elif [ -f "$STATE/FAILED" ]; then echo "状态：已停止，出错：$(cat "$STATE/FAILED")。把下面的内容发给 Claude；修好后再次运行 bash run_all.sh 会从断点继续。"
elif [ -f "$STATE/STOPPED" ]; then echo "状态：已手动停止（运行 bash run_all.sh 从断点继续）"
else echo "状态：未在运行（运行 bash run_all.sh 启动或续跑）"; fi
echo "已完成的步骤：$(cd "$STATE" && ls -tr *.done 2>/dev/null | sed 's/.done//' | tr '\n' ' ')"
python3 - <<PYEOF
import json, os, glob
w = os.environ['SUP_WORK']
st = w + '/state'
cfg = json.load(open('$HERE/config/sup.json'))
def n(p):
    try:
        return sum(1 for l in open(p, encoding='utf-8') if l.strip())
    except FileNotFoundError:
        return 0
def uniq(p, key='aid', ok=None):
    s = set()
    try:
        for l in open(p, encoding='utf-8'):
            try:
                r = json.loads(l)
            except Exception:
                continue
            if ok is None or ok(r):
                s.add(r.get(key) if isinstance(key, str) else tuple(r.get(k) for k in key))
    except FileNotFoundError:
        pass
    return len(s)
try:
    sv = json.load(open(st + '/servers.json'))
    print('vLLM 服务：{} 模式，{}'.format(sv.get('mode'), '；'.join('{} 卡{} 方式{}'.format(k, v['devices'], v['variant'])
                                                             for k, v in sv.get('instances', {}).items())))
except Exception:
    print('vLLM 服务：未运行')
for f in ('preflight_gpu', 'preflight_vllm', 'preflight_api_all', 'preflight_api_judge'):
    try:
        r = json.load(open('{}/{}.json'.format(st, f)))
        print('预检 {}：{}{}'.format(f.replace('preflight_', ''), '未通过 ' + '；'.join(r['fatal']) if r.get('fatal') else '通过',
                                  '（注意：' + '；'.join(r['warnings']) + '）' if r.get('warnings') else ''))
    except FileNotFoundError:
        pass
tot = len(cfg['conditions']) * 300
print('官方评测：{}/{} 个“条件×病例”'.format(uniq(w + '/results/official/official_results.jsonl', ('cond', 'qid')), tot))
if cfg['rejudge']['enabled']:
    print('32B 重评：{}/18,290 条答案'.format(uniq(w + '/results/rejudge/kpr_qwen3_32b.jsonl', ok=lambda r: r.get('valid'))))
if cfg.get('s4', {}).get('enabled'):
    tr = len(glob.glob(w + '/s4/train/q*_s[0-9]*/DONE.json')) - len(glob.glob(w + '/s4/train/*_smoke/DONE.json'))
    sc = len([p for p in glob.glob(w + '/s4/scores/*/DONE.json') if '/smoke/' not in p])
    ans = sum(n(p) for p in glob.glob(w + '/results/s4/answers/*.jsonl'))
    print('S4：微调 {}/12，打分 {}/14，生成答案 {}/约 10,000，选择题 {}/4,200，32B 评审 {}'.format(
        tr, sc, ans, uniq(w + '/results/s4/mcq_results.jsonl', ('cond', 'qid')),
        uniq(w + '/results/s4/kpr_qwen3_32b_s4.jsonl', ok=lambda r: r.get('valid'))))
PYEOF
echo "---- 最近日志 ----"
tail -n 8 "$SUP_WORK/logs/run_all.log" 2>/dev/null
for f in preflight s4 official rejudge pipeline; do [ -f "$SUP_WORK/logs/$f.log" ] && tail -n 1 "$SUP_WORK/logs/$f.log"; done
echo "---- 显卡 ----"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null
echo "---- 磁盘 ----"
df -h "$SUP_WORK" | tail -1
