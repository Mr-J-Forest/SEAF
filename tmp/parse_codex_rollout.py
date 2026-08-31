"""Parse a Codex rollout jsonl (event_msg/response_item layout) into a readable transcript."""
import json
import sys
import os

path = sys.argv[1]
out = sys.argv[2]

users = []
finals = []
goals = []
toolcalls = []
compactions = []
first_ts = None
last_ts = None

with open(path, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        ts = r.get("timestamp", "")
        if ts:
            first_ts = first_ts or ts
            last_ts = ts
        p = r.get("payload") or {}
        t = r.get("type")
        if t == "event_msg":
            et = p.get("type")
            if et == "item_completed":
                item = p.get("item") or {}
                it = item.get("type")
                content = item.get("content") or []
                texts = []
                for c in content:
                    if isinstance(c, dict):
                        for k in ("text", "input", "output"):
                            if k in c and isinstance(c[k], str):
                                texts.append(c[k])
                blob = " ".join(texts)
                if it == "UserMessage":
                    users.append((ts, blob))
                elif it == "AgentMessage":
                    finals.append((ts, blob, "agent"))
                elif it in ("FunctionCall", "ToolCall", "CustomToolCall"):
                    toolcalls.append((ts, item.get("name") or it, item.get("arguments") or item.get("input") or "", ""))
                elif it in ("FunctionCallOutput", "ToolResult"):
                    toolcalls.append((ts, "OUTPUT", str(blob)[:400], ""))
            elif et == "task_complete":
                finals.append((ts, p.get("last_agent_message") or "", "task"))
            elif et == "thread_goal_updated":
                g = p.get("goal") or {}
                goals.append((ts, g.get("objective") or "", g.get("status")))
            elif et == "user_message":
                users.append((ts, p.get("message") or ""))
        elif t == "compacted":
            compactions.append((ts, str(p)[:200]))

def clean(s):
    return " ".join(str(s).split())

lines = []
lines.append(f"FILE: {os.path.basename(path)}")
lines.append(f"range: {first_ts} -> {last_ts}")
lines.append(f"user turns: {len(users)}   agent finals: {len(finals)}   tool events: {len(toolcalls)}   compactions: {len(compactions)}")
lines.append("")
lines.append("### THREAD GOALS")
for ts, g, st in goals:
    lines.append(f"[{ts}] ({st}) {clean(g)[:3000]}")
lines.append("")
lines.append("### USER MESSAGES")
for i, (ts, u) in enumerate(users, 1):
    lines.append(f"[{ts}] U{i}: {clean(u)[:8000]}")
lines.append("")
lines.append("### AGENT FINAL MESSAGES")
for i, (ts, a, kind) in enumerate(finals, 1):
    lines.append(f"[{ts}] A{i}({kind}): {clean(a)[:8000]}")

with open(out, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("done users", len(users), "finals", len(finals), "goals", len(goals), "tools", len(toolcalls))
print("range", first_ts, last_ts)
