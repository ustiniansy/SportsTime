const fs = require("fs");
const path = require("path");

const DATA_DIRS = ["data", "data_en"];
const CANONICAL_TASK_TYPES = new Set([
  "Perception",
  "Temporal",
  "Tactical",
  "Causal",
  "Counterfactual",
]);

const EXACT_MAP = new Map([
  ["Perception", "Perception"],
  ["Temporal", "Temporal"],
  ["Tactical", "Tactical"],
  ["Causal", "Causal"],
  ["Counterfactual", "Counterfactual"],

  ["感知", "Perception"],
  ["感知推理", "Perception"],
  ["感知相关", "Perception"],
  ["感知相关推理", "Perception"],
  ["感知相关的推理", "Perception"],
  ["感知相关的推理（基础动作与事件识别）", "Perception"],
  ["基础感知", "Perception"],
  ["覆盖感知", "Perception"],
  ["动作分析", "Perception"],
  ["行为识别", "Perception"],
  ["结果判断", "Perception"],
  ["结果预测/感知", "Perception"],
  ["????", "Perception"],

  ["时序", "Temporal"],
  ["時序", "Temporal"],
  ["时序推理", "Temporal"],
  ["时序相关", "Temporal"],
  ["时序相关推理", "Temporal"],
  ["时序相关的推理", "Temporal"],
  ["时序相关的推理（比赛进程与阶段分析）", "Temporal"],
  ["多事件时序推理", "Temporal"],
  ["时空推理", "Temporal"],
  ["时序分析", "Temporal"],
  ["时序/感知推理", "Temporal"],
  ["时序与结果推理", "Temporal"],
  ["时序与事件推理", "Temporal"],

  ["战术", "Tactical"],
  ["戰術", "Tactical"],
  ["战术推理", "Tactical"],
  ["战术分析", "Tactical"],
  ["战术相关", "Tactical"],
  ["战术相关推理", "Tactical"],
  ["战术相关的推理", "Tactical"],
  ["战术相关的推理（战术配合相关问题）", "Tactical"],
  ["战术与感知", "Tactical"],
  ["战术执行分析", "Tactical"],
  ["战术/结果分析", "Tactical"],
  ["防守分析", "Tactical"],
  ["关键事件分析", "Tactical"],

  ["因果", "Causal"],
  ["因果推理", "Causal"],
  ["规则", "Causal"],
  ["规则推理", "Causal"],
  ["规则解读", "Causal"],
  ["规则与因果推理", "Causal"],
  ["规则与事件推理", "Causal"],
  ["感知与结果推理", "Causal"],

  ["反事实", "Counterfactual"],
  ["反事实推理", "Counterfactual"],
  ["反事實推理", "Counterfactual"],
  ["反事实/归因推理", "Counterfactual"],
  ["反事实推理（战术可能性推演）", "Counterfactual"],
]);

function normalizeTaskType(raw) {
  const value = String(raw || "").trim();
  if (EXACT_MAP.has(value)) return EXACT_MAP.get(value);

  if (/counterfactual/i.test(value) || value.includes("反事实") || value.includes("反事實")) {
    return "Counterfactual";
  }
  if (/causal/i.test(value) || value.includes("因果") || value.includes("归因") || value.includes("规则")) {
    return "Causal";
  }
  if (/tactical/i.test(value) || value.includes("战术") || value.includes("戰術") || value.includes("防守")) {
    return "Tactical";
  }
  if (/temporal/i.test(value) || value.includes("时序") || value.includes("時序") || value.includes("时空")) {
    return "Temporal";
  }
  if (/perception/i.test(value) || value.includes("感知") || value.includes("动作") || value.includes("行为") || value.includes("结果")) {
    return "Perception";
  }
  return null;
}

function collectJsonFiles(baseDir) {
  const files = [];
  for (const sport of fs.readdirSync(baseDir)) {
    const sportDir = path.join(baseDir, sport);
    if (!fs.statSync(sportDir).isDirectory()) continue;
    for (const file of fs.readdirSync(sportDir)) {
      if (file.endsWith(".json")) files.push(path.join(sportDir, file));
    }
  }
  return files.sort();
}

function normalizeFile(filePath) {
  const items = JSON.parse(fs.readFileSync(filePath, "utf8"));
  const counts = new Map();

  for (const item of items) {
    const raw = item.task_type_raw || item.task_type;
    const normalized = normalizeTaskType(raw);
    if (!CANONICAL_TASK_TYPES.has(normalized)) {
      throw new Error(`Unmapped task_type "${raw}" in ${filePath} (${item.id || "unknown id"})`);
    }

    item.task_type_raw = raw;
    item.task_type = normalized;
    counts.set(normalized, (counts.get(normalized) || 0) + 1);
  }

  fs.writeFileSync(filePath, stringifyItems(items) + "\n", "utf8");
  return counts;
}

function orderedItem(item) {
  const preferred = [
    "id",
    "video_id",
    "task_type",
    "task_type_raw",
    "question",
    "CoT",
    "answer",
    "time_refs",
  ];
  const out = {};
  for (const key of preferred) {
    if (Object.prototype.hasOwnProperty.call(item, key)) {
      out[key] = item[key];
    }
  }
  for (const key of Object.keys(item)) {
    if (!Object.prototype.hasOwnProperty.call(out, key)) {
      out[key] = item[key];
    }
  }
  return out;
}

function compactObject(obj) {
  const keys = ["type", "ts", "start", "end", ...Object.keys(obj).filter(
    (key) => !["type", "ts", "start", "end"].includes(key),
  )];
  const parts = [];
  for (const key of keys) {
    if (Object.prototype.hasOwnProperty.call(obj, key)) {
      parts.push(`${JSON.stringify(key)}: ${JSON.stringify(obj[key])}`);
    }
  }
  return `{${parts.join(", ")}}`;
}

function stringifyValue(value, indent) {
  const pad = " ".repeat(indent);
  const nextPad = " ".repeat(indent + 2);

  if (Array.isArray(value)) {
    if (value.every((item) => item && typeof item === "object" && !Array.isArray(item))) {
      const rows = value.map((item) => `${nextPad}${compactObject(item)}`);
      return `[\n${rows.join(",\n")}\n${pad}]`;
    }
    return JSON.stringify(value, null, 2)
      .split("\n")
      .map((line, i) => (i === 0 ? line : pad + line))
      .join("\n");
  }

  return JSON.stringify(value);
}

function stringifyItems(items) {
  const blocks = items.map((rawItem) => {
    const item = orderedItem(rawItem);
    const keys = Object.keys(item);
    const lines = ["  {"];
    keys.forEach((key, index) => {
      const comma = index === keys.length - 1 ? "" : ",";
      lines.push(`    ${JSON.stringify(key)}: ${stringifyValue(item[key], 4)}${comma}`);
    });
    lines.push("  }");
    return lines.join("\n");
  });
  return `[\n${blocks.join(",\n")}\n]`;
}

function main() {
  const totalCounts = new Map();
  let fileCount = 0;
  let itemCount = 0;

  for (const baseDir of DATA_DIRS) {
    for (const filePath of collectJsonFiles(baseDir)) {
      const counts = normalizeFile(filePath);
      fileCount += 1;
      for (const [taskType, count] of counts.entries()) {
        totalCounts.set(taskType, (totalCounts.get(taskType) || 0) + count);
        itemCount += count;
      }
    }
  }

  console.log(`Normalized ${itemCount} items across ${fileCount} files.`);
  for (const taskType of [...CANONICAL_TASK_TYPES].sort()) {
    console.log(`${taskType}: ${totalCounts.get(taskType) || 0}`);
  }
}

if (require.main === module) {
  main();
}

module.exports = { normalizeTaskType };
