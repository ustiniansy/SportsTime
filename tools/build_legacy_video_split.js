const fs = require("fs");
const path = require("path");

function parseArgs(argv) {
  const args = {
    train: "_tmp_remote_splits/v2_train.json",
    test: "_tmp_remote_splits/v2_test.json",
    dataDir: "data",
    outDir: "splits",
  };

  for (let i = 2; i < argv.length; i += 1) {
    const key = argv[i];
    const value = argv[i + 1];
    if (!key.startsWith("--") || value === undefined) continue;
    args[key.slice(2)] = value;
    i += 1;
  }
  return args;
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function writeJson(filePath, value) {
  fs.writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

function legacyVideoId(item) {
  return path.basename(String(item.path || "")).replace(/\.mp4$/i, "");
}

function collectCurrentItems(dataDir) {
  const items = [];
  for (const sport of fs.readdirSync(dataDir).sort()) {
    const sportDir = path.join(dataDir, sport);
    if (!fs.statSync(sportDir).isDirectory()) continue;

    for (const fileName of ["full_game.json", "highlight.json"]) {
      const filePath = path.join(sportDir, fileName);
      if (!fs.existsSync(filePath)) continue;

      const dataSplit = fileName.replace(".json", "");
      for (const item of readJson(filePath)) {
        items.push({
          id: item.id,
          video_id: item.video_id,
          sport,
          data_split: dataSplit,
          task_type: item.task_type,
        });
      }
    }
  }
  return items;
}

function countBy(items, key) {
  const counts = {};
  for (const item of items) {
    const value = item[key] || "unknown";
    counts[value] = (counts[value] || 0) + 1;
  }
  return Object.fromEntries(Object.entries(counts).sort(([a], [b]) => a.localeCompare(b)));
}

function summarize(items, videoIds) {
  return {
    num_samples: items.length,
    num_video_ids: videoIds.length,
    by_sport: countBy(items, "sport"),
    by_data_split: countBy(items, "data_split"),
    by_task_type: countBy(items, "task_type"),
  };
}

function main() {
  const args = parseArgs(process.argv);
  const legacyTrain = readJson(args.train);
  const legacyTest = readJson(args.test);

  const legacyTrainVideos = new Set(legacyTrain.map(legacyVideoId));
  const legacyTestVideos = new Set(legacyTest.map(legacyVideoId));
  const overlap = [...legacyTrainVideos].filter((videoId) => legacyTestVideos.has(videoId));
  if (overlap.length > 0) {
    throw new Error(`Legacy train/test split is not video-disjoint. Example: ${overlap[0]}`);
  }

  const currentItems = collectCurrentItems(args.dataDir);

  const trainItems = [];
  const testItems = [];
  const legacyUnassignedItems = [];

  for (const item of currentItems) {
    if (legacyTrainVideos.has(item.video_id)) {
      trainItems.push(item);
    } else if (legacyTestVideos.has(item.video_id)) {
      testItems.push(item);
    } else {
      trainItems.push(item);
      legacyUnassignedItems.push(item);
    }
  }

  const trainVideoIds = [...new Set(trainItems.map((item) => item.video_id))].sort();
  const testVideoIds = [...new Set(testItems.map((item) => item.video_id))].sort();
  const legacyUnassignedVideoIds = [...new Set(legacyUnassignedItems.map((item) => item.video_id))].sort();
  const splitVideoOverlap = trainVideoIds.filter((videoId) => testVideoIds.includes(videoId));
  if (splitVideoOverlap.length > 0) {
    throw new Error(`Generated train/test split is not video-disjoint. Example: ${splitVideoOverlap[0]}`);
  }

  fs.mkdirSync(args.outDir, { recursive: true });
  const split = {
    name: "SportsTime train/test split",
    split_level: "video_id",
    source: "Official SportsTime train/test split.",
    applies_to: ["data", "data_en"],
    notes: [
      "No video_id appears in both train and test.",
      "The split applies to both data/ and data_en/.",
    ],
    train: {
      ...summarize(trainItems, trainVideoIds),
      sample_ids: trainItems.map((item) => item.id).sort(),
      video_ids: trainVideoIds,
    },
    test: {
      ...summarize(testItems, testVideoIds),
      sample_ids: testItems.map((item) => item.id).sort(),
      video_ids: testVideoIds,
    },
  };
  writeJson(path.join(args.outDir, "split.json"), split);

  console.log(JSON.stringify({
    train: summarize(trainItems, trainVideoIds),
    test: summarize(testItems, testVideoIds),
    legacy_unassigned_current_samples_assigned_to_train: legacyUnassignedItems.length,
    legacy_unassigned_current_video_ids_assigned_to_train: legacyUnassignedVideoIds.length,
  }, null, 2));
}

if (require.main === module) {
  main();
}
