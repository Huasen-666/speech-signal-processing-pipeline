import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const projectRoot = process.cwd();
const outputDir = path.join(projectRoot, "outputs");
const outputPath = path.join(outputDir, "bp_accuracy_test_sentences.xlsx");

const sentences = [
  {
    id: "S01",
    level: "Short",
    sentence: "Bob packed the big bag.",
    targets: [
      ["Bob", "B"],
      ["packed", "P"],
      ["big", "B"],
      ["bag", "B"],
    ],
  },
  {
    id: "S02",
    level: "Short",
    sentence: "Pat picked the pink pen.",
    targets: [
      ["Pat", "P"],
      ["picked", "P"],
      ["pink", "P"],
      ["pen", "P"],
    ],
  },
  {
    id: "S03",
    level: "Long",
    sentence:
      "Bob packed the big bag and put it in the box, then Pat picked up the pink pen and placed it on the pad.",
    targets: [
      ["Bob", "B"],
      ["packed", "P"],
      ["big", "B"],
      ["bag", "B"],
      ["put", "P"],
      ["box", "B"],
      ["Pat", "P"],
      ["picked", "P"],
      ["pink", "P"],
      ["pen", "P"],
      ["placed", "P"],
      ["pad", "P"],
    ],
  },
  {
    id: "S04",
    level: "Long",
    sentence:
      "The boy put the box by the bed, and Bob paid for the pass before Pat picked the best path.",
    targets: [
      ["boy", "B"],
      ["put", "P"],
      ["box", "B"],
      ["bed", "B"],
      ["Bob", "B"],
      ["paid", "P"],
      ["pass", "P"],
      ["before", "B"],
      ["Pat", "P"],
      ["picked", "P"],
      ["best", "B"],
      ["path", "P"],
    ],
  },
];

function sequence(targets) {
  return targets.map(([, label]) => label).join(" ");
}

function targetWords(targets) {
  return targets.map(([word, label]) => `${word}(${label})`).join(", ");
}

const workbook = Workbook.create();

const setSheet = workbook.worksheets.add("Sentence Set");
setSheet.getRange(`A1:G${sentences.length + 1}`).values = [
  ["ID", "Level", "Sentence", "Target Words", "Ground Truth B/P Sequence", "Target Count", "Purpose"],
  ...sentences.map((item) => [
    item.id,
    item.level,
    item.sentence,
    targetWords(item.targets),
    sequence(item.targets),
    item.targets.length,
    item.level === "Short" ? "Basic B/P sanity check" : "Connected-speech accuracy test",
  ]),
];

const scoringRows = [];
for (const item of sentences) {
  item.targets.forEach(([word, label], index) => {
    scoringRows.push([item.id, item.level, index + 1, word, label, "", ""]);
  });
}

const scoreSheet = workbook.worksheets.add("Scoring Template");
scoreSheet.getRange(`A1:G${scoringRows.length + 1}`).values = [
  ["Sentence ID", "Level", "Target #", "Target Word", "Ground Truth", "Model Prediction", "Correct?"],
  ...scoringRows,
];
scoreSheet.getRange(`G2:G${scoringRows.length + 1}`).formulas = scoringRows.map((_, idx) => {
  const row = idx + 2;
  return [`=IF(F${row}="","",IF(UPPER(F${row})=E${row},1,0))`];
});

const summarySheet = workbook.worksheets.add("Summary");
summarySheet.getRange("A1:B8").values = [
  ["B/P Accuracy Test Sheet", ""],
  ["Short sentences", 2],
  ["Long sentences", 2],
  ["Total target B/P words", scoringRows.length],
  ["How to score", "Enter B or P in Scoring Template column F."],
  ["Accuracy formula", "Accuracy = correct predictions / total target words."],
  ["Recommended recording", "Record each sentence 3 times with the same device setup."],
  ["Important", "Do not use these exact sentences for training."],
];
summarySheet.getRange("A10:B13").values = [
  ["Metric", "Formula"],
  [
    "Overall Accuracy",
    `=IF(COUNT('Scoring Template'!G2:G${scoringRows.length + 1})=0,"",SUM('Scoring Template'!G2:G${scoringRows.length + 1})/COUNT('Scoring Template'!G2:G${scoringRows.length + 1}))`,
  ],
  [
    "Short Sentence Accuracy",
    `=IF(COUNTIFS('Scoring Template'!B2:B${scoringRows.length + 1},"Short",'Scoring Template'!G2:G${scoringRows.length + 1},">=0")=0,"",SUMIF('Scoring Template'!B2:B${scoringRows.length + 1},"Short",'Scoring Template'!G2:G${scoringRows.length + 1})/COUNTIFS('Scoring Template'!B2:B${scoringRows.length + 1},"Short",'Scoring Template'!G2:G${scoringRows.length + 1},">=0"))`,
  ],
  [
    "Long Sentence Accuracy",
    `=IF(COUNTIFS('Scoring Template'!B2:B${scoringRows.length + 1},"Long",'Scoring Template'!G2:G${scoringRows.length + 1},">=0")=0,"",SUMIF('Scoring Template'!B2:B${scoringRows.length + 1},"Long",'Scoring Template'!G2:G${scoringRows.length + 1})/COUNTIFS('Scoring Template'!B2:B${scoringRows.length + 1},"Long",'Scoring Template'!G2:G${scoringRows.length + 1},">=0"))`,
  ],
];

const instructionSheet = workbook.worksheets.add("Instructions");
instructionSheet.getRange("A1:B9").values = [
  ["Step", "Instruction"],
  ["1", "Ask the patient to read each sentence naturally."],
  ["2", "Record three takes for each sentence."],
  ["3", "Run your B/P detector on the audio."],
  ["4", "For every target word, write the model output as B or P."],
  ["5", "Use the Correct? column to compute accuracy."],
  ["6", "If short sentences work but long sentences fail, the problem is likely segmentation or connected-speech timing."],
  ["7", "If both short and long sentences fail, the model probably needs more patient-specific calibration data."],
  ["8", "Keep this test set separate from training data."],
];

const inspect = await workbook.inspect({
  kind: "table",
  range: "Sentence Set!A1:G5",
  include: "values",
  tableMaxRows: 5,
  tableMaxCols: 7,
});
console.log(inspect.ndjson);

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "formula error scan",
});
console.log(errors.ndjson);

await workbook.render({ sheetName: "Sentence Set", range: "A1:G5", scale: 1 });
await workbook.render({ sheetName: "Scoring Template", range: "A1:G18", scale: 1 });
await workbook.render({ sheetName: "Summary", range: "A1:B13", scale: 1 });
await workbook.render({ sheetName: "Instructions", range: "A1:B9", scale: 1 });

await fs.mkdir(outputDir, { recursive: true });
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(outputPath);
