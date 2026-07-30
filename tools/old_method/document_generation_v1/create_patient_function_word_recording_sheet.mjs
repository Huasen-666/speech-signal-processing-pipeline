import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const projectRoot = process.cwd();
const libraryRoot = path.join(projectRoot, "word_library", "function_words");
const outputDir = path.join(projectRoot, "outputs");
const outputPath = path.join(outputDir, "patient_function_word_recording_sheet.xlsx");

const categoryOrder = [
  "pronouns",
  "auxiliaries",
  "articles",
  "determiners",
  "conjunctions",
  "prepositions_particles",
];

const categoryLabels = {
  pronouns: "Pronoun",
  auxiliaries: "Auxiliary / Verb",
  articles: "Article",
  determiners: "Determiner",
  conjunctions: "Conjunction",
  prepositions_particles: "Preposition / Particle",
};

const coreWords = new Set([
  "i",
  "am",
  "you",
  "he",
  "she",
  "we",
  "they",
  "it",
  "the",
  "a",
  "and",
  "but",
  "is",
  "are",
  "can",
  "will",
  "to",
  "in",
  "on",
  "up",
  "with",
  "for",
  "at",
  "this",
  "that",
  "my",
  "your",
  "his",
  "her",
  "their",
]);

async function listWordRows() {
  const rows = [];
  for (const category of categoryOrder) {
    const categoryPath = path.join(libraryRoot, category);
    let wordDirs = [];
    try {
      wordDirs = await fs.readdir(categoryPath, { withFileTypes: true });
    } catch {
      continue;
    }
    for (const entry of wordDirs.filter((item) => item.isDirectory()).sort((a, b) => a.name.localeCompare(b.name))) {
      const word = entry.name.toLowerCase();
      const wavPath = path.join(categoryPath, entry.name, `${entry.name}.wav`);
      rows.push({
        category,
        categoryLabel: categoryLabels[category] ?? category,
        word,
        priority: coreWords.has(word) ? "Core" : "Useful",
        repetitions: coreWords.has(word) ? 5 : 3,
        pause: "1.0 sec",
        prompt: `Say "${word}" clearly.`,
        fileName: `{patient_id}_${word}_001.wav`,
        notes: "Record isolated word first; keep volume and device position consistent.",
        wavPath,
      });
    }
  }
  return rows;
}

function titleCase(text) {
  return text
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

const instructions = [
  ["Purpose", "Create a short patient-specific function-word library for calibration and sentence rendering."],
  ["Device setup", "Use the same device position, gain, and microphone distance planned for normal product use."],
  ["Recording style", "Record each word naturally. Do not exaggerate pronunciation unless the patient normally speaks that way."],
  ["Repetitions", "Core words: 5 clean repetitions. Useful words: 3 clean repetitions."],
  ["Silence", "Leave about 1 second of silence before and after each word."],
  ["Quality check", "Reject files with clipping, very low volume, background speech, or wrong word."],
  ["File naming", "Use {patient_id}_{word}_{take}.wav, for example david_the_001.wav."],
  ["Training use", "Use these isolated words for patient-specific calibration, then test with controlled B/P sentences."],
];

const sentenceRows = [
  ["S01", "I am with you.", "Pronouns + auxiliary + preposition"],
  ["S02", "It is in the bag.", "Pronoun + auxiliary + article + preposition"],
  ["S03", "He can put it up.", "Pronoun + auxiliary + particle"],
  ["S04", "She will be with them.", "Pronoun + auxiliary + preposition"],
  ["S05", "They are on the path.", "Pronoun + auxiliary + article + preposition"],
  ["S06", "This is for you and me.", "Determiner + auxiliary + conjunction"],
  ["S07", "That was in their box.", "Determiner + auxiliary + possessive"],
  ["S08", "We can go to the beach.", "Pronoun + auxiliary + preposition + article"],
  ["S09", "My pen is on the pad.", "Determiner + auxiliary + preposition + article"],
  ["S10", "Your bag is up there.", "Determiner + auxiliary + particle"],
];

const rows = await listWordRows();
const workbook = Workbook.create();

const sheet = workbook.worksheets.add("Recording Sheet");
const headers = [
  "ID",
  "Category",
  "Word",
  "Priority",
  "Repetitions",
  "Pause",
  "Patient Prompt",
  "Recommended File Name",
  "Notes",
  "Current Library WAV",
];
const data = rows.map((row, idx) => [
  `FW${String(idx + 1).padStart(3, "0")}`,
  row.categoryLabel,
  row.word,
  row.priority,
  row.repetitions,
  row.pause,
  row.prompt,
  row.fileName,
  row.notes,
  row.wavPath,
]);
sheet.getRange(`A1:J${data.length + 1}`).values = [headers, ...data];

const instructionSheet = workbook.worksheets.add("Instructions");
instructionSheet.getRange(`A1:B${instructions.length + 2}`).values = [
  ["Patient Function Word Recording Guide", ""],
  ["Field", "Guidance"],
  ...instructions,
];

const sentenceSheet = workbook.worksheets.add("Controlled Sentences");
sentenceSheet.getRange(`A1:C${sentenceRows.length + 1}`).values = [
  ["ID", "Sentence", "Purpose"],
  ...sentenceRows,
];

const summarySheet = workbook.worksheets.add("Summary");
const categorySummary = categoryOrder.map((category) => {
  const categoryRows = rows.filter((row) => row.category === category);
  return [
    titleCase(category),
    categoryRows.length,
    categoryRows.filter((row) => row.priority === "Core").length,
    categoryRows.filter((row) => row.priority === "Useful").length,
  ];
});
summarySheet.getRange(`A1:D${categorySummary.length + 4}`).values = [
  ["Patient Function Word Recording Sheet", "", "", ""],
  ["Source library", libraryRoot, "", ""],
  ["Total words", rows.length, "", ""],
  ["Category", "Word Count", "Core", "Useful"],
  ...categorySummary,
];

const inspect = await workbook.inspect({
  kind: "table",
  range: "Recording Sheet!A1:J12",
  include: "values",
  tableMaxRows: 12,
  tableMaxCols: 10,
});
console.log(inspect.ndjson);

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "formula error scan",
});
console.log(errors.ndjson);

await workbook.render({ sheetName: "Recording Sheet", range: "A1:J20", scale: 1 });
await workbook.render({ sheetName: "Instructions", range: "A1:B10", scale: 1 });
await workbook.render({ sheetName: "Controlled Sentences", range: "A1:C11", scale: 1 });
await workbook.render({ sheetName: "Summary", range: "A1:D10", scale: 1 });

await fs.mkdir(outputDir, { recursive: true });
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(outputPath);
