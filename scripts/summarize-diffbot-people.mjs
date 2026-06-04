import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

const ROOT = resolve(new URL("..", import.meta.url).pathname);
const inputPath = process.argv[2] || resolve(ROOT, "data", "raw", "diffbot_people_latest.jsonl");

function topCounts(values, limit = 12) {
  const counts = {};
  for (const value of values.filter(Boolean)) {
    counts[value] = (counts[value] || 0) + 1;
  }
  return Object.entries(counts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, limit)
    .map(([value, count]) => ({ value, count }));
}

function coverage(rows, fields) {
  return Object.fromEntries(fields.map((field) => [field, rows.filter((row) => Boolean(row[field])).length]));
}

const text = await readFile(inputPath, "utf8");
const rows = text
  .trim()
  .split("\n")
  .filter(Boolean)
  .map((line) => JSON.parse(line));

const labels = {};
const currentTitles = [];
const currentEmployers = [];
const schools = [];

for (const row of rows) {
  for (const label of row.archetypeLabels || []) labels[label] = (labels[label] || 0) + 1;
  for (const job of (row.employments || []).filter((job) => job.isCurrent)) {
    currentTitles.push(job.title);
    currentEmployers.push(job.employer);
  }
  for (const education of row.educations || []) {
    schools.push(education.institution);
  }
}

const employmentCounts = rows.map((row) => row.employments?.length || 0).sort((a, b) => a - b);
const educationCounts = rows.map((row) => row.educations?.length || 0).sort((a, b) => a - b);

console.log(
  JSON.stringify(
    {
      inputPath,
      totalPeople: rows.length,
      labels,
      fieldCoverage: coverage(rows, [
        "image",
        "summary",
        "description",
        "linkedInUri",
        "crunchbaseUri",
        "twitterUri",
        "githubUri",
        "homepageUri",
      ]),
      employmentCountRange: [employmentCounts[0] || 0, employmentCounts.at(-1) || 0],
      educationCountRange: [educationCounts[0] || 0, educationCounts.at(-1) || 0],
      topCurrentTitles: topCounts(currentTitles),
      topCurrentEmployers: topCounts(currentEmployers),
      topSchools: topCounts(schools),
    },
    null,
    2,
  ),
);

