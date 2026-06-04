#!/usr/bin/env node
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

const DEFAULT_INPUT = "data/raw/diffbot_people_latest.jsonl";
const DEFAULT_OUTPUT = "data/processed/profile_dossiers_latest.jsonl";

const BIG_TECH = [
  "Google",
  "Meta",
  "Facebook",
  "Apple",
  "Amazon",
  "Netflix",
  "Microsoft",
  "NVIDIA",
  "OpenAI",
  "Anthropic",
  "Alphabet",
];

const VC_FIRMS = [
  "Sequoia",
  "Andreessen Horowitz",
  "a16z",
  "Accel",
  "Greylock",
  "Kleiner Perkins",
  "Founders Fund",
  "Lightspeed",
  "Bessemer",
  "Benchmark",
  "Menlo Ventures",
  "Khosla",
  "General Catalyst",
  "NEA",
  "Y Combinator",
];

const KEYWORD_TAGS = [
  ["ai", /\b(ai|artificial intelligence|machine learning|deep learning|generative|llm|language model|agents?|robotics)\b/i],
  ["crypto", /\b(crypto|cryptocurrency|blockchain|web3|bitcoin|ethereum|defi|token|coinbase|consensys|aptos|wallet)\b/i],
  ["finance", /\b(finance|financial|fintech|banking|investment banking|private equity|hedge fund|capital markets|goldman|jpmorgan|morgan stanley|blackrock|trading)\b/i],
  ["infrastructure", /\b(infrastructure|platform|distributed systems|cloud|database|developer tools|devtools|security|data platform|systems)\b/i],
  ["research", /\b(research|scientist|phd|laboratory|professor|postdoc|academic)\b/i],
  ["product", /\b(product|product management|pm|growth|go-to-market|gtm)\b/i],
  ["operator", /\b(operations|operator|chief operating|coo|strategy|business development|general manager|gm)\b/i],
  ["marketplace", /\b(marketplace|commerce|consumer|creator|social|network|community)\b/i],
  ["health", /\b(health|healthcare|biotech|bio|medical|therapeutics|genomics)\b/i],
  ["climate", /\b(climate|energy|sustainability|carbon|solar|battery)\b/i],
  ["stanford", /\b(stanford)\b/i],
  ["berkeley", /\b(berkeley|uc berkeley|university of california, berkeley)\b/i],
  ["mit", /\b(mit|massachusetts institute of technology)\b/i],
  ["harvard", /\b(harvard)\b/i],
  ["new_york", /\b(new york|nyc|wall street)\b/i],
];

function parseArgs(argv) {
  const args = {
    input: DEFAULT_INPUT,
    output: DEFAULT_OUTPUT,
    summary: "",
    maxEmployments: 24,
    maxEducations: 10,
    maxSkills: 40,
    maxLocations: 12,
    maxCharacters: 7000,
    limit: 0,
  };

  for (const arg of argv.slice(2)) {
    if (arg.startsWith("--input=")) args.input = arg.split("=").slice(1).join("=");
    else if (arg.startsWith("--output=")) args.output = arg.split("=").slice(1).join("=");
    else if (arg.startsWith("--summary=")) args.summary = arg.split("=").slice(1).join("=");
    else if (arg.startsWith("--max-employments=")) args.maxEmployments = Number(arg.split("=")[1]);
    else if (arg.startsWith("--max-educations=")) args.maxEducations = Number(arg.split("=")[1]);
    else if (arg.startsWith("--max-skills=")) args.maxSkills = Number(arg.split("=")[1]);
    else if (arg.startsWith("--max-locations=")) args.maxLocations = Number(arg.split("=")[1]);
    else if (arg.startsWith("--max-characters=")) args.maxCharacters = Number(arg.split("=")[1]);
    else if (arg.startsWith("--limit=")) args.limit = Number(arg.split("=")[1]);
    else throw new Error(`Unknown argument: ${arg}`);
  }

  for (const key of ["maxEmployments", "maxEducations", "maxSkills", "maxLocations", "maxCharacters", "limit"]) {
    if (!Number.isFinite(args[key]) || args[key] < 0) throw new Error(`${key} must be a non-negative number`);
  }

  if (!args.summary) args.summary = args.output.replace(/\.jsonl$/, ".summary.json");
  return args;
}

function compactText(value) {
  return String(value || "")
    .replace(/\s+/g, " ")
    .replace(/\s+([,.;:])/g, "$1")
    .trim();
}

function uniq(values) {
  const seen = new Set();
  const result = [];
  for (const value of values.map(compactText).filter(Boolean)) {
    const key = value.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    result.push(value);
  }
  return result;
}

function sourceKey(person) {
  return person.diffbotUri || person.id || person.linkedInUri || person.name || "";
}

function primaryLabel(person) {
  const labels = Array.isArray(person.archetypeLabels) ? person.archetypeLabels : [];
  return labels[0] || "unlabeled";
}

function locationText(location) {
  return uniq([location.surfaceForm, [location.city, location.region, location.country].filter(Boolean).join(", ")])[0] || "";
}

function employmentText(employment) {
  const role = [employment.title, employment.employer].map(compactText).filter(Boolean).join(" at ");
  const dates = [employment.from, employment.to].filter(Boolean).join(" to ");
  const categories = Array.isArray(employment.categories) ? employment.categories.filter(Boolean).slice(0, 4).join(", ") : "";
  return [role, dates && `(${dates})`, categories && `[${categories}]`].filter(Boolean).join(" ");
}

function educationText(education) {
  const school = compactText(education.institution);
  const degree = compactText(education.degree);
  const dates = [education.from, education.to].filter(Boolean).join(" to ");
  return [degree, school && `at ${school}`, dates && `(${dates})`].filter(Boolean).join(" ");
}

function roleSortKey(employment) {
  const currentBoost = employment.isCurrent ? "z" : "a";
  return `${currentBoost}:${employment.from || ""}:${employment.to || ""}`;
}

function topCounts(values, limit = 20) {
  const counts = new Map();
  for (const value of values.map(compactText).filter(Boolean)) {
    counts.set(value, (counts.get(value) || 0) + 1);
  }
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .slice(0, limit)
    .map(([value, count]) => ({ value, count }));
}

function detectTags(person, textBlob) {
  const tags = new Set();
  const labels = Array.isArray(person.archetypeLabels) ? person.archetypeLabels : [];
  for (const label of labels) {
    if (label === "founders") tags.add("founder");
    if (label === "vcs") tags.add("vc");
    if (label === "faang_engineers") tags.add("big_tech");
  }

  const employments = Array.isArray(person.employments) ? person.employments : [];
  for (const employment of employments) {
    const title = compactText(employment.title);
    const employer = compactText(employment.employer);
    const role = `${title} ${employer}`;
    if (/\b(co-?founder|founder|founding)\b/i.test(title)) tags.add("founder");
    if (/\b(partner|investor|venture|principal|associate)\b/i.test(title) || VC_FIRMS.some((firm) => employer.toLowerCase().includes(firm.toLowerCase()))) {
      tags.add("vc");
    }
    if (BIG_TECH.some((company) => employer.toLowerCase() === company.toLowerCase() || employer.toLowerCase().includes(company.toLowerCase()))) {
      tags.add("big_tech");
    }
    if (/\b(engineer|engineering|software|machine learning|research)\b/i.test(role)) tags.add("technical");
    if (/\b(founder|partner|investor|engineer|product|strategy|finance)\b/i.test(role)) {
      // Role text is intentionally folded into keyword scan below.
    }
  }

  for (const [tag, pattern] of KEYWORD_TAGS) {
    if (pattern.test(textBlob)) tags.add(tag);
  }

  if (tags.has("new_york") && tags.has("finance") && tags.has("crypto") && tags.has("ai")) {
    tags.add("finance_crypto_ai_rotation");
  }
  if (tags.has("big_tech") && tags.has("founder") && tags.has("infrastructure")) {
    tags.add("ex_big_tech_infra_founder");
  }
  if (tags.has("vc") && tags.has("founder")) {
    tags.add("founder_investor_hybrid");
  }
  return [...tags].sort();
}

function buildDossier(person, options, rowIndex) {
  const labels = Array.isArray(person.archetypeLabels) ? person.archetypeLabels : [];
  const employments = (Array.isArray(person.employments) ? person.employments : [])
    .slice()
    .sort((a, b) => roleSortKey(b).localeCompare(roleSortKey(a)));
  const currentEmployments = employments.filter((employment) => employment.isCurrent).slice(0, options.maxEmployments);
  const pastEmployments = employments.filter((employment) => !employment.isCurrent).slice(0, options.maxEmployments);
  const selectedEmployments = uniq([...currentEmployments, ...pastEmployments].map(employmentText)).slice(0, options.maxEmployments);
  const educations = uniq((Array.isArray(person.educations) ? person.educations : []).map(educationText)).slice(0, options.maxEducations);
  const locations = Array.isArray(person.locations) ? person.locations : [];
  const currentLocations = uniq(locations.filter((location) => location.isCurrent).map(locationText)).slice(0, options.maxLocations);
  const pastLocations = uniq(locations.filter((location) => !location.isCurrent).map(locationText)).slice(0, options.maxLocations);
  const skills = uniq(Array.isArray(person.skills) ? person.skills : []).slice(0, options.maxSkills);
  const employers = uniq(employments.map((employment) => employment.employer)).slice(0, options.maxEmployments);
  const titles = uniq(employments.map((employment) => employment.title)).slice(0, options.maxEmployments);
  const schools = uniq((Array.isArray(person.educations) ? person.educations : []).map((education) => education.institution)).slice(0, options.maxEducations);

  const preSignalText = [
    person.name,
    person.summary,
    person.description,
    labels.join(" "),
    selectedEmployments.join(" "),
    educations.join(" "),
    currentLocations.join(" "),
    pastLocations.join(" "),
    skills.join(" "),
  ].join(" ");
  const careerTags = detectTags(person, preSignalText);

  const lines = [
    `Name: ${compactText(person.name) || "Unknown"}.`,
    labels.length ? `Anchor labels: ${labels.join(", ")}.` : "",
    compactText(person.summary) ? `Summary: ${compactText(person.summary)}.` : "",
    compactText(person.description) ? `Description: ${compactText(person.description)}.` : "",
    currentLocations.length ? `Current location: ${currentLocations.join("; ")}.` : "",
    pastLocations.length ? `Past locations: ${pastLocations.join("; ")}.` : "",
    selectedEmployments.length ? `Employment traces: ${selectedEmployments.join("; ")}.` : "",
    educations.length ? `Education traces: ${educations.join("; ")}.` : "",
    skills.length ? `Skills and keywords: ${skills.join(", ")}.` : "",
    careerTags.length ? `Detected career signals: ${careerTags.join(", ")}.` : "",
  ].filter(Boolean);

  let dossierText = compactText(lines.join("\n"));
  if (dossierText.length > options.maxCharacters) {
    dossierText = `${dossierText.slice(0, options.maxCharacters).replace(/\s+\S*$/, "")}.`;
  }

  return {
    rowIndex,
    source: person.source || "diffbot",
    sourceKey: sourceKey(person),
    id: person.id || "",
    diffbotUri: person.diffbotUri || "",
    name: compactText(person.name),
    archetypeLabels: labels,
    primaryLabel: primaryLabel(person),
    summary: compactText(person.summary),
    currentLocations,
    pastLocations,
    currentEmployments: currentEmployments.map((employment) => ({
      title: compactText(employment.title),
      employer: compactText(employment.employer),
      from: employment.from || "",
      to: employment.to || "",
      categories: Array.isArray(employment.categories) ? employment.categories.filter(Boolean) : [],
    })),
    employers,
    titles,
    schools,
    skills,
    careerTags,
    profileUris: {
      linkedInUri: person.linkedInUri || "",
      crunchbaseUri: person.crunchbaseUri || "",
      twitterUri: person.twitterUri || "",
      githubUri: person.githubUri || "",
      homepageUri: person.homepageUri || "",
    },
    image: person.image || "",
    dossierText,
    textLength: dossierText.length,
  };
}

async function main() {
  const args = parseArgs(process.argv);
  const inputPath = resolve(args.input);
  const outputPath = resolve(args.output);
  const summaryPath = resolve(args.summary);
  const input = await readFile(inputPath, "utf8");
  const people = input
    .split(/\r?\n/)
    .filter((line) => line.trim())
    .map((line) => JSON.parse(line));
  const limited = args.limit ? people.slice(0, args.limit) : people;

  const dossiers = limited.map((person, index) => buildDossier(person, args, index));
  await mkdir(resolve(outputPath, ".."), { recursive: true });
  await writeFile(outputPath, `${dossiers.map((record) => JSON.stringify(record)).join("\n")}\n`, "utf8");

  const careerTags = topCounts(dossiers.flatMap((record) => record.careerTags), 40);
  const employers = topCounts(dossiers.flatMap((record) => record.employers), 40);
  const schools = topCounts(dossiers.flatMap((record) => record.schools), 40);
  const labels = topCounts(dossiers.map((record) => record.primaryLabel), 20);
  const summary = {
    generatedAt: new Date().toISOString(),
    input: inputPath,
    output: outputPath,
    records: dossiers.length,
    labels,
    topCareerTags: careerTags,
    topEmployers: employers,
    topSchools: schools,
    avgTextLength: Math.round(dossiers.reduce((sum, record) => sum + record.textLength, 0) / Math.max(1, dossiers.length)),
  };
  await writeFile(summaryPath, JSON.stringify(summary, null, 2), "utf8");
  console.log(JSON.stringify({ wrote: outputPath, summary: summaryPath, records: dossiers.length }, null, 2));
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
