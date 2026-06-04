import { mkdir, readFile, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { resolve } from "node:path";

const ROOT = resolve(new URL("..", import.meta.url).pathname);
const DQL_ENDPOINT = "https://kg.diffbot.com/kg/v3/dql";

const bayAreaCities = [
  "San Francisco",
  "Oakland",
  "Berkeley",
  "Daly City",
  "San Mateo",
  "Redwood City",
  "Menlo Park",
  "Palo Alto",
  "Mountain View",
  "Sunnyvale",
  "Santa Clara",
  "Cupertino",
  "Milpitas",
  "Fremont",
  "Hayward",
  "Pleasanton",
  "San Ramon",
  "San Jose",
];

const fields = [
  "id",
  "diffbotUri",
  "name",
  "summary",
  "description",
  "image",
  "images",
  "locations",
  "educations",
  "employments",
  "skills",
  "linkedInUri",
  "crunchbaseUri",
  "twitterUri",
  "githubUri",
  "homepageUri",
  "importance",
  "crawlTimestamp",
];

const cityFilter =
  'locations.{isCurrent:true country.name:"United States" region.name:"California" city.name:or(' +
  bayAreaCities.map((city) => JSON.stringify(city)).join(",") +
  ")}";

const archetypes = [
  {
    id: "founders",
    label: "Bay Area Founders / Cofounders",
    dql: `type:Person
${cityFilter}
has:image
employments.{isCurrent:true title:or("Founder","Co-Founder","Cofounder","Co Founder","Founding Partner")}
get:${fields.join(",")}`,
  },
  {
    id: "vcs",
    label: "Bay Area VCs",
    dql: `type:Person
${cityFilter}
has:image
employments.{isCurrent:true title:or("General Partner","Managing Partner","Partner","Investor","Venture Partner","Principal","Associate")}
employments.employer.name:or("Sequoia Capital","Andreessen Horowitz","a16z","Accel","Greylock","Kleiner Perkins","Founders Fund","Lightspeed Venture Partners","Bessemer Venture Partners","Benchmark","Menlo Ventures","Khosla Ventures","General Catalyst","NEA","Y Combinator")
get:${fields.join(",")}`,
  },
  {
    id: "faang_engineers",
    label: "Bay Area FAANG / Big-Tech Engineers",
    dql: `type:Person
${cityFilter}
has:image
employments.{isCurrent:true title:or("Software Engineer","Senior Software Engineer","Staff Software Engineer","Principal Software Engineer","Engineering Manager","Machine Learning Engineer","Research Engineer") employer.name:or("Google","Meta","Apple","Amazon","Netflix","Microsoft","NVIDIA","OpenAI","Anthropic")}
get:${fields.join(",")}`,
  },
];

function parseArgs(argv) {
  const args = {
    dryRun: false,
    maxPerClass: 25,
    pageSize: 10,
    only: null,
  };

  for (const arg of argv.slice(2)) {
    if (arg === "--dry-run") args.dryRun = true;
    else if (arg.startsWith("--max-per-class=")) args.maxPerClass = Number(arg.split("=")[1]);
    else if (arg.startsWith("--page-size=")) args.pageSize = Number(arg.split("=")[1]);
    else if (arg.startsWith("--only=")) args.only = arg.split("=")[1].split(",").filter(Boolean);
    else throw new Error(`Unknown argument: ${arg}`);
  }

  if (!Number.isInteger(args.maxPerClass) || args.maxPerClass < 1) {
    throw new Error("--max-per-class must be a positive integer");
  }
  if (!Number.isInteger(args.pageSize) || args.pageSize < 1 || args.pageSize > 100) {
    throw new Error("--page-size must be an integer from 1 to 100");
  }

  return args;
}

function parseEnv(text) {
  return Object.fromEntries(
    text
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter((line) => line && !line.startsWith("#"))
      .map((line) => {
        const index = line.indexOf("=");
        if (index === -1) return [line, ""];
        const key = line.slice(0, index).trim();
        let value = line.slice(index + 1).trim();
        if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
          value = value.slice(1, -1);
        }
        return [key, value];
      }),
  );
}

async function loadToken() {
  if (process.env.DIFFBOT_TOKEN) return process.env.DIFFBOT_TOKEN;

  const paths = [
    resolve(ROOT, ".env.local"),
    resolve(ROOT, ".env"),
    resolve(ROOT, "..", "chindian-heatmap", ".env.local"),
  ];

  for (const path of paths) {
    if (!existsSync(path)) continue;
    const env = parseEnv(await readFile(path, "utf8"));
    if (env.DIFFBOT_TOKEN) return env.DIFFBOT_TOKEN;
  }

  throw new Error("Missing DIFFBOT_TOKEN. Add it to .env.local or export it in the shell.");
}

function unwrap(value) {
  if (Array.isArray(value)) return value.map(unwrap).filter((item) => item !== undefined && item !== null);
  if (!value || typeof value !== "object") return value;

  const keys = Object.keys(value);
  if (keys.includes("value") && keys.every((key) => ["value", "confidence", "origin", "origins"].includes(key))) {
    return unwrap(value.value);
  }

  return Object.fromEntries(
    Object.entries(value)
      .map(([key, nested]) => [key, unwrap(nested)])
      .filter(([, nested]) => nested !== undefined && nested !== null && nested !== ""),
  );
}

function text(value) {
  if (typeof value === "string") return value;
  if (value && typeof value === "object" && typeof value.name === "string") return value.name;
  return "";
}

function dateString(value) {
  if (!value) return "";
  if (typeof value === "string") return value;
  if (typeof value.str === "string") return value.str;
  return "";
}

function imageUrl(value) {
  if (!value) return "";
  if (typeof value === "string") return value;
  if (typeof value.url === "string") return value.url;
  return "";
}

function normalizeImages(person) {
  const urls = new Set();
  const primary = imageUrl(person.image);
  if (primary) urls.add(primary);
  for (const item of Array.isArray(person.images) ? person.images : []) {
    const url = imageUrl(item);
    if (url) urls.add(url);
  }
  return [...urls];
}

function normalizeLocations(person) {
  return (Array.isArray(person.locations) ? person.locations : []).map((location) => ({
    isCurrent: Boolean(location.isCurrent),
    city: text(location.city),
    region: text(location.region),
    country: text(location.country),
    surfaceForm: location.surfaceForm || location.address || "",
  }));
}

function normalizeEducations(person) {
  return (Array.isArray(person.educations) ? person.educations : []).map((education) => ({
    isCurrent: Boolean(education.isCurrent),
    institution: text(education.institution),
    degree: text(education.degree),
    from: dateString(education.from),
    to: dateString(education.to),
  }));
}

function normalizeEmployments(person) {
  return (Array.isArray(person.employments) ? person.employments : []).map((employment) => ({
    isCurrent: Boolean(employment.isCurrent),
    title: employment.title || "",
    employer: text(employment.employer),
    employerDiffbotUri: employment.employer?.diffbotUri || "",
    categories: (Array.isArray(employment.categories) ? employment.categories : []).map(text).filter(Boolean),
    from: dateString(employment.from),
    to: dateString(employment.to),
  }));
}

function normalizeSkills(person) {
  return (Array.isArray(person.skills) ? person.skills : []).map(text).filter(Boolean);
}

function normalizePerson(raw, archetypeId) {
  const person = unwrap(raw);
  const images = normalizeImages(person);
  return {
    source: "diffbot",
    archetypeLabels: [archetypeId],
    id: person.id || "",
    diffbotUri: person.diffbotUri || "",
    name: person.name || "",
    summary: person.summary || "",
    description: person.description || "",
    image: images[0] || "",
    images,
    locations: normalizeLocations(person),
    educations: normalizeEducations(person),
    employments: normalizeEmployments(person),
    skills: normalizeSkills(person),
    linkedInUri: person.linkedInUri || "",
    crunchbaseUri: person.crunchbaseUri || "",
    twitterUri: person.twitterUri || "",
    githubUri: person.githubUri || "",
    homepageUri: person.homepageUri || "",
    importance: typeof person.importance === "number" ? person.importance : null,
    crawlTimestamp: person.crawlTimestamp || null,
  };
}

function personKey(person) {
  return person.diffbotUri || person.id || person.linkedInUri || person.name;
}

async function queryDql(token, query, { size, from }) {
  const url = new URL(DQL_ENDPOINT);
  url.searchParams.set("token", token);
  url.searchParams.set("type", "query");
  url.searchParams.set("size", String(size));
  url.searchParams.set("from", String(from));
  url.searchParams.set("query", query);

  const response = await fetch(url, { headers: { Accept: "application/json" } });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(`DQL request failed: ${response.status} ${body.slice(0, 500)}`);
  }
  return response.json();
}

function dataRows(json) {
  if (Array.isArray(json.data)) return json.data;
  if (Array.isArray(json.data?.hits)) return json.data.hits;
  if (Array.isArray(json.hits)) return json.hits;
  return [];
}

async function fetchArchetype(token, archetype, args) {
  const rows = [];
  for (let from = 0; rows.length < args.maxPerClass; from += args.pageSize) {
    const size = Math.min(args.pageSize, args.maxPerClass - rows.length);
    const json = await queryDql(token, archetype.dql, { size, from });
    const pageRows = dataRows(json);
    if (!pageRows.length) break;
    rows.push(...pageRows);
    if (pageRows.length < size) break;
  }
  return rows.slice(0, args.maxPerClass);
}

function timestampSlug() {
  return new Date().toISOString().replace(/[-:]/g, "").replace(/\.\d{3}Z$/, "Z");
}

async function main() {
  const args = parseArgs(process.argv);
  const selected = args.only ? archetypes.filter((archetype) => args.only.includes(archetype.id)) : archetypes;
  if (!selected.length) throw new Error("No archetypes selected.");

  if (args.dryRun) {
    console.log(
      JSON.stringify(
        {
          endpoint: DQL_ENDPOINT,
          maxPerClass: args.maxPerClass,
          pageSize: args.pageSize,
          archetypes: selected.map(({ id, label, dql }) => ({ id, label, dql })),
        },
        null,
        2,
      ),
    );
    return;
  }

  const token = await loadToken();
  const byKey = new Map();
  const stats = {};

  for (const archetype of selected) {
    process.stdout.write(`Querying ${archetype.id}... `);
    const rows = await fetchArchetype(token, archetype, args);
    let added = 0;
    let merged = 0;
    let noImage = 0;

    for (const row of rows) {
      const person = normalizePerson(row.entity || row, archetype.id);
      if (!person.image) {
        noImage += 1;
        continue;
      }
      const key = personKey(person);
      if (byKey.has(key)) {
        const existing = byKey.get(key);
        existing.archetypeLabels = [...new Set([...existing.archetypeLabels, archetype.id])];
        merged += 1;
      } else {
        byKey.set(key, person);
        added += 1;
      }
    }

    stats[archetype.id] = {
      requested: args.maxPerClass,
      returned: rows.length,
      added,
      merged,
      skippedNoImage: noImage,
    };
    process.stdout.write(`${rows.length} returned, ${added} added, ${merged} merged\n`);
  }

  const people = [...byKey.values()];
  const outDir = resolve(ROOT, "data", "raw");
  await mkdir(outDir, { recursive: true });

  const stamp = timestampSlug();
  const snapshotPath = resolve(outDir, `diffbot_people_${stamp}.jsonl`);
  const latestPath = resolve(outDir, "diffbot_people_latest.jsonl");
  const summaryPath = resolve(outDir, `diffbot_people_${stamp}.summary.json`);
  const latestSummaryPath = resolve(outDir, "diffbot_people_latest.summary.json");

  const ndjson = people.map((person) => JSON.stringify(person)).join("\n") + "\n";
  const summary = {
    generatedAt: new Date().toISOString(),
    source: "diffbot-dql",
    privacyNote: "Profile metadata and image URLs only. Email and phone fields are intentionally not requested.",
    maxPerClass: args.maxPerClass,
    pageSize: args.pageSize,
    totalPeople: people.length,
    stats,
    archetypes: selected.map(({ id, label, dql }) => ({ id, label, dql })),
  };

  await writeFile(snapshotPath, ndjson);
  await writeFile(latestPath, ndjson);
  await writeFile(summaryPath, JSON.stringify(summary, null, 2));
  await writeFile(latestSummaryPath, JSON.stringify(summary, null, 2));

  console.log(`Wrote ${people.length} people`);
  console.log(snapshotPath);
  console.log(summaryPath);
}

main().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});
