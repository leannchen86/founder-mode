import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
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

const cityFilter =
  'locations.{isCurrent:true country.name:"United States" region.name:"California" city.name:or(' +
  bayAreaCities.map((city) => JSON.stringify(city)).join(",") +
  ")}";

const buckets = [
  {
    id: "founders_broad_with_ceo",
    label: "Founder-ish, broad founder/CEO title filter",
    dql: `type:Person
${cityFilter}
has:image
employments.{isCurrent:true title:or("Founder","Co-Founder","Cofounder","Co Founder","Founding Partner","CEO","Chief Executive Officer")}`,
  },
  {
    id: "founders_strict_no_ceo",
    label: "Founder/cofounder title filter, excluding generic CEO",
    dql: `type:Person
${cityFilter}
has:image
employments.{isCurrent:true title:or("Founder","Co-Founder","Cofounder","Co Founder","Founding Partner")}`,
  },
  {
    id: "ceos_only",
    label: "CEO title filter only",
    dql: `type:Person
${cityFilter}
has:image
employments.{isCurrent:true title:or("CEO","Chief Executive Officer")}`,
  },
  {
    id: "vcs_current_filter",
    label: "VCs at selected Bay Area venture firms",
    dql: `type:Person
${cityFilter}
has:image
employments.{isCurrent:true title:or("General Partner","Managing Partner","Partner","Investor","Venture Partner","Principal","Associate")}
employments.employer.name:or("Sequoia Capital","Andreessen Horowitz","a16z","Accel","Greylock","Kleiner Perkins","Founders Fund","Lightspeed Venture Partners","Bessemer Venture Partners","Benchmark","Menlo Ventures","Khosla Ventures","General Catalyst","NEA","Y Combinator")`,
  },
  {
    id: "faang_engineers_current_filter",
    label: "Big-tech engineers in selected Bay Area cities",
    dql: `type:Person
${cityFilter}
has:image
employments.{isCurrent:true title:or("Software Engineer","Senior Software Engineer","Staff Software Engineer","Principal Software Engineer","Engineering Manager","Machine Learning Engineer","Research Engineer") employer.name:or("Google","Meta","Apple","Amazon","Netflix","Microsoft","NVIDIA","OpenAI","Anthropic")}`,
  },
];

function parseEnv(text) {
  return Object.fromEntries(
    text
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter((line) => line && !line.startsWith("#"))
      .map((line) => {
        const index = line.indexOf("=");
        if (index === -1) return [line, ""];
        let value = line.slice(index + 1).trim();
        if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
          value = value.slice(1, -1);
        }
        return [line.slice(0, index).trim(), value];
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

async function countQuery(token, query) {
  const url = new URL(DQL_ENDPOINT);
  url.searchParams.set("token", token);
  url.searchParams.set("type", "query");
  url.searchParams.set("size", "0");
  url.searchParams.set("query", query);

  const response = await fetch(url, { headers: { Accept: "application/json" } });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(`DQL count failed: ${response.status} ${body.slice(0, 500)}`);
  }
  const json = await response.json();
  return Number(json.hits) || 0;
}

async function main() {
  const token = await loadToken();
  const rows = [];
  for (const bucket of buckets) {
    process.stdout.write(`Counting ${bucket.id}... `);
    const withImage = await countQuery(token, bucket.dql);
    const withoutImage = await countQuery(token, bucket.dql.replace("\nhas:image", ""));
    rows.push({ id: bucket.id, label: bucket.label, withImage, withoutImage });
    process.stdout.write(`${withImage.toLocaleString()} with image, ${withoutImage.toLocaleString()} total\n`);
  }
  console.log(JSON.stringify({ generatedAt: new Date().toISOString(), rows }, null, 2));
}

main().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});

