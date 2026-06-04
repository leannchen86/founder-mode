const archetypes = [
  {
    score: 71,
    title: "Finance/Operator Founder",
    anchors: [
      ["Founder", 44],
      ["VC", 39],
      ["Big-Tech", 17],
    ],
    signals: ["founder", "VC", "infrastructure", "operator"],
    neighbors: [
      ["Founder-investor hybrid", "boardroom mythology"],
      ["Product-market operator", "platform weather"],
      ["Finance-native investor", "term sheet static"],
      ["Infrastructure systems builder", "calendar full of syncs"],
    ],
    fortune:
      'The oracle detects credible founder weather. Primary aura: finance/operator founder. Secondary aura: founder-investor hybrid. Likely fake backstory: survived a strategy offsite and now describes normal software as "agentic infrastructure."',
  },
  {
    score: 68,
    title: "New York Finance To Sand Hill",
    anchors: [
      ["VC", 42],
      ["Founder", 38],
      ["Big-Tech", 20],
    ],
    signals: ["finance", "operator", "VC", "product"],
    neighbors: [
      ["Operator-turned-investor", "portfolio whisperer"],
      ["Founder-investor hybrid", "soft power"],
      ["Capital markets migrant", "spreadsheet aura"],
      ["Consumer marketplace investor", "network effects detected"],
    ],
    fortune:
      "The oracle sees imported capital markets energy trying to learn California adjectives. Primary aura: New York finance to Sand Hill. Secondary aura: operator-turned-investor. Likely fake backstory: once said liquidity, now says frontier.",
  },
  {
    score: 66,
    title: "Infra Builder With A Stealth Deck",
    anchors: [
      ["Big-Tech", 55],
      ["Founder", 36],
      ["VC", 9],
    ],
    signals: ["technical", "infrastructure", "big tech", "founder"],
    neighbors: [
      ["Infrastructure systems builder", "pager-duty residue"],
      ["Founder-curious platform person", "weekend deck"],
      ["AI research engineer", "benchmark opinions"],
      ["Product-market operator", "roadmap survivor"],
    ],
    fortune:
      "The oracle sees builder energy one coffee away from a pre-seed. Primary aura: infra builder with a stealth deck. Secondary aura: founder-curious platform person. Likely fake backstory: left a design review and accidentally incorporated.",
  },
  {
    score: 78,
    title: "AI Research Founder",
    anchors: [
      ["Founder", 61],
      ["Big-Tech", 27],
      ["VC", 12],
    ],
    signals: ["AI", "research", "technical", "infrastructure"],
    neighbors: [
      ["AI research founder", "lab-to-product arc"],
      ["Technical AI investor", "benchmark fluent"],
      ["Infrastructure founder", "GPU allocation anxiety"],
      ["Stanford research circuit", "paper-to-company pipeline"],
    ],
    fortune:
      "The oracle sees a paper becoming a company before lunch. Primary aura: AI research founder. Secondary aura: technical AI/infrastructure investor. Likely fake backstory: owns one good demo, three eval spreadsheets, and a suspiciously confident roadmap.",
  },
  {
    score: 65,
    title: "Founder-Curious Product Platform Person",
    anchors: [
      ["Big-Tech", 49],
      ["Founder", 32],
      ["VC", 19],
    ],
    signals: ["product", "platform", "operator", "marketplace"],
    neighbors: [
      ["Product-platform operator", "memo strong"],
      ["Consumer marketplace investor", "network effect hobby"],
      ["Startup generalist", "taste for chaos"],
      ["Big-tech technical lane", "scope negotiation"],
    ],
    fortune:
      "The oracle sees product sense with a mild case of founder tabs open. Primary aura: founder-curious product platform person. Secondary aura: product-market operator. Likely fake backstory: has renamed a dashboard to a command center.",
  },
  {
    score: 28,
    title: "No archetype detected",
    miss: true,
    anchors: [
      ["Founder", 14],
      ["VC", 6],
      ["Big-Tech", 9],
    ],
    signals: [],
    neighbors: [],
    fortune: "No archetype detected.",
  },
];

const camera = document.querySelector("#camera");
const viewer = document.querySelector("#viewer");
const signalCanvas = document.querySelector("#signalCanvas");
const cameraButton = document.querySelector("#cameraButton");
const analyzeButton = document.querySelector("#analyzeButton");
const shuffleButton = document.querySelector("#shuffleButton");
const signalState = document.querySelector("#signalState");
const miniScore = document.querySelector("#miniScore");
const scoreRing = document.querySelector("#scoreRing");
const scoreValue = document.querySelector("#scoreValue");
const archetypeTitle = document.querySelector("#archetypeTitle");
const mixLine = document.querySelector("#mixLine");
const auraGrid = document.querySelector("#auraGrid");
const fortuneText = document.querySelector("#fortuneText");
const fortuneBox = fortuneText.closest(".fortune");
const neighborStrip = document.querySelector("#neighborStrip");

let activeIndex = 0;
let stream = null;
let frameSignal = 0.5;

function drawSignalField() {
  const ctx = signalCanvas.getContext("2d");
  const rect = signalCanvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  signalCanvas.width = Math.max(1, Math.floor(rect.width * dpr));
  signalCanvas.height = Math.max(1, Math.floor(rect.height * dpr));
  ctx.scale(dpr, dpr);

  const w = rect.width;
  const h = rect.height;
  ctx.fillStyle = "#151820";
  ctx.fillRect(0, 0, w, h);

  ctx.strokeStyle = "rgba(255,255,255,.08)";
  ctx.lineWidth = 1;
  for (let x = 0; x < w; x += 42) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
  }
  for (let y = 0; y < h; y += 42) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  const nodes = [
    [0.18, 0.24, "#32e49a"],
    [0.74, 0.2, "#ff6b5f"],
    [0.65, 0.72, "#f6c84f"],
    [0.28, 0.78, "#55c8ff"],
    [0.5, 0.46, "#f7f8fa"],
  ];

  ctx.lineWidth = 2;
  for (let i = 0; i < nodes.length; i += 1) {
    for (let j = i + 1; j < nodes.length; j += 1) {
      ctx.strokeStyle = "rgba(255,255,255,.16)";
      ctx.beginPath();
      ctx.moveTo(nodes[i][0] * w, nodes[i][1] * h);
      ctx.lineTo(nodes[j][0] * w, nodes[j][1] * h);
      ctx.stroke();
    }
  }

  nodes.forEach(([x, y, color]) => {
    ctx.fillStyle = color;
    ctx.fillRect(x * w - 7, y * h - 7, 14, 14);
    ctx.strokeStyle = "rgba(0,0,0,.55)";
    ctx.strokeRect(x * w - 7, y * h - 7, 14, 14);
  });
}

function renderOracle(oracle) {
  const isMiss = Boolean(oracle.miss);
  const mixText = oracle.anchors
    .map(([label, value]) => `${value}% ${label === "VC" ? "VC" : label.toLowerCase()}`)
    .join(" / ");
  scoreValue.textContent = oracle.score;
  miniScore.textContent = `${oracle.score}%`;
  scoreRing.style.setProperty("--score", oracle.score);
  archetypeTitle.textContent = oracle.title;
  mixLine.textContent = mixText;
  fortuneText.textContent = `Aura mix: ${mixText}. ${oracle.fortune}`;
  signalState.textContent = isMiss ? "NO ARCHETYPE" : oracle.signals[0]?.toUpperCase() || "ORACLE";

  mixLine.hidden = isMiss;
  auraGrid.hidden = isMiss;
  fortuneBox.hidden = isMiss;
  neighborStrip.hidden = isMiss;

  auraGrid.classList.toggle("miss", isMiss);
  auraGrid.innerHTML = isMiss
    ? ""
    : oracle.signals
        .slice(0, 4)
        .map((signal) => `<div class="aura"><strong>${signal}</strong><span>signal</span></div>`)
        .join("");

  neighborStrip.innerHTML = isMiss
    ? ""
    : oracle.neighbors
        .map(([name, note]) => `<div class="neighbor"><b>${name}</b><span>${note}</span></div>`)
        .join("");
}

function chooseFromFrame() {
  const frame = Math.abs(Math.sin(frameSignal * 11.7));
  activeIndex = Math.min(archetypes.length - 1, Math.floor(frame * archetypes.length));
  renderOracle(archetypes[activeIndex]);
}

async function openCamera() {
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
    camera.srcObject = null;
    viewer.classList.remove("camera-on");
    signalState.textContent = "STANDBY";
    return;
  }

  stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false,
  });
  camera.srcObject = stream;
  await camera.play();
  viewer.classList.add("camera-on");
  signalState.textContent = "LIVE";
  sampleFrame();
}

function sampleFrame() {
  if (!stream) return;
  const sample = document.createElement("canvas");
  sample.width = 32;
  sample.height = 32;
  const ctx = sample.getContext("2d");
  ctx.drawImage(camera, 0, 0, sample.width, sample.height);
  const pixels = ctx.getImageData(0, 0, sample.width, sample.height).data;
  let total = 0;
  for (let i = 0; i < pixels.length; i += 4) {
    total += pixels[i] * 0.299 + pixels[i + 1] * 0.587 + pixels[i + 2] * 0.114;
  }
  frameSignal = total / (pixels.length / 4) / 255;
  requestAnimationFrame(sampleFrame);
}

cameraButton.addEventListener("click", () => {
  openCamera().catch(() => {
    signalState.textContent = "CAMERA BLOCKED";
  });
});

analyzeButton.addEventListener("click", chooseFromFrame);

shuffleButton.addEventListener("click", () => {
  activeIndex = (activeIndex + 1) % archetypes.length;
  renderOracle(archetypes[activeIndex]);
});

window.addEventListener("resize", drawSignalField);
drawSignalField();
renderOracle(archetypes[0]);

if (window.lucide) {
  window.lucide.createIcons();
}
