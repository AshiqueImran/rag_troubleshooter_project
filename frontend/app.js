const chatWindow = document.getElementById("chatWindow");
const queryInput = document.getElementById("queryInput");
const sendBtn    = document.getElementById("sendBtn");
const statusBar  = document.getElementById("statusBar");

// ── Helpers ───────────────────────────────────────────────────────────────────

function appendMessage(role, html) {
  const div = document.createElement("div");
  div.className = `message ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = html;
  div.appendChild(bubble);
  chatWindow.appendChild(div);
  chatWindow.scrollTop = chatWindow.scrollHeight;
  return bubble;
}

function setStatus(text) {
  statusBar.textContent = text;
}

function formatAnswer(data) {
  // Main answer text — preserve line breaks
  let html = data.answer.replace(/\n/g, "<br/>");

  // Confidence badge
  const confidence = data.confidence || "low";
  html += `<br/><span class="badge ${confidence}">${confidence} confidence</span>`;

  // Optional follow-up question from LLM
  if (data.follow_up) {
    html += `<div class="follow-up">💬 ${data.follow_up}</div>`;
  }

  return html;
}

// ── Send query ────────────────────────────────────────────────────────────────

async function sendQuery() {
  const query = queryInput.value.trim();
  if (!query) return;

  appendMessage("user", query);
  queryInput.value = "";
  sendBtn.disabled = true;
  setStatus("Searching knowledge base...");

  // Typing indicator
  const typingBubble = appendMessage("bot", '<span class="typing">Thinking...</span>');

  try {
    const response = await fetch("/ask", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ query }),
    });

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || "Server error");
    }

    const data = await response.json();

    // Replace typing indicator with real answer
    typingBubble.innerHTML = formatAnswer(data);

    setStatus(
      `Intent: ${data.intent} · ${data.chunks_used} chunks retrieved`
    );

  } catch (err) {
    typingBubble.innerHTML = `Sorry, something went wrong: ${err.message}`;
    setStatus("Error — check server logs.");
  } finally {
    sendBtn.disabled = false;
    queryInput.focus();
  }
}

// ── Event listeners ───────────────────────────────────────────────────────────

sendBtn.addEventListener("click", sendQuery);

queryInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendQuery();
});
